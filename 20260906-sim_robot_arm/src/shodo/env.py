"""Torque-driven RebotArm with coupled elastic brush and wet-paper dynamics."""

from dataclasses import replace
from functools import lru_cache
from typing import ClassVar

import gymnasium as gym
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from shodo.brush import Brush
from shodo.config import SimConfig
from shodo.contracts import ActionContract
from shodo.data import TRAIN, trajectory
from shodo.ink import Paper
from shodo.robot import DOWN, RebotArm

OBSERVATION_VERSION = 3
OBSERVATIONS = 40
ACTIONS = 6
INK_LOAD_THRESHOLD = 1e-6  # N; unloaded geometric contact does not transfer ink.
HISTORY_COLUMNS = [
    "tip_x",
    "tip_y",
    "tip_z",
    "target_x",
    "target_y",
    "target_z",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "reserved_joint_slot",
    "stroke_id",
    "normal_force",
    "target_force",
    "tilt_error",
    "contact_bundles",
    "torque_fraction",
    "pigment_mass",
    "ink_x",
    "ink_y",
    *[f"velocity_j{i}_rad_s" for i in range(1, 7)],
    *[f"command_j{i}_rad" for i in range(1, 7)],
    *[f"command_velocity_j{i}_rad_s" for i in range(1, 7)],
    *[f"torque_j{i}_nm" for i in range(1, 7)],
]


@lru_cache(maxsize=256)
def reference(char, paper_x=0.5, touchdown_speed=0.04):
    """Read-only world XYZ, stroke IDs and world-axis tilt rotation vectors."""
    path, ids = trajectory(char, touchdown_speed)
    path = path.copy()
    path[:, 0] += paper_x
    path = gaussian_filter1d(path, 1.2, axis=0, mode="nearest")
    tilt = np.zeros_like(path)  # Rigid brush remains vertical throughout writing and lifts.
    path.flags.writeable = tilt.flags.writeable = False
    return path, ids, tilt


class ShodoEnv(gym.Env):
    metadata: ClassVar = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, chars=TRAIN, render_mode=None, config=None):
        super().__init__()
        config = config or SimConfig()
        if not chars or render_mode not in (None, "rgb_array"):
            raise ValueError("Expected characters and optional rgb_array render mode")
        self.chars, self.render_mode, self.config = chars, render_mode, config
        self.robot = RebotArm(config.timestep, config.brush, config.robot)
        self.model, self.data = self.robot.model, self.robot.data
        self.action_space = gym.spaces.Box(-1, 1, (ACTIONS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATIONS,), dtype=np.float32)
        self.renderer = None
        self.zero = np.zeros(3)
        self.action_contract = ActionContract(
            config.translation_step, config.rotation_step, config.dt
        )
        self.scales = self.action_contract.scales

    @property
    def tip(self):
        return self.robot.tip.copy()

    @property
    def pose(self):
        return np.r_[self.tip, Rotation.from_matrix(self.robot.rotation @ DOWN.T).as_rotvec()]

    @property
    def tracking_pose(self):
        pose = self.pose
        if (
            self.stroke_ids[min(self.index, len(self.path) - 1)] >= 0
            and self.brush.ink_loads.sum() > INK_LOAD_THRESHOLD
        ):
            pose[:2] = np.average(
                self.brush.ink_positions[:, :2], axis=0, weights=self.brush.ink_loads
            )
        return pose

    def _obs(self):
        i = min(self.index, len(self.path) - 1)
        target = np.r_[self.path[i], self.tilts[i]]
        preview = np.r_[
            self.path[min(i + 3, len(self.path) - 1)], self.tilts[min(i + 3, len(self.path) - 1)]
        ]
        pose = self.pose
        return np.r_[
            (target - self.tracking_pose) / self.scales,
            (self.command - pose) / self.scales,
            (preview - target) / self.scales,
            self.robot.joint_positions / 3,
            self.robot.joint_velocities / 5,
            self.brush.force,
            self.brush.deflection / 0.004,
            self.target_force[i],
            self.brush.touching.mean(),
            self.tip[2] / 0.025,
        ].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.char = (options or {}).get("char", self.np_random.choice(list(self.chars)))
        self.path, self.stroke_ids, self.tilts = reference(
            self.char, self.config.paper_x, self.config.touchdown_speed
        )
        cfg = self.config.brush
        if self.config.randomize:
            variation = self.config.material_variation
            stiffness_scale = self.np_random.uniform(1 - variation, 1 + variation)
            cfg = replace(
                cfg,
                normal_stiffness=cfg.normal_stiffness
                * (stiffness_scale if cfg.backend == "reduced" else 1),
                friction=cfg.friction * self.np_random.uniform(1 - variation, 1 + variation),
            )
        material = (options or {}).get("material", {})
        allowed = {"friction"} if cfg.backend == "cable" else {"friction", "normal_stiffness"}
        if set(material) - allowed:
            raise ValueError(f"Runtime material overrides must be drawn from {sorted(allowed)}")
        cfg = replace(cfg, **material)
        if cfg.backend == "cable":
            from shodo.cable import CableBrush

            self.brush = CableBrush(self.robot, cfg)
            self.model.geom_friction[list(self.brush.geom_bundle), 0] = cfg.friction
        else:
            self.brush = Brush(cfg)
        depth = -self.path[:, 2, None] - self.brush.radial[None, :] ** 2 * 0.0012
        self.target_force = self.config.brush.normal_stiffness * np.maximum(depth, 0).mean(axis=1)
        self.index = 0
        self.command = np.r_[self.path[0], self.tilts[0]]
        self.last_requested_action = np.zeros(ACTIONS)
        self.last_applied_action = np.zeros(ACTIONS)
        self.robot.reset(self.command[:3], self.command[3:])
        self.brush.update(self.tip, self.robot.rotation, self.config.timestep)
        self.paper = Paper(
            self.config.ink, self.config.paper_x, seed=int(self.np_random.integers(2**31))
        )
        self.history = []
        self.ink_clock = 0.0
        self.peak_force = 0.0
        self.force_impulse = np.zeros(3)
        self.peak_torque_fraction = 0.0
        self.peak_joint_speed = np.zeros(6)
        self.peak_joint_torque = np.zeros(6)
        self.forbidden_contact_steps = 0
        self.min_joint_margin = np.inf
        self.done = False
        return self._obs(), {"char": self.char}

    def expert(self):
        i = min(self.index, len(self.path) - 1)
        target = np.r_[self.path[i], self.tilts[i]]
        pose = self.pose
        tracked = self.tracking_pose if self.config.compensate_brush else pose
        desired = target.copy()
        desired[:2] -= np.clip(tracked[:2] - pose[:2], -0.02, 0.02)
        correction = 0.65 * (target - tracked)
        if self.stroke_ids[i] >= 0:
            correction[2] -= self.config.force_feedback_gain * (
                self.target_force[i] - self.brush.force[2]
            )
        return np.clip((desired - self.command + correction) / self.scales, -1, 1).astype(
            np.float32
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; reset before stepping")
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTIONS,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 6-vector")
        self.last_requested_action = action.copy()
        self.command, self.last_applied_action = self.action_contract.apply(self.command, action)
        action = np.clip(action, -1, 1)
        self.robot.set_target(self.robot.inverse(self.command[:3], self.command[3:]))
        cfg = self.config
        deposits, masses = [], []
        for _ in range(cfg.substeps):
            if self.brush.native:
                self.robot.step(self.zero, self.zero)
                self.brush.update(self.tip, self.robot.rotation, cfg.timestep, cfg.paper_z)
            else:
                force, torque = self.brush.update(
                    self.tip, self.robot.rotation, cfg.timestep, cfg.paper_z
                )
                self.robot.step(force, torque)
            self.peak_joint_speed = np.maximum(self.peak_joint_speed, np.abs(self.data.qvel[:6]))
            self.peak_joint_torque = np.maximum(
                self.peak_joint_torque, np.abs(self.data.actuator_force)
            )
            self.forbidden_contact_steps += int(self.robot.forbidden_contacts() > 0)
            self.min_joint_margin = min(
                self.min_joint_margin,
                float(
                    np.min(
                        np.minimum(
                            self.data.qpos[:6] - self.model.jnt_range[:6, 0],
                            self.model.jnt_range[:6, 1] - self.data.qpos[:6],
                        )
                    )
                ),
            )
            self.peak_force = max(self.peak_force, float(self.brush.force[2]))
            if cfg.record:
                self.force_impulse += self.brush.force * cfg.timestep
                self.peak_torque_fraction = max(
                    self.peak_torque_fraction,
                    float(
                        np.max(
                            np.abs(self.data.actuator_force) / self.model.actuator_forcerange[:, 1]
                        )
                    ),
                )
            load = self.brush.ink_loads
            total = load.sum()
            if total > INK_LOAD_THRESHOLD:
                wet = min(1.0, total / cfg.brush.transfer_load)
                deposits.append(self.brush.ink_positions.copy())
                masses.append(load * (cfg.timestep * wet / total))
        if masses:
            weights = np.concatenate(masses)
            supplied_time = weights.sum()
            self.paper.deposit(
                np.concatenate(deposits),
                weights,
                cfg.brush.water_flow * supplied_time,
                cfg.brush.pigment_flow * supplied_time,
            )
        self.ink_clock += cfg.dt
        truncated = bool(
            not np.isfinite(self.data.qpos).all()
            or np.max(self.peak_joint_speed) > 1.5 * cfg.robot.joint_speed
            or self.forbidden_contact_steps > 0
            or self.min_joint_margin < 0
            or self.peak_force > 2.0
            or (self.data.warning.number > 0).any()
        )
        if (
            self.ink_clock >= cfg.ink.transport_dt - 1e-12
            or self.index == len(self.path) - 1
            or truncated
        ):
            self.paper.advance(self.ink_clock)
            self.ink_clock = 0.0
        tracked = self.tracking_pose
        error = float(np.linalg.norm(tracked[:3] - self.path[self.index]))
        angular = float(np.linalg.norm(self.pose[3:] - self.tilts[self.index]))
        force_error = float(self.brush.force[2] - self.target_force[self.index])
        torque_fraction = float(
            np.max(np.abs(self.data.actuator_force) / self.model.actuator_forcerange[:, 1])
        )
        reward = float(
            0.65 * np.exp(-((error / 0.008) ** 2))
            + 0.15 * np.exp(-((angular / 0.15) ** 2))
            + 0.2 * np.exp(-((force_error / cfg.force_tolerance) ** 2))
            - 0.002 * np.square(action).sum()
        )
        if cfg.record:
            self.history.append(
                np.r_[
                    self.tip,
                    self.path[self.index],
                    self.robot.joint_positions,
                    self.stroke_ids[self.index],
                    self.brush.force[2],
                    self.target_force[self.index],
                    angular,
                    self.brush.touching.sum(),
                    torque_fraction,
                    self.paper.mobile.sum() + self.paper.fixed.sum(),
                    tracked[:2],
                    self.data.qvel[:6],
                    self.robot.q_command,
                    self.robot.v_command,
                    self.data.actuator_force,
                ]
            )
        self.index += 1
        terminated = self.index == len(self.path)
        self.done = terminated or truncated
        return (
            self._obs(),
            reward,
            terminated,
            truncated,
            {
                "error_m": error,
                "char": self.char,
                "force_n": float(self.brush.force[2]),
                "force_error_n": force_error,
                "orientation_error_rad": angular,
            },
        )

    def render(self):
        if self.renderer is None:
            from shodo.rendering import Renderer

            self.renderer = Renderer(self.model)
        return self.renderer.frame(self)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
