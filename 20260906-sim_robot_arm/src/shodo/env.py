"""Torque-driven Panda with coupled elastic brush and wet-paper dynamics."""

from dataclasses import replace
from functools import lru_cache
from typing import ClassVar

import gymnasium as gym
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from shodo.brush import Brush
from shodo.config import SimConfig
from shodo.data import TRAIN, trajectory
from shodo.ink import Paper
from shodo.robot import DOWN, Panda

OBSERVATION_VERSION = 2
OBSERVATIONS = 40
ACTIONS = 6
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
    "q7",
    "stroke_id",
    "normal_force",
    "target_force",
    "tilt_error",
    "contact_bundles",
    "torque_fraction",
    "pigment_mass",
]


@lru_cache(maxsize=256)
def reference(char, paper_x=0.5):
    path, ids = trajectory(char)
    path = path.copy()
    path[:, 0] += paper_x - 0.32
    path = gaussian_filter1d(path, 1.2, axis=0, mode="nearest")
    velocity = np.gradient(path, axis=0)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    direction = velocity[:, :2] / np.maximum(speed[:, None], 1e-8)
    tilt = np.c_[-direction[:, 1] * 0.10, direction[:, 0] * 0.10, np.zeros(len(path))]
    tilt[ids < 0] = 0
    tilt = gaussian_filter1d(tilt, 4, axis=0, mode="nearest")
    return path, ids, tilt


class ShodoEnv(gym.Env):
    metadata: ClassVar = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, chars=TRAIN, render_mode=None, config=None):
        super().__init__()
        config = config or SimConfig()
        if not chars or render_mode not in (None, "rgb_array"):
            raise ValueError("Expected characters and optional rgb_array render mode")
        self.chars, self.render_mode, self.config = chars, render_mode, config
        self.robot = Panda(config.timestep)
        self.model, self.data = self.robot.model, self.robot.data
        self.action_space = gym.spaces.Box(-1, 1, (ACTIONS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATIONS,), dtype=np.float32)
        self.renderer = None
        self.scales = np.r_[np.full(3, config.translation_step), np.full(3, config.rotation_step)]

    @property
    def tip(self):
        return self.robot.tip.copy()

    @property
    def pose(self):
        return np.r_[self.tip, Rotation.from_matrix(self.robot.rotation @ DOWN.T).as_rotvec()]

    def _obs(self):
        i = min(self.index, len(self.path) - 1)
        target = np.r_[self.path[i], self.tilts[i]]
        preview = np.r_[
            self.path[min(i + 3, len(self.path) - 1)], self.tilts[min(i + 3, len(self.path) - 1)]
        ]
        pose = self.pose
        return np.r_[
            (target - pose) / self.scales,
            (self.command - pose) / self.scales,
            (preview - target) / self.scales,
            self.data.qpos / 3,
            self.data.qvel / 5,
            self.brush.force,
            self.brush.deflection / 0.004,
            self.target_force[i],
            self.brush.touching.mean(),
            self.tip[2] / 0.025,
        ].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.char = (options or {}).get("char", self.np_random.choice(list(self.chars)))
        self.path, self.stroke_ids, self.tilts = reference(self.char, self.config.paper_x)
        cfg = self.config.brush
        if self.config.randomize:
            cfg = replace(
                cfg,
                normal_stiffness=cfg.normal_stiffness * self.np_random.uniform(0.8, 1.2),
                friction=cfg.friction * self.np_random.uniform(0.8, 1.2),
            )
        self.brush = Brush(cfg)
        depth = -self.path[:, 2, None] - self.brush.radial[None, :] ** 2 * 0.0012
        self.target_force = self.config.brush.normal_stiffness * np.maximum(depth, 0).mean(axis=1)
        self.index = 0
        self.command = np.r_[self.path[0], self.tilts[0]]
        self.robot.reset(self.command[:3], self.command[3:])
        self.brush.update(self.tip, self.robot.rotation, self.config.timestep)
        self.paper = Paper(
            self.config.ink, self.config.paper_x, seed=int(self.np_random.integers(2**31))
        )
        self.history = []
        self.done = False
        return self._obs(), {"char": self.char}

    def expert(self):
        i = min(self.index, len(self.path) - 1)
        target = np.r_[self.path[i], self.tilts[i]]
        correction = 0.65 * (target - self.pose)
        if self.stroke_ids[i] >= 0:
            correction[2] -= 0.0008 * (self.target_force[i] - self.brush.force[2])
        return np.clip((target - self.command + correction) / self.scales, -1, 1).astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; reset before stepping")
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTIONS,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 6-vector")
        action = np.clip(action, -1, 1)
        self.command = np.clip(
            self.command + self.scales * action,
            [0.395, -0.105, -0.007, -0.3, -0.3, -0.3],
            [0.605, 0.105, 0.05, 0.3, 0.3, 0.3],
        )
        self.robot.q_target = self.robot.inverse(self.command[:3], self.command[3:])
        cfg = self.config
        for j in range(cfg.substeps):
            force, torque = self.brush.update(
                self.tip, self.robot.rotation, cfg.timestep, cfg.paper_z
            )
            self.robot.step(force, torque)
            if j % 5 == 4:
                load = self.brush.normal
                if load.sum() > 1e-6:
                    wet = min(1.0, load.sum() / 0.3)
                    self.paper.deposit(
                        self.brush.contact,
                        load,
                        cfg.brush.water_flow * cfg.timestep * 5 * wet,
                        cfg.brush.pigment_flow * cfg.timestep * 5 * wet,
                    )
        self.paper.advance(cfg.dt)
        error = float(np.linalg.norm(self.tip - self.path[self.index]))
        angular = float(np.linalg.norm(self.pose[3:] - self.tilts[self.index]))
        force_error = float(self.brush.force[2] - self.target_force[self.index])
        torque_fraction = float(
            np.max(np.abs(self.data.ctrl) / self.model.actuator_ctrlrange[:, 1])
        )
        reward = float(
            0.65 * np.exp(-((error / 0.008) ** 2))
            + 0.15 * np.exp(-((angular / 0.15) ** 2))
            + 0.2 * np.exp(-((force_error / 0.3) ** 2))
            - 0.002 * np.square(action).sum()
        )
        if cfg.record:
            self.history.append(
                np.r_[
                    self.tip,
                    self.path[self.index],
                    self.data.qpos,
                    self.stroke_ids[self.index],
                    self.brush.force[2],
                    self.target_force[self.index],
                    angular,
                    self.brush.touching.sum(),
                    torque_fraction,
                    self.paper.mobile.sum() + self.paper.fixed.sum(),
                ]
            )
        self.index += 1
        terminated = self.index == len(self.path)
        truncated = bool(
            not np.isfinite(self.data.qpos).all()
            or np.linalg.norm(self.data.qvel) > 100
            or self.brush.force[2] > 8.0
        )
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
