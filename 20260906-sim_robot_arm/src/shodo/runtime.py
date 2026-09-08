"""Synthetic sensor adapter and synchronous policy execution, separate from physics.

This is not a hardware runtime or a safety-certified controller. The adapter
models tool pose estimates and ideal gravity-compensated contact-force sensing;
it does not simulate a wrist transducer's inertia, filtering or calibration.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from shodo.contracts import (
    SENSOR_FEATURES,
    ActionContract,
    ReferenceSample,
    SensorConfig,
    SensorSample,
    classical_action,
    sensor_contract,
    sensor_features,
)
from shodo.env import ShodoEnv


class SensorEnv(gym.Wrapper):
    """Keep privileged rewards/diagnostics in the simulator, never in policy inputs."""

    def __init__(self, chars, config=None, sensors=None):
        super().__init__(ShodoEnv(chars=chars, config=config))
        self.sensors = sensors or SensorConfig()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (SENSOR_FEATURES * self.sensors.history,), dtype=np.float32
        )
        self.action_contract = self.env.action_contract
        self.samples = deque(maxlen=self.sensors.latency_steps + 1)
        self.features = deque(maxlen=self.sensors.history)

    def _acquire(self):
        env, cfg, rng = self.env, self.sensors, self.sensor_rng
        pose = env.pose
        pose[:3] += np.asarray(cfg.tool_offset) + rng.normal(0, cfg.position_noise, 3)
        return SensorSample(
            float(env.data.time),
            pose,
            np.r_[env.data.qpos[:6] + rng.normal(0, cfg.joint_noise, 6), 0.0],
            np.r_[env.data.qvel[:6] + rng.normal(0, cfg.velocity_noise, 6), 0.0],
            env.brush.force.copy() + cfg.force_bias + rng.normal(0, cfg.force_noise, 3),
        )

    def _prepare_references(self):
        """Calibrate the immutable authored trajectory once for this episode."""
        env, cfg = self.env, self.sensors
        poses = np.concatenate((env.path, env.tilts), axis=1)
        rotation = Rotation.from_rotvec([0, 0, cfg.reference_yaw])
        origin = np.array([env.config.paper_x, 0, env.config.paper_z])
        poses[:, :3] = rotation.apply(poses[:, :3] - origin) + origin + cfg.reference_offset
        # Left-multiply the world orientation; Exp(a)Exp(b), not a+b.
        poses[:, 3:] = (rotation * Rotation.from_rotvec(poses[:, 3:])).as_rotvec()
        poses.flags.writeable = False
        self._reference_poses = poses

    def _reference(self):
        env = self.env
        i = min(env.index, len(env.path) - 1)
        j = min(i + 3, len(env.path) - 1)
        return ReferenceSample(
            self._reference_poses[i].copy(),
            self._reference_poses[j].copy(),
            float(env.target_force[i]),
            env.stroke_ids[i] >= 0,
        )

    def _observe(self, *, reset=False):
        self.samples.append(self._acquire())
        if reset:
            self.sample = self.samples[0]
            fresh = True
        else:
            ready = len(self.samples) > self.sensors.latency_steps
            delivered = self.sensor_rng.random() >= self.sensors.dropout
            fresh = ready and delivered and self.samples[0].timestamp_s > self.sample.timestamp_s
            if fresh:
                self.sample = self.samples[0]
        self.reference = self._reference()
        self.fresh = fresh
        encoded = sensor_features(
            self.sample,
            self.reference,
            self.env.command,
            float(self.env.data.time),
            fresh,
            self.action_contract,
        )
        if reset:
            self.features.extend(encoded.copy() for _ in range(self.sensors.history))
        else:
            self.features.append(encoded)
        self.observation = np.concatenate(self.features)
        return self.observation.copy()

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        # Independent random stream: sensor settings must not change the plant draw.
        if seed is not None or not hasattr(self, "sensor_rng"):
            self.sensor_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(913,)))
        self.samples.clear()
        self.features.clear()
        self._prepare_references()
        return self._observe(reset=True), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._observe()
        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                **info,
                "sensor_age_s": float(self.env.data.time - self.sample.timestamp_s),
                "sensor_fresh": self.fresh,
            },
        )

    def expert(self):
        return classical_action(self.sample, self.reference, self.env.command, self.action_contract)

    def oracle(self):
        return self.env.expert()

    def input_channels(self):
        """Unnormalized policy-available channels at this decision boundary.

        Measurement acquisition can precede this boundary; commanded/reference
        states are current. No privileged contact geometry is exposed here.
        """
        return {
            "acquisition_time_s": np.array(self.sample.timestamp_s),
            "pose": self.sample.pose.copy(),
            "joints": self.sample.joints.copy(),
            "velocities": self.sample.velocities.copy(),
            "force": self.sample.force.copy(),
            "reference_pose": self.reference.pose.copy(),
            "reference_preview": self.reference.preview.copy(),
            "reference_force_n": np.array(self.reference.force_n),
            "reference_drawing": np.array(self.reference.drawing),
            "command": self.env.command.copy(),
            "fresh": np.array(self.fresh),
        }


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    next_observation: np.ndarray
    requested_action: np.ndarray
    applied_action: np.ndarray
    timestamp_s: float
    next_timestamp_s: float
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    expert_requested_action: np.ndarray | None = None
    expert_applied_action: np.ndarray | None = None


def execute_step(env, observation, policy, *, expert_noise=None, noise_rng=None):
    """One synchronous command transaction; no policy gets an environment argument.

    Applied action means the effective clipped Cartesian command increment, not
    executed robot displacement. Invalid actions are rejected before physics.
    Optional oracle recovery labels are computed before adding behavior noise;
    their clipped command is counterfactual, not the command sent to the robot.
    """
    if expert_noise is not None:
        if not isinstance(policy, str) or policy != "oracle":
            raise ValueError("Recovery supervision requires the oracle policy")
        if not np.isfinite(expert_noise) or expert_noise < 0 or noise_rng is None:
            raise ValueError("Expert noise must be finite and nonnegative with an explicit RNG")
    if isinstance(policy, str):
        if policy in ("expert", "classical"):
            requested = env.expert()
        elif policy == "oracle":
            requested = env.unwrapped.expert()
        elif policy == "zero":
            requested = np.zeros(6)
        else:
            raise ValueError(f"Unknown controller: {policy}")
    else:
        requested = policy(observation.copy())
    requested = np.asarray(requested, dtype=float).copy()
    expert_requested = expert_applied = None
    if expert_noise is not None:
        expert_requested = requested.copy()
        physical = env.unwrapped
        config = physical.config
        _, expert_applied = ActionContract(
            config.translation_step, config.rotation_step, config.dt
        ).apply(physical.command, expert_requested)
        requested += noise_rng.normal(0, expert_noise, 6)
    start = float(env.unwrapped.data.time)
    next_obs, reward, terminated, truncated, info = env.step(requested)
    return Transition(
        observation.copy(),
        next_obs.copy(),
        requested,
        env.unwrapped.last_applied_action.copy(),
        start,
        float(env.unwrapped.data.time),
        reward,
        terminated,
        truncated,
        info,
        expert_requested,
        expert_applied,
    )


def sensor_rollout(
    char,
    policy="classical",
    seed=7,
    config=None,
    sensors=None,
    material=None,
    recorder=None,
    camera_every=0,
    expert_noise=None,
):
    """Sensor-policy episode with privileged metrics; cameras are optional raw views."""
    from shodo.learning import rollout

    return rollout(
        char,
        policy,
        seed=seed,
        config=config,
        material=material,
        sensors=sensors or getattr(policy, "sensor_config", None) or SensorConfig(),
        recorder=recorder,
        camera_every=camera_every,
        expert_noise=expert_noise,
    )[0]


def record_episodes(
    directory,
    *,
    chars,
    episodes=1,
    seed=7,
    policy="classical",
    config=None,
    sensors=None,
    camera_every=0,
    expert_noise=None,
):
    """Collect native episodes without overwrites, optionally with oracle recovery labels.

    expert_noise=None preserves ordinary recording. A nonnegative standard
    deviation enables separate pre-action oracle targets and Gaussian behavior
    perturbations; zero collects clean labels. These labels supervise individual
    decisions, not coherent future expert chunks along the perturbed trajectory.
    """
    from shodo.artifacts import provenance
    from shodo.config import SimConfig
    from shodo.dataset import EpisodeRecorder
    from shodo.env import HISTORY_COLUMNS

    if not chars or type(episodes) is not int or episodes < 1:
        raise ValueError("Recording needs characters and a positive episode count")
    if type(seed) is not int or seed < 0:
        raise ValueError("Recording seed must be a nonnegative integer")
    if type(camera_every) is not int or camera_every < 0:
        raise ValueError("Camera cadence must be a nonnegative integer")
    if expert_noise is not None:
        if not np.isfinite(expert_noise) or expert_noise < 0:
            raise ValueError("Expert noise must be finite and nonnegative")
        if not isinstance(policy, str) or policy != "oracle":
            raise ValueError("Recovery supervision requires the oracle policy")
    sensors = sensors or getattr(policy, "sensor_config", None) or SensorConfig()
    config = config or SimConfig()
    directory = Path(directory)
    targets = [directory / f"episode-{i:06d}.npz" for i in range(episodes)]
    if any(path.exists() for path in targets):
        raise FileExistsError("Recording destination contains episodes; use a new directory")
    output = []
    for i, path in enumerate(targets):
        char = chars[i % len(chars)]
        metadata = {
            "char": char,
            "seed": seed + i,
            "controller": policy if isinstance(policy, str) else "learned",
            "checkpoint_provenance": getattr(policy, "checkpoint_provenance", None),
            "observation_contract": sensor_contract(sensors),
            "sensors": sensors.to_dict(),
            "config": config.to_dict(),
            "action_contract": ActionContract(
                config.translation_step, config.rotation_step, config.dt
            ).to_dict(),
            "applied_action_semantics": "effective clipped command increment; not robot displacement",
            "observation_time_semantics": "decision boundary; acquisition time = boundary minus sample age",
            "input_channels": {
                "acquisition_time_s": "held sensor acquisition timestamp, seconds",
                "pose": "measured world XYZ m and world-axis rotvec rad",
                "joints": "measured joint angles, rad",
                "velocities": "measured joint velocities, rad/s",
                "force": "measured world force vector, N; not a full six-axis wrench",
                "reference_pose": "estimated-reference world XYZ m and rotvec rad",
                "reference_preview": "estimated reference three controller intervals ahead",
                "reference_force_n": "authored target load, N",
                "reference_drawing": "authored drawing flag, not physical contact",
                "command": "current clipped command, world XYZ m and rotvec rad",
                "fresh": "new sensor packet received at this decision boundary",
            },
            "privileged_history_columns": HISTORY_COLUMNS,
            "privileged_time_semantics": "post-action state; target refers to the executed reference step",
            "camera": {
                "enabled": bool(camera_every),
                "every_control_steps": camera_every,
                "format": "raw perspective RGB uint8, top row first, no inset or diagnostics",
                "width": 640,
                "height": 480,
                "free_camera": {
                    "lookat": [0.3, 0, 0.22],
                    "distance": 1.35,
                    "azimuth_deg": 135,
                    "elevation_deg": -35,
                },
                "calibration": "synthetic MuJoCo free camera, not a measured camera calibration",
                "timing": "reset, cadence boundaries and terminal state; no synthetic holds",
            },
            "provenance": provenance(),
            "attribution": "KanjiVG, Ulrich Apel and contributors, CC BY-SA 3.0; transformed trajectories",
        }
        if expert_noise is not None:
            metadata["recovery_supervision"] = {
                "version": 1,
                "teacher": "privileged oracle",
                "noise_std": float(expert_noise),
                "noise_units": "normalized command increments",
                "noise_seed": [seed + i, 0x53484F44],
                "label_timing": "pre-action decision boundary",
                "requested_action_semantics": "unperturbed oracle normalized command increment",
                "behavior_action_semantics": "oracle requested increment plus independent Gaussian noise; applied_actions retains actual clipping",
                "action_semantics": "counterfactual effective clipped command increment; not executed robot displacement",
                "coherent_action_chunks": False,
            }
        recorder = EpisodeRecorder(metadata)
        metrics = sensor_rollout(
            char,
            policy,
            seed=seed + i,
            config=config,
            sensors=sensors,
            recorder=recorder,
            camera_every=camera_every,
            expert_noise=expert_noise,
        )
        recorder.metadata["metrics"] = metrics
        recorder.save(path)
        output.append(path)
        status = "TRUNCATED" if metrics["truncated"] else "Complete"
        print(
            f"[{i + 1}/{episodes}] {char}: {status}, {metrics['steps']} transitions → {path.resolve()}",
            flush=True,
        )
    return output
