"""Environment-independent action and sensor-policy contracts, with explicit SI units."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ActionContract:
    """World XYZ increments and additive world-axis rotation-vector coordinates.

    Rotation coordinates describe R = Exp(rotvec) @ DOWN, not body-frame twists.
    These are commands to a position/force tracking stack, never direct torques.
    """

    translation_step: float = 0.004
    rotation_step: float = 0.03
    dt: float = 0.02

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Action scales and period must be finite and positive")

    @property
    def scales(self):
        return np.r_[np.full(3, self.translation_step), np.full(3, self.rotation_step)]

    def apply(self, command, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite 6-vector")
        command = np.asarray(command, dtype=float)
        if command.shape != (6,) or not np.isfinite(command).all():
            raise ValueError("Command must be a finite 6-vector")
        scales = self.scales
        result = np.clip(
            command + scales * np.clip(action, -1, 1),
            [0.395, -0.105, -0.007, -0.3, -0.3, -0.3],
            [0.605, 0.105, 0.05, 0.3, 0.3, 0.3],
        )
        return result, (result - command) / scales

    def to_dict(self):
        return {
            "version": 1,
            **asdict(self),
            "frame": "world",
            "rotation": "additive rotvec coordinates; R=Exp(rotvec)@DOWN",
            "units": ["m", "m", "m", "rad", "rad", "rad"],
            "normalized_bounds": [-1, 1],
            "command_lower": [0.395, -0.105, -0.007, -0.3, -0.3, -0.3],
            "command_upper": [0.605, 0.105, 0.05, 0.3, 0.3, 0.3],
        }


@dataclass(frozen=True)
class SensorConfig:
    """Synthetic sensor imperfections; not a calibrated hardware sensor model.

    Translation calibration errors are world-frame vectors. Reference yaw is an
    estimated paper-frame error about the nominal paper center, not moved physics.
    Drops hold the last received sample, whose acquisition time remains unchanged.
    """

    history: int = 4
    latency_steps: int = 0
    dropout: float = 0.0
    position_noise: float = 0.0
    joint_noise: float = 0.0
    velocity_noise: float = 0.0
    force_noise: float = 0.0
    force_bias: tuple = (0.0, 0.0, 0.0)
    tool_offset: tuple = (0.0, 0.0, 0.0)
    reference_offset: tuple = (0.0, 0.0, 0.0)
    reference_yaw: float = 0.0

    def __post_init__(self):
        if type(self.history) is not int or not 1 <= self.history <= 64:
            raise ValueError("Sensor history must be an integer in [1, 64]")
        if type(self.latency_steps) is not int or not 0 <= self.latency_steps <= 100:
            raise ValueError("Sensor latency_steps must be an integer in [0, 100]")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout <= 1:
            raise ValueError("Sensor dropout must be in [0, 1]")
        for name in ("position_noise", "joint_noise", "velocity_noise", "force_noise"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid sensor noise: {name}")
        for name in ("force_bias", "tool_offset", "reference_offset"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if not np.isfinite(self.reference_yaw):
            raise ValueError("Reference yaw must be finite")

    def to_dict(self):
        return asdict(self)


SENSOR_VERSION = 1
SENSOR_FEATURES = 39


@dataclass(frozen=True)
class SensorSample:
    timestamp_s: float
    pose: np.ndarray
    joints: np.ndarray
    velocities: np.ndarray
    force: np.ndarray


@dataclass(frozen=True)
class ReferenceSample:
    pose: np.ndarray
    preview: np.ndarray
    force_n: float
    drawing: bool


def sensor_features(sample, reference, command, timestamp_s, fresh, action):
    """Encode measured channels only; simulator contact/ink state has no input slot."""
    scales = action.scales
    features = np.empty(SENSOR_FEATURES, dtype=np.float32)
    features[:6] = (reference.pose - sample.pose) / scales
    features[6:12] = (command - sample.pose) / scales
    features[12:18] = (reference.preview - reference.pose) / scales
    features[18:25] = sample.joints / 3
    features[25:32] = sample.velocities / 5
    features[32:35] = sample.force
    features[35:] = (
        reference.force_n,
        float(reference.drawing),
        max(0.0, timestamp_s - sample.timestamp_s),
        float(fresh),
    )
    return features


def sensor_contract(config):
    return {
        "name": "sensor-history",
        "version": SENSOR_VERSION,
        "features_per_sample": SENSOR_FEATURES,
        "history": config.history,
        "ordering": "oldest to newest; reset repeats first sample (no pre-reset history)",
        "fields": {
            "0:6": "reference minus measured tool pose / action scales",
            "6:12": "command minus measured tool pose / action scales",
            "12:18": "three-step reference preview delta / action scales",
            "18:25": "joint angles / 3 rad",
            "25:32": "joint velocities / 5 rad/s",
            "32:35": "world tool contact force, N (ideal compensated force proxy)",
            "35": "authored target force, N",
            "36": "reference drawing flag (not sensed contact)",
            "37": "received sample age, seconds",
            "38": "new sample received this step, boolean",
        },
        "privileged_inputs": False,
    }


def classical_action(sample, reference, command, action, *, force_gain=0.02):
    """Measured-pose proportional tracking with normal-force feedback.

    No contact-center compensation or hidden bristle-state access. The force gain
    is a specified baseline setting, not claimed optimal for unknown hardware.
    """
    correction = 0.65 * (reference.pose - sample.pose)
    if reference.drawing:
        correction[2] -= force_gain * (reference.force_n - sample.force[2])
    return np.clip((reference.pose - command + correction) / action.scales, -1, 1).astype(
        np.float32
    )
