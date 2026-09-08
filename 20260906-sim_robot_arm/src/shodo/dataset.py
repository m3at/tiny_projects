"""Lossless, timestamped policy episodes, independent of presentation recordings.

Each action row connects observation i to observation i + 1. Privileged channels
are never concatenated into policy inputs. History describes the resulting state;
optional expert labels describe the pre-action decision, as declared in metadata.
Optional named input channels retain raw measurements and references at all T + 1
decision boundaries, before normalization and history stacking.
Camera frames retain their own acquisition timestamps, without synthetic holds.
This is a native NPZ schema, not a LeRobot-compatible dataset.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
_TRANSITION_KEYS = ("requested_actions", "applied_actions", "rewards", "terminated", "truncated")


def _numeric(value, name, shape=None, *, boolean=False, copy=True):
    array = np.asarray(value)
    if array.dtype.kind not in ("fiub" if boolean else "fiu") or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    return array.copy() if copy else array


def _channels(values, label, previous=None, *, boolean=False):
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError(f"{label} channels must be a dictionary")
    channels = {}
    for name, value in values.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"{label} names must be simple identifiers")
        channels[name] = _numeric(value, f"{label} {name}", boolean=boolean)
    if previous is not None and (
        channels.keys() != previous.keys()
        or any(channels[name].shape != previous[name].shape for name in channels)
    ):
        raise ValueError(f"{label} keys and shapes must remain constant")
    return channels


def _time(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("timestamp_s must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class EpisodeDataset:
    """Loaded episode arrays and JSON metadata; windows never cross an episode boundary."""

    arrays: dict
    metadata: dict

    def __len__(self):
        return len(self.arrays["requested_actions"])

    def window(self, start, length):
        """Return copied arrays for consecutive transitions plus their endpoint observation.

        Camera selection includes both boundary timestamps. Adjacent windows can therefore
        share a boundary frame, just as they share an endpoint observation.
        """
        if not isinstance(start, int) or not isinstance(length, int):
            raise TypeError("window indices must be integers")
        if start < 0 or length <= 0 or start + length > len(self):
            raise ValueError("window must fit inside this episode")
        stop = start + length
        result = {}
        for name, array in self.arrays.items():
            if name in ("observations", "observation_times_s") or name.startswith("input_"):
                result[name] = array[start : stop + 1].copy()
            elif name in _TRANSITION_KEYS or name.startswith("privileged_"):
                result[name] = array[start:stop].copy()
        if "camera_frames" in self.arrays:
            times = self.arrays["camera_times_s"]
            bounds = self.arrays["observation_times_s"][[start, stop]]
            mask = (times >= bounds[0]) & (times <= bounds[1])
            result["camera_frames"] = self.arrays["camera_frames"][mask].copy()
            result["camera_times_s"] = times[mask].copy()
        return result


class EpisodeRecorder:
    """In-memory recorder for one episode, published atomically as a single NPZ file.

    Metadata should declare observation/action contracts, units, coordinate frames,
    configuration, calibration, provenance and source attribution. The caller owns
    these semantics; this module validates alignment and storage, not sensor fidelity.
    Camera storage is lossless but memory grows with the complete episode.
    """

    def __init__(self, metadata):
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a JSON object")
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self._observations = []
        self._inputs = []
        self._times = []
        self._rows = []
        self._frames = []
        self._frame_times = []

    def start(self, observation, timestamp_s=0.0, inputs=None):
        """Record the initial decision and optional raw, nonprivileged input channels."""
        if self._observations:
            raise ValueError("episode already started")
        observation = _numeric(observation, "observation")
        if observation.ndim != 1 or not observation.size:
            raise ValueError("observation must be a nonempty vector")
        timestamp_s = _time(timestamp_s)
        inputs = _channels(inputs, "input", boolean=True)
        self._observations.append(observation)
        self._inputs.append(inputs)
        self._times.append(timestamp_s)

    def append(
        self,
        requested_action,
        applied_action,
        next_observation,
        *,
        timestamp_s,
        reward=0.0,
        terminated=False,
        truncated=False,
        privileged=None,
        inputs=None,
    ):
        """Record a transition; inputs describe next_observation's decision boundary."""
        if not self._observations:
            raise ValueError("start the episode before recording transitions")
        if self._rows and (self._rows[-1][3] or self._rows[-1][4]):
            raise ValueError("cannot append after episode termination or truncation")
        timestamp_s = _time(timestamp_s)
        if timestamp_s <= self._times[-1]:
            raise ValueError("observation timestamps must strictly increase")
        requested = _numeric(requested_action, "requested_action")
        if requested.ndim != 1 or not requested.size:
            raise ValueError("requested_action must be a nonempty vector")
        if self._rows and requested.shape != self._rows[0][0].shape:
            raise ValueError("action shape must remain constant")
        applied = _numeric(applied_action, "applied_action", requested.shape)
        observation = _numeric(next_observation, "next_observation", self._observations[0].shape)
        reward = float(_numeric(reward, "reward", ()))
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise TypeError("terminated and truncated must be booleans")
        diagnostics = _channels(
            privileged, "privileged diagnostic", self._rows[0][5] if self._rows else None
        )
        inputs = _channels(inputs, "input", self._inputs[0], boolean=True)
        self._rows.append((requested, applied, reward, terminated, truncated, diagnostics))
        self._observations.append(observation)
        self._inputs.append(inputs)
        self._times.append(timestamp_s)

    def add_camera_frame(self, frame, *, timestamp_s):
        if not self._observations:
            raise ValueError("start the episode before recording camera frames")
        timestamp_s = _time(timestamp_s)
        if timestamp_s < self._times[0] or (
            self._frame_times and timestamp_s <= self._frame_times[-1]
        ):
            raise ValueError("camera timestamps must increase within the episode")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("camera frame must be a nonempty uint8 H x W x 3 RGB array")
        if self._frames and frame.shape != self._frames[0].shape:
            raise ValueError("camera frame shape must remain constant")
        self._frames.append(frame.copy())
        self._frame_times.append(timestamp_s)

    def save(self, path):
        """Publish a complete or explicitly incomplete episode; never replace an existing file."""
        if not self._rows:
            raise ValueError("an episode must contain at least one transition")
        arrays = {
            "observations": np.stack(self._observations),
            "observation_times_s": np.asarray(self._times, dtype=np.float64),
            **{
                f"input_{name}": np.stack([row[name] for row in self._inputs])
                for name in self._inputs[0]
            },
            **{
                name: np.asarray([row[i] for row in self._rows])
                for i, name in enumerate(_TRANSITION_KEYS)
            },
            **{
                f"privileged_{name}": np.stack([row[5][name] for row in self._rows])
                for name in self._rows[0][5]
            },
        }
        if self._frames:
            arrays["camera_frames"] = np.stack(self._frames)
            arrays["camera_times_s"] = np.asarray(self._frame_times, dtype=np.float64)
        metadata = {
            **self.metadata,
            "schema": "shodo-policy-episode",
            "schema_version": SCHEMA_VERSION,
            "transition_count": len(self._rows),
            "complete": bool(self._rows[-1][3] or self._rows[-1][4]),
            "alignment": "action[i] connects observation[i] to observation[i+1]",
            "timestamp_unit": "seconds",
        }
        _validate(arrays, metadata)
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("episode path must have .npz suffix")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
                temporary = Path(f.name)
                np.savez_compressed(f, metadata_json=np.array(json.dumps(metadata)), **arrays)
                f.flush()
                os.fsync(f.fileno())
            # A hard link publishes atomically and fails if the destination already exists.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path


def _validate(arrays, metadata):
    if metadata.get("schema") != "shodo-policy-episode" or metadata.get("schema_version") != 1:
        raise ValueError("unsupported episode schema")
    required = {"observations", "observation_times_s", *_TRANSITION_KEYS}
    if not required.issubset(arrays):
        raise ValueError("episode is missing required arrays")
    observations = _numeric(arrays["observations"], "observations", copy=False)
    count = len(observations) - 1
    if observations.ndim != 2 or count < 1 or not observations.shape[1]:
        raise ValueError("observations must have shape [transitions + 1, features]")
    times = _numeric(arrays["observation_times_s"], "observation_times_s", (count + 1,), copy=False)
    if times[0] < 0 or not (np.diff(times) > 0).all():
        raise ValueError("observation timestamps must be nonnegative and strictly increasing")
    requested = _numeric(arrays["requested_actions"], "requested_actions", copy=False)
    if requested.ndim != 2 or requested.shape[0] != count or not requested.shape[1]:
        raise ValueError("requested_actions must have shape [transitions, actions]")
    _numeric(arrays["applied_actions"], "applied_actions", requested.shape, copy=False)
    _numeric(arrays["rewards"], "rewards", (count,), copy=False)
    for key in ("terminated", "truncated"):
        if arrays[key].shape != (count,) or arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} must be a boolean transition vector")
        if arrays[key][:-1].any():
            raise ValueError("transitions after termination or truncation are invalid")
    if metadata.get("transition_count") != count or metadata.get("complete") != bool(
        arrays["terminated"][-1] or arrays["truncated"][-1]
    ):
        raise ValueError("episode metadata disagrees with transition arrays")
    for name, array in arrays.items():
        if name.startswith("input_"):
            if not re.fullmatch(r"input_[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError("input names must be simple identifiers")
            _numeric(array, name, boolean=True, copy=False)
            if array.ndim < 1 or array.shape[0] != count + 1:
                raise ValueError("input channels must align with observation decisions")
        elif name.startswith("privileged_"):
            _numeric(array, name, copy=False)
            if array.ndim < 1 or array.shape[0] != count:
                raise ValueError("privileged diagnostics must align with transitions")
        elif name not in required | {"camera_frames", "camera_times_s"}:
            raise ValueError(f"unknown episode array: {name}")
    if ("camera_frames" in arrays) != ("camera_times_s" in arrays):
        raise ValueError("camera frames and timestamps must both be present")
    if "camera_frames" in arrays:
        frames = arrays["camera_frames"]
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("camera_frames must be uint8 [frames, height, width, 3]")
        if not frames.size:
            raise ValueError("camera_frames must not be empty")
        camera_times = _numeric(
            arrays["camera_times_s"], "camera_times_s", (len(frames),), copy=False
        )
        if (
            camera_times[0] < times[0]
            or camera_times[-1] > times[-1]
            or not (np.diff(camera_times) > 0).all()
        ):
            raise ValueError("camera timestamps must increase within episode bounds")


def load_episode(path):
    """Load and validate a native episode without permitting pickled objects."""
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError("episode is missing JSON metadata")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(metadata, dict):
            raise TypeError("episode metadata must be a JSON object")
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
    _validate(arrays, metadata)
    return EpisodeDataset(arrays, metadata)
