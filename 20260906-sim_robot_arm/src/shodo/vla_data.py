"""Causal, memory-mapped visuomotor training data derived from native episodes.

This is a Shodo prepared schema, not LeRobotDataset. Only sensor observations,
raw camera frames and effective command increments enter the learning samples.
"""

import bisect
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from filelock import FileLock
from PIL import Image
from torch.utils.data import Dataset

from shodo.contracts import ActionContract, SensorConfig, sensor_contract
from shodo.data import TEST
from shodo.dataset import load_episode

STATE_INDICES = tuple(range(25)) + tuple(range(32, 39))
STATE_CONTRACT = {
    "name": "shodo-vla-sensor-state",
    "version": 1,
    "features": 32,
    "source": "latest sensor-history version 1 slice; joint velocities omitted",
    "source_indices": list(STATE_INDICES),
    "privileged_inputs": False,
}
SCHEMA = "shodo-vla-prepared"
STD_FLOOR = 1e-6
NORMALIZATION_CONTRACT = {
    "method": "mean_std",
    "fit": "all action-decision rows, excluding padded targets",
    "state_min_std": STD_FLOOR,
    "state_constant_std": 1.0,
    "state_rule": "use 1.0 when population std < state_min_std; otherwise use population std",
    "action_min_std": STD_FLOOR,
    "action_rule": "maximum of population std and action_min_std",
}


def _image_contract(image_size):
    return {
        "size": image_size,
        "resize": "bilinear, aspect preserving, round dimensions, centered black padding",
        "storage": "uint8 CHW RGB",
        "model_input": "float32 RGB divided by 255",
        "selection": "latest acquisition timestamp <= action decision timestamp",
    }


def validate_manifest_contract(manifest):
    """Validate shared training/inference semantics without opening training arrays.

    Adapter artifacts embed this manifest so inference can reject unsupported or
    malformed observations, actions, image preprocessing and normalization early.
    """
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("version") not in (1, 2)
    ):
        raise ValueError("Unsupported prepared dataset schema")
    if manifest.get("state_contract") != STATE_CONTRACT:
        raise ValueError("Unsupported prepared state contract")
    _contracts(manifest)
    if manifest.get("stats_std_floor") != STD_FLOOR:
        raise ValueError("Unsupported normalization floor")
    if manifest["version"] == 2 and manifest.get("normalization") != NORMALIZATION_CONTRACT:
        raise ValueError("Unsupported normalization contract")
    if manifest["version"] == 2 and manifest.get("supervision") not in (
        _supervision_contract("applied"),
        _supervision_contract("expert"),
    ):
        raise ValueError("Unsupported action supervision contract")
    try:
        size = manifest["image_preprocessing"]["size"]
        if type(size) is not int or size < 1:
            raise ValueError("Invalid prepared image size")
        if manifest["image_preprocessing"] != _image_contract(size):
            raise ValueError("Unsupported prepared image preprocessing contract")
        for key, width in (("state", 32), ("actions", 6)):
            for statistic in ("mean", "std"):
                value = np.asarray(manifest["stats"][key][statistic], dtype=np.float32)
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError("Invalid normalization statistics")
                if statistic == "std" and np.any(value <= 0):
                    raise ValueError("Normalization standard deviations must be positive")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Missing or invalid prepared preprocessing contract") from exc


def state_from_observation(observation, history):
    observation = np.asarray(observation)
    if (
        type(history) is not int
        or history < 1
        or observation.shape[-1:] != (39 * history,)
        or not np.isfinite(observation).all()
    ):
        raise ValueError("Expected finite sensor-history observation with matching history")
    return observation[..., -39:][..., STATE_INDICES].astype(np.float32)


def preprocess_image(frame, image_size=256):
    """RGB -> aspect-preserving bilinear resize and centered black square padding."""
    frame = np.asarray(frame)
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Expected nonempty uint8 RGB camera frame")
    height, width = frame.shape[:2]
    scale = image_size / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size))
    canvas.paste(resized, ((image_size - size[0]) // 2, (image_size - size[1]) // 2))
    return np.asarray(canvas).transpose(2, 0, 1).copy()


def _contracts(metadata):
    try:
        sensors = SensorConfig(**metadata["sensors"])
        observation = sensor_contract(sensors)
        config = metadata["config"]
        action = ActionContract(
            config["translation_step"],
            config["rotation_step"],
            config["timestep"] * config["substeps"],
        ).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid native sensor/action configuration") from exc
    if metadata.get("observation_contract") != observation:
        raise ValueError("Native observation contract must be canonical sensor-history version 1")
    if metadata.get("action_contract") != action:
        raise ValueError("Native action contract must match its canonical configuration")
    return observation, action


def _supervision_contract(supervision):
    return {
        "source": supervision,
        "array": "applied_actions"
        if supervision == "applied"
        else "privileged_expert_applied_action",
        "coherent_action_chunks": supervision == "applied",
        "semantics": "executed effective command increments"
        if supervision == "applied"
        else "pre-decision counterfactual expert effective command increments; labels only, never actor inputs",
    }


def prepare_dataset(
    episode_paths, output, image_size=256, *, allow_heldout=False, supervision="applied"
):
    """Publish validated training arrays atomically; never replace a prepared dataset."""
    paths = [Path(path) for path in episode_paths]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Provide a nonempty set of distinct episode paths")
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if supervision not in ("applied", "expert"):
        raise ValueError("supervision must be applied or expert")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            raise FileExistsError(output)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            manifest = _prepare(paths, temporary, image_size, allow_heldout, supervision)
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
            if output.exists():
                raise FileExistsError(output)
            os.rename(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return output


def _prepare(paths, directory, image_size, allow_heldout, supervision):
    manifest = {
        "schema": SCHEMA,
        "version": 2,
        "state_contract": STATE_CONTRACT,
        "allow_heldout": bool(allow_heldout),
        "supervision": _supervision_contract(supervision),
        "image_preprocessing": _image_contract(image_size),
        "action_semantics": "effective clipped normalized command increments, not displacement",
        "episodes": [],
    }
    sums = {"state": np.zeros(32), "actions": np.zeros(6)}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    count = 0
    for index, path in enumerate(paths):
        episode = load_episode(path)
        metadata, arrays = episode.metadata, episode.arrays
        observation, action = _contracts(metadata)
        char = metadata.get("char")
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError("Each episode needs a single character label")
        if char in TEST and not allow_heldout:
            raise ValueError(f"Held-out character {char} is forbidden in training data")
        shared = {
            "observation_contract": observation,
            "action_contract": action,
            "sensors": metadata["sensors"],
            "config": metadata["config"],
        }
        if index == 0:
            manifest.update(shared)
        elif any(manifest[key] != value for key, value in shared.items()):
            raise ValueError("All episodes must have matching configurations and contracts")
        if "camera_frames" not in arrays or not len(arrays["camera_frames"]):
            raise ValueError("Visuomotor data requires raw RGB camera frames")
        state = state_from_observation(arrays["observations"][:-1], observation["history"])
        if supervision == "expert":
            recovery = metadata.get("recovery_supervision", {})
            if (
                recovery.get("version") != 1
                or recovery.get("teacher") != "privileged oracle"
                or recovery.get("label_timing") != "pre-action decision boundary"
                or recovery.get("action_semantics")
                != "counterfactual effective clipped command increment; not executed robot displacement"
                or recovery.get("coherent_action_chunks") is not False
                or "privileged_expert_applied_action" not in arrays
            ):
                raise ValueError(
                    "Expert supervision requires declared pre-decision counterfactual labels"
                )
        actions = np.asarray(arrays[manifest["supervision"]["array"]], dtype=np.float32)
        if actions.shape != (len(episode), 6) or not np.isfinite(actions).all():
            raise ValueError("Applied actions must have shape [T, 6] and be finite")
        if np.any(np.abs(actions) > 1 + 1e-6):
            raise ValueError("Applied actions exceed normalized command bounds")
        actions = np.clip(actions, -1, 1)
        camera_index = (
            np.searchsorted(
                arrays["camera_times_s"], arrays["observation_times_s"][:-1], side="right"
            )
            - 1
        )
        if np.any(camera_index < 0):
            raise ValueError("Every decision requires a camera frame acquired at or before it")
        folder = directory / f"episode-{index:06d}"
        folder.mkdir()
        for name, value in {
            "state": state,
            "actions": actions,
            "camera_index": camera_index.astype(np.int64),
            "decision_times_s": arrays["observation_times_s"][:-1],
            "camera_times_s": arrays["camera_times_s"],
        }.items():
            np.save(folder / f"{name}.npy", value, allow_pickle=False)
        images = np.lib.format.open_memmap(
            folder / "images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(arrays["camera_frames"]), 3, image_size, image_size),
        )
        for frame_index, frame in enumerate(arrays["camera_frames"]):
            images[frame_index] = preprocess_image(frame, image_size)
        images.flush()
        del images
        for key, value in (("state", state), ("actions", actions)):
            value = value.astype(np.float64)
            sums[key] += value.sum(axis=0)
            squares[key] += np.square(value).sum(axis=0)
        count += len(episode)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        manifest["episodes"].append(
            {
                "directory": folder.name,
                "transitions": len(episode),
                "frames": len(arrays["camera_frames"]),
                "char": char,
                "source": str(path.resolve()),
                "sha256": digest,
                "attribution": metadata.get("attribution"),
                "provenance": metadata.get("provenance"),
                "camera": metadata.get("camera"),
                "camera_calibration": metadata.get("camera_calibration"),
                "complete": metadata["complete"],
                "recovery_supervision": metadata.get("recovery_supervision"),
            }
        )
    manifest["transitions"] = count
    manifest["chars"] = sorted({entry["char"] for entry in manifest["episodes"]})
    manifest["stats"] = {}
    for key, total in sums.items():
        mean = total / count
        population_std = np.sqrt(np.maximum(squares[key] / count - mean**2, 0))
        # Constant flags/ages must not turn ordinary unseen sensor events into
        # million-scale inputs. Action floors remain small: they define output units.
        std = (
            np.where(population_std < STD_FLOOR, 1.0, population_std)
            if key == "state"
            else np.maximum(population_std, STD_FLOOR)
        )
        manifest["stats"][key] = {"mean": mean.tolist(), "std": std.tolist()}
    manifest["stats_std_floor"] = STD_FLOOR
    manifest["normalization"] = NORMALIZATION_CONTRACT
    return manifest


class PreparedDataset(Dataset):
    """Read-only memory maps; suffix padding is masked and never crosses episodes."""

    def __init__(self, directory, chunk_size=16):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.directory = Path(directory)
        self.chunk_size = chunk_size
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        manifest = self.manifest
        validate_manifest_contract(manifest)
        if manifest.get("supervision", {}).get("source") == "expert" and chunk_size != 1:
            raise ValueError("Counterfactual expert supervision requires chunk_size=1")
        for key in ("state", "actions"):
            for statistic in ("mean", "std"):
                value = np.asarray(manifest["stats"][key][statistic], dtype=np.float32)
                setattr(self, f"{'action' if key == 'actions' else key}_{statistic}", value)
        self.episodes = []
        self.ends = []
        total = 0
        size = manifest["image_preprocessing"]["size"]
        seen = set()
        for entry in manifest["episodes"]:
            name = entry["directory"]
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise ValueError("Invalid prepared episode path")
            if name in seen:
                raise ValueError("Duplicate prepared episode directory")
            seen.add(name)
            char = entry.get("char")
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("Invalid prepared character label")
            if char in TEST and not manifest.get("allow_heldout", False):
                raise ValueError("Held-out character requires explicit research override")
            arrays = {
                key: np.load(
                    self.directory / name / f"{key}.npy", mmap_mode="r", allow_pickle=False
                )
                for key in (
                    "state",
                    "actions",
                    "images",
                    "camera_index",
                    "decision_times_s",
                    "camera_times_s",
                )
            }
            count, frames = entry["transitions"], entry["frames"]
            shapes = {
                "state": (count, 32),
                "actions": (count, 6),
                "images": (frames, 3, size, size),
                "camera_index": (count,),
                "decision_times_s": (count,),
                "camera_times_s": (frames,),
            }
            if (
                count < 1
                or frames < 1
                or any(arrays[key].shape != shape for key, shape in shapes.items())
            ):
                raise ValueError("Prepared episode shape mismatch")
            if arrays["images"].dtype != np.uint8 or arrays["camera_index"].dtype.kind not in "iu":
                raise ValueError("Prepared image/index dtype mismatch")
            if any(arrays[key].dtype != np.float32 for key in ("state", "actions")):
                raise ValueError("Prepared states and actions must be float32")
            if any(
                not np.isfinite(arrays[key]).all()
                for key in ("state", "actions", "decision_times_s", "camera_times_s")
            ):
                raise ValueError("Prepared episode has nonfinite values")
            for key in ("decision_times_s", "camera_times_s"):
                if arrays[key][0] < 0 or not np.all(np.diff(arrays[key]) > 0):
                    raise ValueError("Prepared timestamps must be nonnegative and increasing")
            indices = arrays["camera_index"]
            expected = (
                np.searchsorted(arrays["camera_times_s"], arrays["decision_times_s"], side="right")
                - 1
            )
            if np.any(indices < 0) or not np.array_equal(indices, expected):
                raise ValueError("Prepared camera mapping is not causal latest-frame selection")
            if np.any(np.abs(arrays["actions"]) > 1 + 1e-6):
                raise ValueError("Prepared actions exceed normalized command bounds")
            self.episodes.append((entry, arrays))
            total += count
            self.ends.append(total)
        if total < 1 or total != manifest["transitions"]:
            raise ValueError("Prepared transition count mismatch")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[episode_index - 1] if episode_index else 0)
        entry, arrays = self.episodes[episode_index]
        length = min(self.chunk_size, entry["transitions"] - local)
        actions = np.zeros((self.chunk_size, 6), dtype=np.float32)
        actions[:length] = (
            arrays["actions"][local : local + length] - self.action_mean
        ) / self.action_std
        return {
            "state": (arrays["state"][local] - self.state_mean) / self.state_std,
            "image": arrays["images"][arrays["camera_index"][local]].astype(np.float32) / 255,
            "actions": actions,
            "action_is_pad": np.arange(self.chunk_size) >= length,
            "task": f"Draw {entry['char']} on paper following the supplied stroke reference.",
        }
