"""Portable run artifacts and the provenance needed to reproduce them."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from shodo.data import SHA256, TEST, TRAIN
from shodo.env import HISTORY_COLUMNS
from shodo.robot import REVISION


def provenance():
    root = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "gymnasium", "numpy", "torch", "stable-baselines3")
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir())
            if p.suffix in (".py", ".xml")
        },
        "dataset_sha256": SHA256,
        "robot_revision": REVISION,
        "train_chars": TRAIN,
        "held_out_chars": TEST,
    }


def save_rollout(base, result):
    metrics, history, paper, frames = result
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paper.save(base.with_suffix(".png"))
    np.savez_compressed(
        base.with_suffix(".npz"),
        history=history,
        columns=np.array(HISTORY_COLUMNS),
    )
    if frames:
        frames[-1].save(base.with_name(base.name + "-scene").with_suffix(".png"))
        frames[0].save(
            base.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=100, loop=0
        )
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "provenance": provenance(),
                "derived_data_attribution": "KanjiVG, Ulrich Apel and contributors, CC BY-SA 3.0; "
                "resampled, transformed to 3D and executed with simulated arm dynamics",
            },
            indent=2,
        )
        + "\n"
    )
