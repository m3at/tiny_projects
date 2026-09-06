"""Portable run artifacts and the provenance needed to reproduce them."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from shodo.data import SHA256, TEST, TRAIN
from shodo.env import HISTORY_COLUMNS
from shodo.robot import REVISION

_ROOT = Path(__file__).parent
# Capture once: a long-running experiment must not report later edits as its source.
_SOURCE_HASHES = {
    p.relative_to(_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(_ROOT.rglob("*"))
    if p.suffix in (".py", ".xml", ".json")
}
_LOCK_HASH = (
    hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest() if Path("uv.lock").exists() else None
)


def provenance():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "mujoco",
                "gymnasium",
                "numpy",
                "torch",
                "stable-baselines3",
                "scipy",
                "pillow",
                "filelock",
            )
        },
        "source_sha256": dict(_SOURCE_HASHES),
        "source_hash_capture": "artifact module import",
        "lock_sha256": _LOCK_HASH,
        "dataset_sha256": SHA256,
        "robot_revision": REVISION,
        "train_chars": TRAIN,
        "held_out_chars": TEST,
    }


def snapshot_source(directory, label):
    """Importable package snapshot: PYTHONPATH=<snapshot> reproduces this source."""
    snapshot = Path(directory) / f"source-{label}-{time.time_ns()}"
    shutil.copytree(
        Path(__file__).parent, snapshot / "shodo", ignore=shutil.ignore_patterns("__pycache__")
    )
    copied = {
        p.relative_to(snapshot / "shodo").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (snapshot / "shodo").rglob("*")
        if p.suffix in (".py", ".xml", ".json")
    }
    if copied != _SOURCE_HASHES:
        raise RuntimeError("Source changed after this process imported it; restart before training")
    return snapshot.name


def save_rollout(base, result):
    metrics, history, paper, frames = result
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    # `base` is an experiment identifier, not a filename with an extension.
    # Decimal parameters (e.g. dt-0.0001) must survive artifact suffixes.
    paper.save(Path(f"{base}.png"))
    np.savez_compressed(
        Path(f"{base}.npz"),
        history=history,
        columns=np.array(HISTORY_COLUMNS),
    )
    if frames:
        frames[-1].save(Path(f"{base}-scene.png"))
        frames[0].save(
            Path(f"{base}.gif"), save_all=True, append_images=frames[1:], duration=100, loop=0
        )
    Path(f"{base}.json").write_text(
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
