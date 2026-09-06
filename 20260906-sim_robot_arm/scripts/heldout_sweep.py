"""Fixed-seed, broader KanjiVG cohort; streaming results survive interruption."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from shodo.artifacts import provenance, snapshot_source
from shodo.config import SimConfig
from shodo.data import DATA, TEST, TRAIN
from shodo.learning import load_policy, load_ppo, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("runs/v2/heldout-sweep"))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=[
            "expert",
            "bc",
            "ppo",
            "residual",
            "pressure-residual",
            "ink-residual",
            "pressure-ink-residual",
        ],
        default=["expert", "bc", "ppo", "residual"],
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    candidates = sorted(
        chr(int(p.stem, 16))
        for p in DATA.glob("*.svg")
        if chr(int(p.stem, 16)) not in TRAIN + TEST and 0x4E00 <= int(p.stem, 16) <= 0x9FFF
    )
    if not 0 <= args.count <= len(candidates):
        parser.error(f"--count must be between 0 and {len(candidates)}")
    rng = np.random.default_rng(20260906)
    chars = TEST + "".join(rng.choice(candidates, size=args.count, replace=False))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "bc": Path("runs/v2/bc.pt"),
        "ppo": Path("runs/v2/ppo-seed7/ppo.zip"),
        "residual": Path("runs/v2/residual-seed7/ppo.zip"),
        "pressure-residual": Path("runs/v2/pressure-residual/ppo.zip"),
        "ink-residual": Path("runs/v2/ink-residual/ppo.zip"),
        "pressure-ink-residual": Path("runs/v2/pressure-ink-residual/ppo.zip"),
    }
    checkpoints = {k: v for k, v in checkpoints.items() if k in args.policies}
    metadata = {
        "chars": chars,
        "material_seeds": [7, 17, 27],
        "sampling_seed": 20260906,
        "provenance": provenance(),
        "status": "running",
        "policies": list(dict.fromkeys(args.policies)),
        "source_snapshot": snapshot_source(args.output, "cohort"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": {
            k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in checkpoints.items()
        },
    }
    entrypoint = args.output / metadata["source_snapshot"] / Path(__file__).name
    shutil.copy2(__file__, entrypoint)
    metadata["entrypoint_snapshot"] = str(entrypoint.relative_to(args.output))
    meta_path = args.output / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    policies = {}
    for name in metadata["policies"]:
        policies[name] = (
            "expert"
            if name == "expert"
            else load_policy(checkpoints[name])
            if name == "bc"
            else load_ppo(checkpoints[name].parent)
        )
    start = time.perf_counter()
    rows = []
    with (args.output / "episodes.jsonl").open("w") as stream:
        for name, policy in policies.items():
            for char in chars:
                for seed in metadata["material_seeds"]:
                    metrics, _, _, _ = rollout(
                        char, policy, seed=seed, config=SimConfig(randomize=True)
                    )
                    row = {"policy": name, **metrics}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                if len(rows) % 24 == 0:
                    print(
                        json.dumps(
                            {
                                "episodes": len(rows),
                                "policy": name,
                                "seconds": time.perf_counter() - start,
                                "latest_ink_rmse_mm": metrics["ink_rmse_mm"],
                            }
                        ),
                        flush=True,
                    )
    metadata.update(status="complete", seconds=time.perf_counter() - start, episodes=len(rows))
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
