"""Plant-only material perturbations: target force remains nominal."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from shodo.artifacts import provenance, snapshot_source
from shodo.config import SimConfig
from shodo.data import TEST
from shodo.learning import load_policy, load_ppo, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/v2/material-sweep"))
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["expert", "bc", "residual"],
        choices=[
            "expert",
            "bc",
            "residual",
            "pressure-expert",
            "pressure-bc",
            "pressure-residual",
            "ink-residual",
            "pressure-ink-residual",
        ],
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = {
        "status": "running",
        "provenance": provenance(),
        "results": [],
        "source_snapshot": snapshot_source(output, "material"),
        "description": "Actual stiffness/friction varied; nominal reference force is unchanged",
    }
    checkpoints = {
        "bc": Path("runs/v2/bc.pt"),
        "residual": Path("runs/v2/residual-seed7/ppo.zip"),
        "pressure-bc": Path("runs/v2/pressure/bc.pt"),
        "pressure-residual": Path("runs/v2/pressure-residual/ppo.zip"),
        "ink-residual": Path("runs/v2/ink-residual/ppo.zip"),
        "pressure-ink-residual": Path("runs/v2/pressure-ink-residual/ppo.zip"),
    }
    policies = {}
    for name in dict.fromkeys(args.policies):
        path = checkpoints.get(name)
        policies[name] = (
            "expert"
            if path is None
            else load_policy(path)
            if path.suffix == ".pt"
            else load_ppo(path.parent)
        )
    report["checkpoint_sha256"] = {
        name: hashlib.sha256(checkpoints[name].read_bytes()).hexdigest()
        for name in policies
        if name in checkpoints
    }
    entrypoint = output / report["source_snapshot"] / Path(__file__).name
    shutil.copy2(__file__, entrypoint)
    report["entrypoint_snapshot"] = str(entrypoint.relative_to(output))
    report["script_sha256"] = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    start = time.perf_counter()
    for stiffness in (0.5, 0.8, 1.0, 1.2, 1.5):
        for friction in (0.5, 0.8, 1.0, 1.2, 1.5):
            material = {"normal_stiffness": 220 * stiffness, "friction": 0.55 * friction}
            for name, policy in policies.items():
                for char in TEST:
                    config = (
                        SimConfig(force_feedback_gain=0.02)
                        if name == "pressure-expert"
                        else SimConfig()
                    )
                    metrics, _, _, _ = rollout(char, policy, material=material, config=config)
                    row = {
                        "policy": name,
                        "stiffness_scale": stiffness,
                        "friction_scale": friction,
                        **metrics,
                    }
                    report["results"].append(row)
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "stiffness": stiffness,
                        "friction": friction,
                        "episodes": len(report["results"]),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    report.update(status="complete", seconds=time.perf_counter() - start)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
