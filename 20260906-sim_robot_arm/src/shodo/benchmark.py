"""Warm end-to-end CPU throughput with physical completion checks."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from shodo.artifacts import provenance
from shodo.config import SimConfig
from shodo.env import ShodoEnv


def benchmark(output="runs/benchmark.json", *, config=None, char="永", repeats=3, seed=7):
    if repeats < 1 or len(char) != 1:
        raise ValueError("Benchmark needs positive repeats and exactly one character")
    config = replace(config or SimConfig(), record=False)
    env = ShodoEnv(chars=char, config=config)
    results = []
    try:
        for trial in range(repeats + 1):
            env.reset(seed=seed)
            steps, squared_error = 0, 0.0
            start = time.perf_counter()
            while True:
                _, _, terminated, truncated, info = env.step(env.expert())
                steps += 1
                squared_error += info["error_m"] ** 2
                if truncated:
                    raise RuntimeError("Unstable benchmark is not a valid throughput measurement")
                if terminated:
                    break
            seconds = time.perf_counter() - start
            if trial:
                row = {
                    "steps": steps,
                    "seconds": seconds,
                    "control_steps_s": steps / seconds,
                    "simulated_seconds_per_wall_second": steps * config.dt / seconds,
                    "tracking_rmse_mm": (squared_error / steps) ** 0.5 * 1000,
                    "max_force_n": env.peak_force,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
        report = {
            "trials": results,
            "config": config.to_dict(),
            "char": char,
            "seed": seed,
            "median_control_steps_s": float(np.median([r["control_steps_s"] for r in results])),
            "provenance": provenance(),
            "dofs": env.model.nv,
            "scope": "Warm teacher+control+dynamics+ink; excludes reset/model load/rendering; host/load dependent",
        }
    finally:
        env.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
