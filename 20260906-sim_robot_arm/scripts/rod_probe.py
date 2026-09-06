"""Sweep native rod integration/contact stability on the same drawing approach."""

import json
import time
from pathlib import Path

import numpy as np

from shodo.artifacts import provenance
from shodo.config import BrushConfig, SimConfig
from shodo.env import ShodoEnv


def main():
    rows = []
    start = time.perf_counter()
    for young in (3e8, 1e9):
        for damping in (1e-6, 3e-6, 1e-5):
            for dt in (0.0002, 0.0001, 0.00005):
                config = SimConfig(
                    timestep=dt,
                    substeps=round(0.02 / dt),
                    brush=BrushConfig(
                        backend="cable",
                        bundles=7,
                        young_modulus=young,
                        shear_modulus=young / 3,
                        rod_damping=damping,
                    ),
                )
                env = ShodoEnv(chars="一", config=config)
                forces, velocities = [], []
                timer = time.perf_counter()
                try:
                    env.reset(seed=7)
                    for _ in range(150):
                        _, _, _, truncated, info = env.step(env.expert())
                        forces.append(info["force_n"])
                        velocities.append(float(np.max(np.abs(env.data.qvel[7:]))))
                        if truncated:
                            break
                    row = {
                        "young": young,
                        "damping": damping,
                        "dt": dt,
                        "steps": len(forces),
                        "truncated": truncated,
                        "max_force": max(forces),
                        "force_std_last20": float(np.std(forces[-20:])),
                        "max_velocity": max(velocities),
                        "penetration_mm": env.brush.max_penetration * 1000,
                        "seconds": time.perf_counter() - timer,
                        "warnings": env.data.warning.number.tolist(),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    env.close()
    output = Path("runs/v2/rod-probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"seconds": time.perf_counter() - start, "rows": rows, "provenance": provenance()},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
