"""Native touchdown sensitivity; includes deliberately harsh stress cases."""

import json
import time
from dataclasses import replace
from pathlib import Path

from shodo.artifacts import provenance
from shodo.config import load_config
from shodo.learning import rollout


def main():
    output = Path("runs/v2/contact-sweep.json")
    config = load_config(Path("experiments/cable.toml"))
    report = {"provenance": provenance(), "status": "running", "results": []}
    for curve in (0.0, 0.0005, 0.001, 0.002):
        for speed in (0.04, 0.005, 0.001):
            for dt in (0.0002, 0.0001, 0.00005):
                cfg = replace(
                    config,
                    timestep=dt,
                    substeps=round(0.02 / dt),
                    touchdown_speed=speed,
                    brush=replace(config.brush, rod_tip_offset=curve),
                )
                start = time.perf_counter()
                metrics, _, _, _ = rollout("一", "expert", config=cfg)
                row = {
                    "curvature_m": curve,
                    "approach_m_s": speed,
                    "timestep_s": dt,
                    "seconds": time.perf_counter() - start,
                    "metrics": metrics,
                }
                report["results"].append(row)
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "metrics"}
                        | {
                            k: metrics[k]
                            for k in (
                                "steps",
                                "truncated",
                                "max_force_n",
                                "max_penetration_mm",
                                "ink_rmse_mm",
                            )
                        }
                    ),
                    flush=True,
                )
    report["status"] = "complete"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
