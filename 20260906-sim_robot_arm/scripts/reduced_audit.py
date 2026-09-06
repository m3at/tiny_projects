"""Reduced-brush timestep and bundle-count sensitivity on held-out glyphs."""

import json
import time
from dataclasses import replace
from pathlib import Path

from shodo.artifacts import provenance, save_rollout, snapshot_source
from shodo.config import BrushConfig, SimConfig
from shodo.data import TEST
from shodo.learning import rollout


def main():
    output = Path("runs/v2/reduced-audit")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "results": [],
        "provenance": provenance(),
        "source_snapshot": snapshot_source(output, "reduced-audit"),
    }
    start = time.perf_counter()
    for bundles in (7, 19, 37):
        for dt in (0.004, 0.002, 0.001, 0.0005):
            config = SimConfig(
                timestep=dt,
                substeps=round(0.02 / dt),
                brush=replace(BrushConfig(), bundles=bundles),
            )
            for char in TEST:
                result = rollout(char, "expert", config=config)
                row = {"bundles": bundles, "timestep": dt, **result[0]}
                report["results"].append(row)
                save_rollout(output / f"bundles-{bundles}-dt-{dt}-{ord(char):05x}", result)
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "bundles": bundles,
                        "timestep": dt,
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
