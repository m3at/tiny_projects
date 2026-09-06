"""Held-out native-rod audit across timestep, discretization and friction."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from shodo.artifacts import provenance, save_rollout, snapshot_source
from shodo.config import free_hair_bundle, load_config
from shodo.data import TEST
from shodo.learning import rollout


def main():
    base = load_config(Path("experiments/cable.toml"))
    cases = {
        f"dt-{dt}": replace(base, timestep=dt, substeps=round(0.02 / dt))
        for dt in (0.0002, 0.0001, 0.00005)
    }
    for bundles, segments in ((7, 12), (19, 6), (19, 12)):
        brush = replace(free_hair_bundle(bundles=bundles, segments=segments), rod_tip_offset=0.001)
        cases[f"bundles-{bundles}-segments-{segments}"] = replace(
            base, brush=brush, timestep=0.00005, substeps=400
        )
    for friction in (0.44, 0.66):
        cases[f"friction-{friction}"] = replace(base, brush=replace(base.brush, friction=friction))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/native-audit"))
    parser.add_argument("--chars", default=TEST)
    parser.add_argument("--cases", nargs="+", choices=list(cases), default=list(cases))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "provenance": provenance(),
        "results": [],
        "source_snapshot": snapshot_source(output, "audit"),
    }
    start = time.perf_counter()
    for name in args.cases:
        config = cases[name]
        for char in args.chars:
            result = rollout(char, "expert", config=config)
            save_rollout(output / f"{name}-{ord(char):05x}", result)
            row = {"case": name, **result[0]}
            report["results"].append(row)
            (output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "case": name,
                        "char": char,
                        "seconds": time.perf_counter() - start,
                        **{
                            k: row[k]
                            for k in (
                                "steps",
                                "ink_rmse_mm",
                                "max_force_n",
                                "max_penetration_mm",
                                "truncated",
                            )
                        },
                    }
                ),
                flush=True,
            )
    report.update(status="complete", seconds=time.perf_counter() - start)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
