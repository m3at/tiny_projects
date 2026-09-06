"""Exhaustive geometry sanity checks for the pinned local KanjiVG collection."""

import json
import time
from pathlib import Path

import numpy as np

from shodo.artifacts import provenance
from shodo.data import DATA, strokes, trajectory


def main():
    files = sorted(DATA.glob("*.svg"))
    if not files:
        raise RuntimeError("No KanjiVG SVGs found; run make data before auditing")
    rows, failures = [], []
    start = time.perf_counter()
    for file in files:
        char = chr(int(file.stem, 16))
        try:
            path, ids = trajectory(char)
            maximum_step = float(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            if not np.isfinite(path).all():
                raise ValueError("Nonfinite trajectory")
            if maximum_step > 0.001201:
                raise ValueError(f"Excessive per-step displacement: {maximum_step}")
            if not np.all(np.diff(ids[ids >= 0]) >= 0):
                raise ValueError("Stroke order changed")
            rows.append(
                {
                    "char": char,
                    "strokes": len(strokes(char)),
                    "steps": len(path),
                    "max_step_m": maximum_step,
                    "minimum_xyz": path.min(axis=0).tolist(),
                    "maximum_xyz": path.max(axis=0).tolist(),
                }
            )
        except (ValueError, ArithmeticError, OSError, SyntaxError) as error:
            failures.append({"char": char, "error": repr(error)})
        if (len(rows) + len(failures)) % 250 == 0:
            print(
                json.dumps(
                    {
                        "checked": len(rows) + len(failures),
                        "failures": len(failures),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    report = {
        "passed": not failures,
        "results": rows,
        "failures": failures,
        "seconds": time.perf_counter() - start,
        "provenance": provenance(),
    }
    output = Path("runs/dataset-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} dataset failures; see runs/dataset-audit.json")


if __name__ == "__main__":
    main()
