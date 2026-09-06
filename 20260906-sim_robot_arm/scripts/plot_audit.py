"""Headless figures from recorded measurements; no simulator reruns or GUI."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("runs/v2")


def mechanics():
    rows = json.loads((ROOT / "mechanics/cantilever.json").read_text())["results"]
    rows = [r for r in rows if r["timestep"] == 0.0001 and not r["recovery_s"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = [r["segments"] for r in rows]
    axes[0].plot(
        x, [r["displacement_m"] / r["beam_upper_m"] for r in rows], "o-", label="Native rods"
    )
    axes[0].plot(
        x,
        [r["discrete_beam_m"] / r["beam_upper_m"] for r in rows],
        "x--",
        label="Discrete beam prediction",
    )
    axes[0].axhline(1, color="0.5", linestyle=":", label="Continuum limit")
    axes[0].set(
        xlabel="Segments per rod",
        ylabel="Deflection / continuum prediction",
        title="Clamped first segment: discretization matters",
        xticks=x,
    )
    axes[0].legend(fontsize=8)
    rows = json.loads((ROOT / "contact-sweep.json").read_text())["results"]
    curves = sorted({r["curvature_m"] for r in rows})
    speeds = [0.04, 0.005, 0.001]
    forces = np.full((len(curves), len(speeds)), np.nan)
    for r in rows:
        if r["timestep_s"] == 0.0001:
            forces[curves.index(r["curvature_m"]), speeds.index(r["approach_m_s"])] = r["metrics"][
                "max_force_n"
            ]
    plot = axes[1].imshow(np.log10(forces), cmap="magma", aspect="auto")
    for (i, j), f in np.ndenumerate(forces):
        axes[1].text(
            j, i, f"{f:.2f} N", ha="center", va="center", color="white" if f < 5 else "black"
        )
    axes[1].set(
        xticks=range(3),
        xticklabels=[40, 5, 1],
        yticks=range(len(curves)),
        yticklabels=[f"{c * 1000:g}" for c in curves],
        xlabel="Touchdown reference speed (mm/s)",
        ylabel="Initial lateral tip offset (mm)",
        title="One-stroke force peaks, 0.1 ms timestep",
    )
    figure.colorbar(plot, ax=axes[1], label="log10 peak force (N)")
    figure.savefig(ROOT / "mechanics-summary.svg")
    figure.savefig(ROOT / "mechanics-summary.png", dpi=150)
    plt.close(figure)


def training():
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for directory in ("ppo-seed7", "residual-seed7-3m", "residual-seed17", "residual-seed27"):
        path = ROOT / directory / "ppo-monitor.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.read_text().splitlines()[1:]))
        rows = [r for r in rows if r.get("r") and r.get("l") and r.get("t")]
        lengths = np.array([int(r["l"]) for r in rows])
        returns = np.array([float(r["r"]) / int(r["l"]) for r in rows])
        window = min(50, len(returns))
        if window:
            smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
            axis.plot(np.cumsum(lengths)[window - 1 :], smooth, label=directory)
    axis.set(
        xlabel="Training transitions",
        ylabel="Reward per step (50-episode moving mean)",
        title="Training returns — not held-out accuracy",
        ylim=(0, 1.02),
    )
    axis.legend(fontsize=8)
    figure.savefig(ROOT / "training-curves.svg")
    figure.savefig(ROOT / "training-curves.png", dpi=150)
    plt.close(figure)


def material():
    summary = {}
    for directory in (
        "material-sweep",
        "material-pressure-bc",
        "material-pressure-residual",
        "material-ink-residual",
        "material-pressure-ink-residual",
    ):
        path = ROOT / directory / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for name in dict.fromkeys(r["policy"] for r in report["results"]):
            group = [r for r in report["results"] if r["policy"] == name]
            good = [r for r in group if not r["truncated"] and r["ink_rmse_mm"] is not None]
            summary[name] = {
                "source": str(path),
                "status": report["status"],
                "episodes": len(group),
                "truncated": sum(r["truncated"] for r in group),
                "missing_ink": sum(r["ink_rmse_mm"] is None for r in group),
                "mean_tracking_rmse_mm_completed": float(np.mean([r["rmse_mm"] for r in good]))
                if good
                else None,
                "mean_ink_rmse_mm_completed": float(np.mean([r["ink_rmse_mm"] for r in good]))
                if good
                else None,
                "mean_force_rmse_n_completed": float(np.mean([r["force_rmse_n"] for r in good]))
                if good
                else None,
                "worst_force_rmse_n": max(r["force_rmse_n"] or 0 for r in group),
                "minimum_raster_coverage": min(r["raster_coverage_fraction"] for r in group),
                "maximum_force_n": max(r["max_force_n"] for r in group),
            }
    (ROOT / "material-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


def cohort(directory="heldout-sweep", prefix="heldout"):
    path = ROOT / directory / "episodes.jsonl"
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.endswith("}")]
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for name in dict.fromkeys(r["policy"] for r in rows):
        values = sorted(
            r["ink_rmse_mm"]
            for r in rows
            if r["policy"] == name and r["ink_rmse_mm"] is not None and not r["truncated"]
        )
        axis.plot(
            values, np.arange(1, len(values) + 1) / len(values), label=f"{name} (n={len(values)})"
        )
    status = json.loads((path.parent / "metadata.json").read_text())["status"]
    summary = {"status": status, "episodes": len(rows), "policies": {}}
    for name in dict.fromkeys(r["policy"] for r in rows):
        group = [r for r in rows if r["policy"] == name]
        good = [r for r in group if not r["truncated"] and r["ink_rmse_mm"] is not None]
        summary["policies"][name] = {
            "episodes": len(group),
            "truncated": sum(r["truncated"] for r in group),
            "missing_ink": sum(r["ink_rmse_mm"] is None for r in group),
            "completed_with_ink": len(good),
            "mean_ink_rmse_mm_completed": float(np.mean([r["ink_rmse_mm"] for r in good]))
            if good
            else None,
            "p95_ink_rmse_mm_completed": float(np.quantile([r["ink_rmse_mm"] for r in good], 0.95))
            if good
            else None,
            "mean_force_rmse_n_completed": float(np.mean([r["force_rmse_n"] for r in good]))
            if good
            else None,
            "minimum_raster_coverage": min(r["raster_coverage_fraction"] for r in group),
            "maximum_force_n": max(r["max_force_n"] for r in group),
        }
    if status == "complete":
        # Resample glyphs, not correlated material-seed repetitions within a glyph.
        grouped = {}
        for name in ("bc", "residual"):
            grouped[name] = {}
            for r in rows:
                if r["policy"] == name and not r["truncated"] and r["ink_rmse_mm"] is not None:
                    grouped[name].setdefault(r["char"], []).append(r["ink_rmse_mm"])
        chars = sorted(
            c
            for c in grouped["bc"]
            if len(grouped["bc"][c]) == 3 and len(grouped["residual"].get(c, [])) == 3
        )
        if chars:
            delta = np.array(
                [np.mean(grouped["bc"][c]) - np.mean(grouped["residual"][c]) for c in chars]
            )
            rng = np.random.default_rng(20260906)
            samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
            summary["paired_ink_improvement_mm_bc_minus_residual"] = {
                "completed_glyph_pairs": len(chars),
                "mean": float(delta.mean()),
                "glyph_bootstrap_95pct_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
                "bootstrap_seed": 20260906,
                "bootstrap_samples": 10000,
            }
    (ROOT / f"{prefix}-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    axis.set(
        xlabel="Ink-center RMSE (mm)",
        ylabel="Fraction of completed episodes",
        title=f"Broader held-out cohort — {status}",
    )
    axis.legend()
    figure.savefig(ROOT / f"{prefix}-distribution.svg")
    figure.savefig(ROOT / f"{prefix}-distribution.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    mechanics()
    training()
    material()
    cohort()
    cohort("heldout-pressure", "heldout-pressure")
    cohort("heldout-ink", "heldout-ink")
    cohort("heldout-pressure-ink", "heldout-pressure-ink")
