"""End-to-end validation with explicit, optimization-safe acceptance checks."""

import json
import time
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from shodo.artifacts import provenance, save_rollout
from shodo.config import SimConfig
from shodo.data import TEST, TRAIN
from shodo.env import ShodoEnv
from shodo.learning import evaluate, load_policy, load_ppo, rollout, train
from shodo.video import require_ffmpeg


def validate(
    directory,
    *,
    config=None,
    chars=TEST,
    seed=7,
    algorithm="learned",
    episodes=28,
    epochs=35,
    device="cpu",
):
    require_ffmpeg()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not chars or set(chars) & set(TRAIN):
        raise ValueError(
            "Validation characters must be nonempty and disjoint from default training"
        )
    if algorithm not in ("learned", "ppo"):
        raise ValueError(
            "Validate a learned or PPO controller; teacher/zero are included automatically"
        )
    training_config = config
    config = config or SimConfig()
    start = time.perf_counter()
    output = directory / "validation.json"
    report = {
        "passed": False,
        "status": "running",
        "provenance": provenance(),
        "algorithm": algorithm,
        "config": config.to_dict(),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    env = ShodoEnv(chars="一", config=config)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    checkpoint = directory / "bc.pt"
    if algorithm == "learned" and not checkpoint.exists():
        train(
            episodes=episodes,
            epochs=epochs,
            seed=seed,
            output=checkpoint,
            config=training_config,
            device=device,
        )
    metadata_path = directory / ("bc.json" if algorithm == "learned" else "ppo.json")
    if metadata_path.exists():
        trained = json.loads(metadata_path.read_text()).get("train_chars", TRAIN)
        if set(chars) & set(trained):
            raise ValueError("Validation characters overlap this checkpoint's training characters")
    results = evaluate(
        directory / f"evaluation-{algorithm}.json",
        checkpoint,
        algorithm=algorithm,
        config=config,
        chars=chars,
        seed=seed,
        device=device,
    )
    checks = {"gymnasium_api": True}
    for name in ("expert", algorithm):
        for row in results[name]:
            prefix = f"{name}/{row['char']}/"
            checks[prefix + "complete"] = not row["truncated"]
            for key, threshold in (
                ("rmse_mm", 4.0),
                ("ink_rmse_mm", 4.0),
                ("force_rmse_n", 0.15),
                ("max_force_n", 1.5),
                ("orientation_rmse_deg", 3.0),
                ("raster_spill_fraction", 0.15),
            ):
                checks[prefix + key] = row[key] is not None and row[key] < threshold
            for key, threshold in (
                ("draw_contact_fraction", 0.95),
                ("lift_clear_fraction", 0.98),
                ("raster_coverage_fraction", 0.95),
            ):
                checks[prefix + key] = row[key] is not None and row[key] > threshold
            checks[prefix + "pigment_conservation"] = row["pigment_mass_error"] < 1e-8
            checks[prefix + "torque_limits"] = row["max_torque_fraction"] <= 1 + 1e-12
            checks[prefix + "no_arm_collisions"] = row["forbidden_contact_substeps"] == 0
            checks[prefix + "joint_margin"] = row["minimum_joint_margin_rad"] > 0.01
            checks[prefix + "joint_speed"] = (
                max(row["peak_joint_speed_rad_s"]) < 1.5 * config.robot.joint_speed
            )
            if config.brush.backend == "cable":
                checks[prefix + "penetration"] = row["max_penetration_mm"] < 0.15
    learned = np.mean([r["rmse_mm"] for r in results[algorithm]])
    zero = np.mean([r["rmse_mm"] for r in results["zero"]])
    checks["baseline_improvement_80pct"] = bool(learned < zero * 0.2)
    policy = (
        load_policy(checkpoint, device=device)
        if algorithm == "learned"
        else load_ppo(directory, device=device)
    )
    result = rollout(chars[0], policy, frames=True, config=config, seed=seed)
    checks["render_nonempty"] = bool(result[3] and np.asarray(result[3][-1]).std() > 10)
    save_rollout(directory / "validation-rollout", result)
    report.update(
        passed=all(checks.values()),
        status="complete",
        checks=checks,
        seconds=time.perf_counter() - start,
        held_out=results,
        render_frames=len(result[3]),
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Validation failed ({output}): {', '.join(failures)}")
    print(f"Validation passed: {len(checks)} checks; report {output}")
    return report
