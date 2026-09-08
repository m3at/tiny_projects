"""Paired sensor/calibration stress tests, not a hardware qualification protocol."""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from shodo.artifacts import provenance
from shodo.config import SimConfig
from shodo.contracts import SensorConfig
from shodo.data import TEST

METRICS = (
    "ink_rmse_mm",
    "force_rmse_n",
    "max_force_n",
    "raster_coverage_fraction",
    "draw_contact_fraction",
    "lift_clear_fraction",
)


@dataclass(frozen=True)
class RobustnessCase:
    """Complete sensor settings and optional absolute plant material coefficients."""

    name: str
    sensors: SensorConfig
    description: str
    material: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not self.name or not self.description:
            raise ValueError("Cases need a name and description")
        if not isinstance(self.sensors, SensorConfig):
            raise TypeError("Cases require an immutable SensorConfig")
        if not isinstance(self.material, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.material
        ):
            raise ValueError("Material settings must be immutable name/value pairs")
        allowed = {"normal_stiffness", "friction"}
        if len(dict(self.material)) != len(self.material) or any(
            key not in allowed or not math.isfinite(value) or value <= 0
            for key, value in self.material
        ):
            raise ValueError("Invalid or duplicate material coefficients")


def default_cases(sensors=None, config=None, *, include_material=False):
    """Exploratory fixed severities in SI units; no acceptance thresholds implied.

    Perturbed fields replace their base values, rather than adding to them. Paper
    cases perturb estimated registration only; the physical sheet is not moved.
    """
    base = sensors or SensorConfig()
    cases = [RobustnessCase("nominal", base, "Unmodified supplied sensor configuration")]

    def add(name, description, **changes):
        cases.append(RobustnessCase(name, replace(base, **changes), description))

    for sign, label in ((1, "positive"), (-1, "negative")):
        add(
            f"paper_xy_{label}",
            "Estimated paper origin error of 2 mm in both world X and Y",
            reference_offset=(sign * 0.002, sign * 0.002, 0.0),
        )
        add(
            f"paper_height_{label}",
            "Estimated paper height error of 1 mm; physical plane remains at zero",
            reference_offset=(0.0, 0.0, sign * 0.001),
        )
        add(
            f"paper_yaw_{label}",
            "Estimated paper yaw error of 2 degrees about the paper center",
            reference_yaw=sign * math.radians(2),
        )
        add(
            f"force_bias_{label}",
            "World Z force measurement bias of 0.1 N",
            force_bias=(0.0, 0.0, sign * 0.1),
        )
    add(
        "tool_calibration",
        "Measured tool position error of +1 mm X and +0.5 mm Z",
        tool_offset=(0.001, 0.0, 0.0005),
    )
    add("latency", "Two control intervals of sensor latency (40 ms)", latency_steps=2)
    add("dropout", "Independent sensor packet loss probability of 10%", dropout=0.1)
    noise = {
        "position_noise": 0.0002,
        "joint_noise": 0.0005,
        "velocity_noise": 0.005,
        "force_noise": 0.02,
    }
    add("noise", "Gaussian sensor noise; standard deviations recorded in case settings", **noise)
    add(
        "combined",
        "Joint registration, calibration, sensing noise, packet loss and latency stress",
        reference_offset=(0.002, -0.002, 0.001),
        reference_yaw=math.radians(2),
        tool_offset=(0.001, 0.0, 0.0005),
        force_bias=(0.0, 0.0, 0.1),
        latency_steps=2,
        dropout=0.1,
        **noise,
    )
    if include_material:
        cfg = config or SimConfig()
        for scale, name in ((0.8, "soft_low_friction"), (1.2, "stiff_high_friction")):
            material = [("friction", cfg.brush.friction * scale)]
            if cfg.brush.backend == "reduced":
                material.append(("normal_stiffness", cfg.brush.normal_stiffness * scale))
            cases.append(
                RobustnessCase(
                    f"material_{name}",
                    base,
                    f"Actual material coefficients scaled by {scale}; nominal reference unchanged",
                    tuple(material),
                )
            )
    return tuple(cases)


def _number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def _summary(rows):
    result = {
        "episodes": len(rows),
        "truncated": sum(row["truncated"] for row in rows),
        "incomplete": sum(row["incomplete"] for row in rows),
        "missing_ink": sum(row["missing_ink"] for row in rows),
        "metrics": {},
        "paired_nominal": {},
    }
    for field in METRICS:
        for key, values in (
            ("metrics", [row["metrics"].get(field) for row in rows]),
            ("paired_nominal", [row["delta_from_nominal"].get(field) for row in rows]),
        ):
            valid = [value for value in values if _number(value)]
            result[key][field] = {
                "valid_pairs" if key == "paired_nominal" else "valid_episodes": len(valid),
                "mean": float(np.mean(valid)) if valid else None,
                "min": float(min(valid)) if valid else None,
                "max": float(max(valid)) if valid else None,
            }
    return result


def evaluate_robustness(
    output,
    *,
    chars=TEST,
    seeds=(7, 17, 27),
    config=None,
    sensors=None,
    policy=None,
    cases=None,
    include_material=False,
    runner=None,
):
    """Run matched character/seed episodes and save raw metrics plus paired deltas.

    A supplied learned callable declares its contract through a SensorConfig-valued
    sensor_config attribute. Corruption may differ, but every case must preserve
    its trained history length. The episode runner owns policy reset(), once per
    episode; this orchestrator never resets it. No controller is tuned or selected
    by this suite. Exceptions propagate instead of becoming silently excluded
    episodes. ``runner`` supports focused contract tests with the same reset contract.
    """
    config = config or SimConfig()
    chars, seeds = tuple(chars), tuple(seeds)
    if not chars or any(not isinstance(char, str) or len(char) != 1 for char in chars):
        raise ValueError("Provide one or more individual characters")
    if len(set(chars)) != len(chars):
        raise ValueError("Characters must be unique for paired evaluation")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique for paired evaluation")
    if policy is not None and not callable(policy):
        raise ValueError("Learned policy must be a sensor-compatible callable")
    trained_sensors = getattr(policy, "sensor_config", None)
    if policy is not None and not isinstance(trained_sensors, SensorConfig):
        raise ValueError("Learned policy must declare a SensorConfig in sensor_config")
    sensors = sensors or trained_sensors
    cases = tuple(
        default_cases(sensors, config, include_material=include_material)
        if cases is None
        else cases
    )
    names = [case.name for case in cases]
    if not names or names[0] != "nominal" or len(set(names)) != len(names):
        raise ValueError("Cases must start with nominal and have unique names")
    if cases[0].material:
        raise ValueError("Nominal case must use the supplied plant configuration without overrides")
    if policy is not None and any(
        case.sensors.history != trained_sensors.history for case in cases
    ):
        raise ValueError("All case sensor history lengths must match the learned policy")
    if config.brush.backend == "cable" and any(
        "normal_stiffness" in dict(case.material) for case in cases
    ):
        raise ValueError("Native rods do not support runtime normal_stiffness overrides")
    if runner is None:
        from shodo.runtime import sensor_rollout

        runner = sensor_rollout
    run = runner
    controllers = {"classical": "classical", "oracle": "oracle", "zero": "zero"}
    if policy is not None:
        controllers["learned"] = policy
    report = {
        "schema_version": 1,
        "purpose": "Exploratory paired sensor/calibration robustness, not hardware qualification",
        "limitations": [
            "Paper perturbations are registration errors, not physical changes of the contact plane",
            "Oracle uses privileged simulator state and true reference; it is not a deployable baseline",
            "Cases and seeds are fixed engineering probes, not an untouched final test",
            "No acceptance thresholds or aggregate safety score are inferred",
            "Scalar summaries include incomplete episodes; inspect their counts and raw metrics",
        ],
        "pairing": "Same character, seed and controller; case minus its own nominal result",
        "chars": list(chars),
        "seeds": list(seeds),
        "config": config.to_dict(),
        "cases": [{**asdict(case), "material": dict(case.material)} for case in cases],
        "controllers": list(controllers),
        "checkpoint_provenance": getattr(policy, "checkpoint_provenance", None),
        "provenance": provenance(),
        "rows": [],
        "summary": {},
    }
    nominal = {}
    for case in cases:
        for name, controller in controllers.items():
            group = []
            for char in chars:
                for seed in seeds:
                    metrics = run(
                        char,
                        policy=controller,
                        seed=seed,
                        config=config,
                        sensors=case.sensors,
                        material=dict(case.material),
                    )
                    key = name, char, seed
                    if case.name == "nominal":
                        nominal[key] = metrics
                    baseline = nominal[key]
                    truncated = bool(metrics["truncated"])
                    incomplete = truncated or metrics.get("terminated") is False
                    if "expected_steps" in metrics:
                        incomplete |= metrics["steps"] < metrics["expected_steps"]
                    row = {
                        "case": case.name,
                        "controller": name,
                        "char": char,
                        "seed": seed,
                        "truncated": truncated,
                        "incomplete": bool(incomplete),
                        "missing_ink": not _number(metrics.get("ink_rmse_mm")),
                        "metrics": metrics,
                        "delta_from_nominal": {
                            field: float(metrics[field] - baseline[field])
                            if _number(metrics.get(field)) and _number(baseline.get(field))
                            else None
                            for field in METRICS
                        },
                    }
                    report["rows"].append(row)
                    group.append(row)
            report["summary"].setdefault(case.name, {})[name] = _summary(group)
        print(f"Robustness: {case.name} complete ({len(report['rows'])} episodes)", flush=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"Robustness report: {path.resolve()}", flush=True)
    return report
