"""Paired reporting preserves failures and never imputes absent ink as zero error."""

import json
from dataclasses import FrozenInstanceError

import pytest

from shodo.config import BrushConfig, SimConfig
from shodo.contracts import SensorConfig
from shodo.robustness import RobustnessCase, default_cases, evaluate_robustness


def cases():
    return (
        RobustnessCase("nominal", SensorConfig(), "Nominal sensors"),
        RobustnessCase("delay", SensorConfig(latency_steps=2), "40 ms latency"),
    )


def test_pairs_are_matched_and_failures_are_not_dropped(tmp_path):
    calls = []

    def runner(char, *, policy, seed, sensors, **kwargs):
        calls.append((char, policy, seed, sensors.latency_steps))
        failure = sensors.latency_steps > 0 and seed == 17 and policy == "classical"
        delta = 0 if policy == "oracle" else sensors.latency_steps
        return {
            "ink_rmse_mm": None if failure else seed + delta,
            "force_rmse_n": 0.2 + delta,
            "max_force_n": 0.5 + delta,
            "raster_coverage_fraction": 0.0 if failure else 1.0,
            "truncated": failure,
            "terminated": not failure,
            "steps": 5 if failure else 10,
            "expected_steps": 10,
        }

    output = tmp_path / "report.json"
    result = evaluate_robustness(output, chars="永", seeds=(7, 17), cases=cases(), runner=runner)
    assert len(calls) == 12
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    delayed = result["summary"]["delay"]["classical"]
    assert delayed["episodes"] == 2
    assert delayed["truncated"] == delayed["incomplete"] == delayed["missing_ink"] == 1
    assert delayed["paired_nominal"]["ink_rmse_mm"] == {
        "valid_pairs": 1,
        "mean": 2.0,
        "min": 2.0,
        "max": 2.0,
    }
    oracle = result["summary"]["delay"]["oracle"]
    assert oracle["paired_nominal"]["ink_rmse_mm"]["mean"] == 0
    missing = [row for row in result["rows"] if row["missing_ink"]]
    assert len(missing) == 1
    assert missing[0]["delta_from_nominal"]["ink_rmse_mm"] is None


def test_learned_state_resets_before_each_episode(tmp_path):
    class Policy:
        resets = 0
        sensor_config = SensorConfig()

        def reset(self):
            self.resets += 1

        def __call__(self, observation):
            return observation

    controller = Policy()
    controller.checkpoint_provenance = {"sha256": "test-checkpoint"}

    def runner(char, *, policy, **kwargs):
        if callable(policy):
            policy.reset()
        return {"ink_rmse_mm": 1.0, "truncated": False, "steps": 10}

    report = evaluate_robustness(
        tmp_path / "learned.json",
        chars="永水",
        seeds=(7,),
        cases=cases(),
        policy=controller,
        runner=runner,
    )
    assert controller.resets == 4
    assert report["controllers"] == ["classical", "oracle", "zero", "learned"]
    assert report["checkpoint_provenance"] == controller.checkpoint_provenance


def test_policy_sensor_settings_are_inferred(tmp_path):
    def policy(observation):
        return observation

    policy.sensor_config = SensorConfig(history=2)

    def runner(char, *, sensors, **kwargs):
        assert sensors.history == 2
        return {"ink_rmse_mm": 1.0, "truncated": False, "steps": 10}

    report = evaluate_robustness(
        tmp_path / "inferred.json", chars="永", seeds=(7,), policy=policy, runner=runner
    )
    assert all(case["sensors"]["history"] == 2 for case in report["cases"])


@pytest.mark.parametrize("declared", [False, True])
def test_policy_contract_fails_before_any_baseline_episode(tmp_path, declared):
    def policy(observation):
        return observation

    if declared:
        policy.sensor_config = SensorConfig(history=2)

    def unexpected(*args, **kwargs):
        pytest.fail("Preflight must precede every baseline episode")

    with pytest.raises(ValueError, match="history lengths" if declared else "sensor_config"):
        evaluate_robustness(
            tmp_path / "bad.json",
            chars="永",
            seeds=(7,),
            policy=policy,
            cases=cases(),
            runner=unexpected,
        )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"chars": ""}, "individual characters"),
        ({"chars": "永永"}, "unique"),
        ({"seeds": ()}, "nonnegative"),
        ({"seeds": (-1,)}, "nonnegative"),
        ({"seeds": (True,)}, "nonnegative"),
        ({"seeds": (7, 7)}, "unique"),
        ({"cases": ()}, "start with nominal"),
        ({"cases": tuple(reversed(cases()))}, "start with nominal"),
        ({"cases": (cases()[0], cases()[0])}, "unique names"),
        ({"policy": "old-checkpoint.pt"}, "sensor-compatible"),
    ],
)
def test_invalid_protocol_fails_before_rollouts(tmp_path, options, match):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid protocol must not execute")

    settings = {"chars": "永", "seeds": (7,), "cases": cases(), "runner": unexpected}
    settings.update(options)
    with pytest.raises(ValueError, match=match):
        evaluate_robustness(tmp_path / "invalid.json", **settings)
    assert not (tmp_path / "invalid.json").exists()


def test_exception_does_not_overwrite_existing_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("existing report\n")

    def runner(*args, **kwargs):
        raise RuntimeError("integration failed")

    with pytest.raises(RuntimeError, match="integration failed"):
        evaluate_robustness(output, chars="永", seeds=(7,), cases=cases(), runner=runner)
    assert output.read_text() == "existing report\n"


def test_default_cases_are_fixed_immutable_sensor_errors():
    base = SensorConfig(history=2)
    suite = default_cases(base)
    assert suite == default_cases(base)
    assert suite[0].name == "nominal" and suite[0].sensors is base
    assert len({case.name for case in suite}) == len(suite)
    assert all(case.sensors.history == 2 for case in suite)
    assert not any(case.material for case in suite)
    by_name = {case.name: case for case in suite}
    assert by_name["paper_height_negative"].sensors.reference_offset == (0, 0, -0.001)
    assert by_name["latency"].sensors.latency_steps == 2
    with pytest.raises(FrozenInstanceError):
        suite[0].name = "changed"
    with pytest.raises(FrozenInstanceError):
        suite[0].sensors.dropout = 0.5


def test_material_cases_do_not_apply_inoperative_native_stiffness():
    reduced = default_cases(include_material=True)
    assert dict(reduced[-2].material)["normal_stiffness"] == 176
    native = SimConfig(timestep=0.0001, substeps=200, brush=BrushConfig(backend="cable"))
    cases = default_cases(config=native, include_material=True)
    assert all(set(dict(case.material)) == {"friction"} for case in cases[-2:])
