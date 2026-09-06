"""Sensor-policy CLI routing and actionable failures without training or rendering."""

import json
import sys

import numpy as np
import pytest

from shodo import cli
from shodo.contracts import SensorConfig


def _controller(sensors=None):
    def policy(obs):
        return np.zeros(6, dtype=np.float32)

    policy.sensor_config = sensors
    return policy


def _argv(monkeypatch, directory, *arguments):
    monkeypatch.setattr(sys, "argv", ["shodo", *arguments, "--run-dir", str(directory)])


@pytest.mark.parametrize("command", ["train", "ppo"])
@pytest.mark.parametrize("use_file", [False, True])
def test_sensor_training_routes_explicit_contract_and_json_settings(
    tmp_path, monkeypatch, command, use_file
):
    from shodo import rl

    calls = []
    monkeypatch.setattr(cli, "train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(rl, "train_ppo", lambda *args, **kwargs: calls.append(kwargs))
    options = ["--observation", "sensor"]
    sensors = SensorConfig()
    if use_file:
        sensors = SensorConfig(history=2, latency_steps=1, force_noise=0.03)
        path = tmp_path / "sensors.json"
        path.write_text(json.dumps(sensors.to_dict()))
        options = ["--sensor-config", str(path)]
    _argv(monkeypatch, tmp_path / "output", command, *options, "--chars", "一二", "--seed", "17")
    cli.main()
    assert len(calls) == 1
    assert calls[0]["sensors"] == sensors
    assert calls[0]["chars"] == "一二"
    assert calls[0]["seed"] == 17
    assert calls[0]["device"] == "cpu"


@pytest.mark.parametrize("policy", ["classical", "oracle", "zero"])
def test_record_uses_sensor_inputs_and_raw_camera_without_ffmpeg(tmp_path, monkeypatch, policy):
    from shodo import runtime

    calls = []
    monkeypatch.setattr(
        runtime, "record_episodes", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native camera data recording must not need FFmpeg or a checkpoint")

    monkeypatch.setattr(cli, "require_ffmpeg", forbidden)
    monkeypatch.setattr(cli, "load_policy", forbidden)
    _argv(
        monkeypatch,
        tmp_path,
        "record",
        "--policy",
        policy,
        "--chars",
        "自在",
        "--episodes",
        "2",
        "--camera-every",
        "30",
    )
    cli.main()
    args, kwargs = calls[0]
    assert args == (tmp_path / "episodes",)
    assert kwargs["policy"] == policy
    assert kwargs["sensors"] == SensorConfig()
    assert kwargs["camera_every"] == 30
    assert kwargs["episodes"] == 2 and kwargs["chars"] == "自在"


@pytest.mark.parametrize("policy_name", ["learned", "ppo"])
def test_record_infers_sensor_configuration_from_loaded_checkpoint(
    tmp_path, monkeypatch, policy_name
):
    from shodo import runtime

    sensors = SensorConfig(history=2, position_noise=0.0001)
    controller = _controller(sensors)
    loads, recordings = [], []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return controller

    monkeypatch.setattr(cli, "load_policy", load)
    monkeypatch.setattr(cli, "load_ppo", load)
    monkeypatch.setattr(
        runtime, "record_episodes", lambda *args, **kwargs: recordings.append(kwargs)
    )
    _argv(monkeypatch, tmp_path, "record", "--policy", policy_name, "--episodes", "1")
    cli.main()
    assert len(loads) == 1
    assert recordings[0]["sensors"] == sensors
    assert recordings[0]["policy"] is controller


def test_evaluation_routes_sensor_settings_without_duplicate_checkpoint_load(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: calls.append(kwargs))

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate owns checkpoint loading")

    monkeypatch.setattr(cli, "load_policy", forbidden)
    _argv(monkeypatch, tmp_path, "evaluate", "--observation", "sensor", "--chars", "水")
    cli.main()
    # Selecting the mode does not overwrite a checkpoint's history/corruption settings.
    assert calls[0]["sensors"] is None
    assert calls[0]["observation_mode"] == "sensor"
    assert calls[0]["algorithm"] == "learned"


def test_robustness_routes_paired_seed_list_and_sensor_checkpoint(tmp_path, monkeypatch):
    from shodo import robustness

    controller = _controller(SensorConfig(history=2))
    monkeypatch.setattr(cli, "load_policy", lambda *args, **kwargs: controller)
    calls = []
    monkeypatch.setattr(
        robustness, "evaluate_robustness", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    _argv(monkeypatch, tmp_path, "robustness", "--chars", "永水", "--seeds", "11", "23")
    cli.main()
    args, kwargs = calls[0]
    assert args == (tmp_path / "robustness.json",)
    assert kwargs["seeds"] == (11, 23)
    assert kwargs["sensors"] == controller.sensor_config
    assert kwargs["policy"] is controller
    assert kwargs["chars"] == "永水"


@pytest.mark.parametrize(
    "options,message",
    [
        (["record", "--observation", "privileged"], "require sensor observations"),
        (["train", "--camera-every", "1"], "only to record"),
        (["record", "--camera-every", "-1"], "nonnegative"),
        (["train", "--seeds", "1"], "only to robustness"),
        (["validate", "--observation", "sensor"], "canonical validate"),
        (["benchmark", "--observation", "sensor"], "canonical validate"),
    ],
)
def test_invalid_sensor_command_flags_fail_before_creating_outputs(
    tmp_path, monkeypatch, capsys, options, message
):
    destination = tmp_path / "unused"
    _argv(monkeypatch, destination, *options)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("contents", ["not JSON", "[]", '{"history": 0}', '{"unknown": 1}'])
def test_invalid_sensor_configuration_file_is_actionable(tmp_path, monkeypatch, capsys, contents):
    path = tmp_path / "sensors.json"
    path.write_text(contents)
    destination = tmp_path / "unused"
    _argv(monkeypatch, destination, "train", "--sensor-config", str(path))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Invalid sensor configuration" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("command", ["record", "robustness"])
def test_privileged_checkpoint_rejected_for_sensor_workflow(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setattr(cli, "load_policy", lambda *args, **kwargs: _controller())
    _argv(monkeypatch, tmp_path, command)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Privileged checkpoint cannot use sensor observations" in capsys.readouterr().err


def test_sensor_checkpoint_rejected_for_explicit_privileged_demo(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(cli, "load_policy", lambda *args, **kwargs: _controller(SensorConfig()))
    _argv(monkeypatch, tmp_path, "demo", "--observation", "privileged")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Sensor checkpoint cannot use privileged observations" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure", [FileNotFoundError("missing checkpoint"), ValueError("invalid contract")]
)
def test_checkpoint_loading_errors_are_actionable(tmp_path, monkeypatch, capsys, failure):
    def load(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cli, "load_policy", load)
    _argv(monkeypatch, tmp_path, "record")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code in (1, 2)
    assert str(failure) in capsys.readouterr().err


def test_sensor_checkpoint_history_mismatch_rejected_before_recording(
    tmp_path, monkeypatch, capsys
):
    from shodo import runtime

    monkeypatch.setattr(
        cli, "load_policy", lambda *args, **kwargs: _controller(SensorConfig(history=2))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("contract mismatch must fail before recording")

    monkeypatch.setattr(runtime, "record_episodes", forbidden)
    sensor_file = tmp_path / "sensors.json"
    sensor_file.write_text(json.dumps({"history": 4}))
    _argv(monkeypatch, tmp_path, "record", "--sensor-config", str(sensor_file))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code in (1, 2)
    assert "history" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("seeds", [["-1"], ["7", "7"]])
def test_invalid_robustness_seeds_rejected_before_evaluation(tmp_path, monkeypatch, capsys, seeds):
    from shodo import robustness

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid seeds must fail before evaluation")

    monkeypatch.setattr(robustness, "evaluate_robustness", forbidden)
    _argv(monkeypatch, tmp_path, "robustness", "--policy", "classical", "--seeds", *seeds)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "seed" in capsys.readouterr().err.lower()


def test_classical_policy_does_not_silently_override_privileged_mode(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("conflicting observation mode must fail before evaluation")

    monkeypatch.setattr(cli, "evaluate", forbidden)
    _argv(monkeypatch, tmp_path, "evaluate", "--policy", "classical", "--observation", "privileged")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "sensor" in capsys.readouterr().err.lower()
