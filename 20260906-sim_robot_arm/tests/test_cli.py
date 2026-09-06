"""Exercise command routing and checkpoint configuration without expensive rollouts."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from shodo import cli
from shodo.config import SimConfig


@pytest.mark.parametrize("ink,truncated", [(0.1234, False), (None, True)])
def test_demo_prints_curated_metrics_and_artifact_paths(
    tmp_path, monkeypatch, capsys, ink, truncated
):
    monkeypatch.setattr(cli, "require_ffmpeg", lambda: "ffmpeg")
    metrics = {
        "steps": 100,
        "config": {"timestep": 0.002, "substeps": 10},
        "ink_rmse_mm": ink,
        "max_force_n": 0.42,
        "truncated": truncated,
        "frame_times_s": list(range(10000)),
    }
    monkeypatch.setattr(cli, "rollout", lambda *args, **kwargs: (metrics, None, None, []))
    outputs = []

    def save(base, result):
        outputs.append(base)
        Path(f"{base}.mp4").write_bytes(b"video")

    monkeypatch.setattr(cli, "save_rollout", save)
    monkeypatch.setattr(
        sys,
        "argv",
        ["shodo", "demo", "--policy", "expert", "--chars", "自在", "--run-dir", str(tmp_path)],
    )
    cli.main()
    printed = capsys.readouterr().out
    assert "[1/2] Drawing 自" in printed and "[2/2] Drawing 在" in printed
    assert "2.00s simulated" in printed and "peak force 0.420 N" in printed
    assert ("TRUNCATED" if truncated else "Complete") in printed
    assert ("no loaded ink" if ink is None else "ink error 0.123 mm") in printed
    assert "frame_times_s" not in printed and len(printed.splitlines()) <= 16
    for base in outputs:
        for suffix in (".mp4", ".png", ".json", ".npz"):
            assert f"{base}{suffix}" in printed


def test_demo_checks_ffmpeg_before_creating_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("shodo.video.shutil.which", lambda _: None)
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["shodo", "demo", "--run-dir", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "requires FFmpeg" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("policy,metadata", [("learned", "bc.json"), ("ppo", "ppo.json")])
def test_evaluation_reuses_nominal_checkpoint_configuration(
    tmp_path, monkeypatch, policy, metadata
):
    config = replace(SimConfig(), force_feedback_gain=0.02, randomize=True, record=True)
    (tmp_path / metadata).write_text(json.dumps({"config": config.to_dict()}))
    calls = []
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["shodo", "evaluate", "--run-dir", str(tmp_path), "--policy", policy, "--chars", "水"],
    )
    cli.main()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == tmp_path / f"evaluation-{policy}.json"
    assert kwargs["chars"] == "水"
    assert kwargs["algorithm"] == policy
    assert kwargs["config"] == replace(config, randomize=False, record=False)


def test_explicit_configuration_overrides_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "bc.json").write_text("invalid metadata deliberately ignored")
    config_path = tmp_path / "experiment.toml"
    config_path.write_text("force_feedback_gain = 0.01\nrandomize = true\n")
    calls = []
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["shodo", "evaluate", "--run-dir", str(tmp_path), "--config", str(config_path)],
    )
    cli.main()
    assert calls[0]["config"].force_feedback_gain == 0.01
    assert calls[0]["config"].randomize


def test_invalid_checkpoint_configuration_is_actionable(tmp_path, monkeypatch, capsys):
    (tmp_path / "bc.json").write_text(json.dumps({"config": {"timestep": -1}}))
    monkeypatch.setattr(sys, "argv", ["shodo", "evaluate", "--run-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Invalid checkpoint configuration metadata" in capsys.readouterr().err


def test_ink_objective_routes_only_to_ppo(tmp_path, monkeypatch):
    from shodo import rl

    calls = []
    monkeypatch.setattr(rl, "train_ppo", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["shodo", "ppo", "--ink-objective", "--residual", "--run-dir", str(tmp_path)],
    )
    cli.main()
    assert calls[0]["objective"] == "ink"
    assert calls[0]["base_checkpoint"] == tmp_path / "bc.pt"
    calls.clear()
    monkeypatch.setattr(cli, "train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["shodo", "train", "--run-dir", str(tmp_path)])
    cli.main()
    assert "objective" not in calls[0]
    assert calls[0]["device"] == "cpu"


@pytest.mark.parametrize("command", ["train", "ppo"])
def test_neural_training_routes_explicit_device(tmp_path, monkeypatch, command):
    from shodo import rl

    calls = []
    monkeypatch.setattr(cli, "train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(rl, "train_ppo", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys, "argv", ["shodo", command, "--run-dir", str(tmp_path), "--device", "cpu"]
    )
    cli.main()
    assert calls[0]["device"] == "cpu"


def test_unavailable_device_is_actionable_before_creating_outputs(tmp_path, monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    destination = tmp_path / "unused"
    monkeypatch.setattr(
        sys, "argv", ["shodo", "train", "--run-dir", str(destination), "--device", "cuda"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Requested cuda device is unavailable" in capsys.readouterr().err
    assert not destination.exists()


def test_default_run_directory_is_unversioned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["shodo", "train"])
    cli.main()
    assert calls[0]["output"] == cli.Path("runs/bc.pt")


def test_benchmark_routes_character_seed_and_repeats(tmp_path, monkeypatch):
    from shodo import benchmark

    calls = []
    monkeypatch.setattr(benchmark, "benchmark", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["shodo", "benchmark", "--run-dir", str(tmp_path), "--seed", "27", "--repeats", "2"],
    )
    cli.main()
    assert calls[0]["seed"] == 27
    assert calls[0]["repeats"] == 2
    assert calls[0]["char"] == "永"


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["benchmark", "--chars", "永水"], "exactly one character"),
        (["benchmark", "--repeats", "0"], "positive --repeats"),
        (["ppo", "--base-policy", "bc.pt"], "--base-policy requires --residual"),
    ],
)
def test_invalid_command_options_do_not_create_outputs(
    tmp_path, monkeypatch, capsys, arguments, message
):
    destination = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["shodo", *arguments, "--run-dir", str(destination)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not destination.exists()
