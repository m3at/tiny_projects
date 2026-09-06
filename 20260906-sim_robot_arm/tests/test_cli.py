"""Exercise command routing and checkpoint configuration without expensive rollouts."""

import json
import sys
from dataclasses import replace

import pytest

from shodo import cli
from shodo.config import SimConfig


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
    _, kwargs = calls[0]
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
