"""Sensor PPO checkpoints cannot silently cross observation or corruption contracts."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shodo import rl
from shodo.config import SimConfig
from shodo.contracts import SENSOR_FEATURES, SensorConfig, sensor_contract
from shodo.data import TRAIN
from shodo.env import ACTIONS, OBSERVATION_VERSION, OBSERVATIONS
from shodo.learning import network


def metadata(sensors=None):
    return {
        "seed": 7,
        "train_chars": TRAIN,
        "observation_version": OBSERVATION_VERSION,
        "config": SimConfig(randomize=True).to_dict(),
        "residual_scale": 0.0,
        "objective": "tracking",
        "actual_steps": 2048,
        "source_snapshot": "source-test",
        "sensors": sensors.to_dict() if sensors else None,
        "sensor_contract": sensor_contract(sensors) if sensors else None,
    }


def write_metadata(directory, value):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ppo.json").write_text(json.dumps(value))
    (directory / "ppo.zip").write_bytes(b"mock PPO checkpoint")


@pytest.mark.parametrize("sensors", [None, SensorConfig(history=2)])
def test_load_controller_uses_saved_observation_contract(tmp_path, monkeypatch, sensors):
    value = metadata(sensors)
    if sensors is None:
        value.pop("sensors")
        value.pop("sensor_contract")
    write_metadata(tmp_path, value)
    size = SENSOR_FEATURES * sensors.history if sensors else OBSERVATIONS
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(size,)),
        action_space=SimpleNamespace(shape=(ACTIONS,)),
        predict=lambda observation, deterministic: (np.ones(ACTIONS) * 0.2, None),
    )
    monkeypatch.setattr(rl.PPO, "load", lambda *args, **kwargs: model)
    policy = rl.load_controller(tmp_path)
    assert policy.sensor_config == sensors
    assert (
        policy.checkpoint_provenance["sha256"] == hashlib.sha256(b"mock PPO checkpoint").hexdigest()
    )
    assert (
        policy.checkpoint_provenance["metadata_sha256"]
        == hashlib.sha256((tmp_path / "ppo.json").read_bytes()).hexdigest()
    )
    np.testing.assert_allclose(policy(np.zeros(size)), 0.2)


def test_corrupt_contract_fails_before_loading_ppo(tmp_path, monkeypatch):
    value = metadata(SensorConfig())
    value["sensor_contract"]["version"] = -1
    write_metadata(tmp_path, value)

    def unexpected(*args, **kwargs):
        pytest.fail("Contract must be validated before model loading")

    monkeypatch.setattr(rl.PPO, "load", unexpected)
    with pytest.raises(ValueError, match="sensor contract"):
        rl.load_controller(tmp_path)


def test_saved_sensor_shape_is_checked(tmp_path, monkeypatch):
    write_metadata(tmp_path, metadata(SensorConfig(history=2)))
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(OBSERVATIONS,)),
        action_space=SimpleNamespace(shape=(ACTIONS,)),
    )
    monkeypatch.setattr(rl.PPO, "load", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="shape mismatch"):
        rl.load_controller(tmp_path)


@pytest.mark.parametrize("sensors", [None, SensorConfig(force_bias=(0, 0, 0.1))])
def test_resume_normalizes_json_arrays_and_supports_legacy(tmp_path, monkeypatch, sensors):
    value = metadata(sensors)
    if sensors is None:
        value.pop("sensors")
        value.pop("sensor_contract")
    write_metadata(tmp_path, value)

    def reached_snapshot(*args, **kwargs):
        raise RuntimeError("metadata accepted")

    monkeypatch.setattr(rl, "snapshot_source", reached_snapshot)
    with pytest.raises(RuntimeError, match="metadata accepted"):
        rl.train_ppo(tmp_path, sensors=sensors, resume=True)


@pytest.mark.parametrize("sensors", [None, SensorConfig(history=1), SensorConfig(force_noise=0.03)])
def test_resume_rejects_sensor_changes_without_writing(tmp_path, monkeypatch, sensors):
    write_metadata(tmp_path, metadata(SensorConfig()))
    original = (tmp_path / "ppo.json").read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("Mismatch must precede source snapshot or simulator creation")

    monkeypatch.setattr(rl, "snapshot_source", unexpected)
    with pytest.raises(ValueError, match="changing sensors"):
        rl.train_ppo(tmp_path, sensors=sensors, resume=True)
    assert (tmp_path / "ppo.json").read_bytes() == original


@pytest.mark.parametrize("sensors", [None, SensorConfig(history=1), SensorConfig(dropout=0.1)])
def test_base_mismatch_precedes_creation_of_output(tmp_path, sensors):
    base = tmp_path / "base.pt"
    settings = SensorConfig()
    torch.save(
        {
            "state_dict": network(SENSOR_FEATURES * settings.history).state_dict(),
            "observation_version": OBSERVATION_VERSION,
            "sensors": settings.to_dict(),
            "sensor_contract": sensor_contract(settings),
        },
        base,
    )
    destination = tmp_path / "new-run"
    with pytest.raises(ValueError, match="Residual base sensor configuration"):
        rl.train_ppo(destination, sensors=sensors, base_checkpoint=base)
    assert not destination.exists()


def test_residual_load_rejects_different_base_sensor_settings(tmp_path, monkeypatch):
    sensors = SensorConfig()
    value = metadata(sensors)
    value["residual_scale"] = 0.3
    checkpoint = tmp_path / "bc.pt"
    checkpoint.write_bytes(b"checkpoint")
    value["base_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_metadata(tmp_path, value)
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(SENSOR_FEATURES * sensors.history,)),
        action_space=SimpleNamespace(shape=(ACTIONS,)),
    )
    monkeypatch.setattr(rl.PPO, "load", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        rl,
        "load_policy",
        lambda *args, **kwargs: SimpleNamespace(sensor_config=replace(sensors, dropout=0.1)),
    )
    with pytest.raises(ValueError, match="Residual base sensor configuration"):
        rl.load_controller(tmp_path)
