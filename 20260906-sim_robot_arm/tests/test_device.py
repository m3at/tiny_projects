import numpy as np
import pytest
import torch

from shodo.config import SimConfig
from shodo.device import resolve_device
from shodo.env import ACTIONS, OBSERVATION_VERSION, OBSERVATIONS
from shodo.learning import load_policy, network, train
from shodo.rebot import ROBOT_CONTRACT


def test_explicit_cpu_does_not_initialize_accelerator_detection(monkeypatch):
    def unexpected_probe():
        raise AssertionError("CPU selection must not probe an accelerator")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_probe)
    assert str(resolve_device("cpu")) == "cpu"


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_automatic_device_selection(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert str(resolve_device()) == expected
    assert str(resolve_device("cpu")) == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_fails_before_training_output(
    tmp_path, monkeypatch, device
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    destination = tmp_path / "unused" / "bc.pt"
    with pytest.raises(ValueError, match="unavailable"):
        train(output=destination, device=device)
    assert not destination.parent.exists()


def test_cpu_checkpoint_loads_on_selected_device_and_returns_numpy(tmp_path):
    checkpoint = tmp_path / "bc.pt"
    torch.save(
        {
            "state_dict": network().state_dict(),
            "observation_version": OBSERVATION_VERSION,
            "robot_contract": ROBOT_CONTRACT,
            "robot_config": SimConfig().to_dict()["robot"],
        },
        checkpoint,
    )
    observation = np.zeros(OBSERVATIONS, dtype=np.float32)
    cpu = load_policy(checkpoint, device="cpu")
    automatic = load_policy(checkpoint, device="auto")
    result = automatic(observation)
    assert result.shape == (ACTIONS,)
    assert result.dtype == np.float32
    assert automatic.device == str(resolve_device())
    assert automatic.requested_device == "auto"
    np.testing.assert_allclose(result, cpu(observation), atol=1e-6)
