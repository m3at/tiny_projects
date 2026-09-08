"""Recorded-state scoring tests without optional models, network access or GPU work."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
import torch

from shodo import smolvla, vla_data
from shodo.vla_evaluation import action_metrics, score_adapter


class FakeDataset:
    state_mean = np.full(32, 50, np.float32)
    state_std = np.full(32, 10, np.float32)
    action_mean = np.full(6, 0.5, np.float32)
    action_std = np.full(6, 0.1, np.float32)

    def __init__(self, directory, chunk_size):
        assert chunk_size == 1
        self.manifest = {
            "state_contract": {"name": "sensor-state"},
            "action_contract": {"frame": "world"},
            "image_preprocessing": {"size": 4},
            "observation_contract": {"name": "sensor-history"},
            "episodes": [{"sha256": "evaluation-episode", "char": "一", "source": "episode.npz"}],
            "chars": ["一"],
            "supervision": {"source": "expert"},
        }

    def __len__(self):
        return 8

    def __getitem__(self, index):
        # All values normalized with validation-only statistics. The scorer must
        # recover originals and normalize actor inputs with training statistics.
        return {
            "state": (np.full(32, 2 + index, np.float32) - self.state_mean) / self.state_std,
            "actions": (
                (np.full((1, 6), 0.02 * index, np.float32) - self.action_mean) / self.action_std
            ),
            "image": np.full((3, 4, 4), index / 10, np.float32),
            "task": "Draw 一",
            "action_is_pad": np.array([False]),
        }


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []
        self.noises = []

    def predict_action_chunk(self, batch, noise):
        self.inputs.append(batch)
        self.noises.append(noise.clone())
        count = len(batch["observation.state"])
        chunk = torch.full((count, 4, 6), 99.0)
        # Raw state feature=2+i; adapter normalization mean2,std2 -> i/2.
        # Raw target .02*i; adapter action normalization mean.1,std.2.
        chunk[:, 0] = (batch["observation.state"][:, :6] * 0.04 - 0.1) / 0.2
        return chunk


@pytest.fixture
def scoring(monkeypatch):
    data = FakeDataset("unused", 1)
    report = {
        "adapter_sha256": "adapter-checksum",
        "chunk_size": 4,
        "dataset": {
            **deepcopy(data.manifest),
            "episodes": [{"sha256": "training-episode", "char": "一", "source": "train.npz"}],
            "stats": {
                "state": {"mean": [2] * 32, "std": [2] * 32},
                "actions": {"mean": [0.1] * 6, "std": [0.2] * 6},
            },
        },
    }
    model = FakeModel()
    loads = []

    def load(*args, **kwargs):
        loads.append(True)
        return model, object(), report

    def batch(inputs, tokenizer, device):
        return {"observation.state": inputs["state"], "observation.images.camera1": inputs["image"]}

    monkeypatch.setattr(vla_data, "PreparedDataset", lambda *args, **kwargs: data)
    monkeypatch.setattr(smolvla, "_read_adapter_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(smolvla, "load_adapter", load)
    monkeypatch.setattr(smolvla, "model_batch", batch)
    return data, report, model, loads


def test_adapter_normalization_first_action_only_and_reproducible_sampling(scoring):
    data, _, model, _ = scoring
    result = score_adapter("adapter", "data", samples=5, batch_size=2, seed=17)
    assert result["prediction"]["rmse"] < 1e-6
    assert result["prediction"]["samples"] == 5
    assert result["training_mean_baseline"]["rmse"] > 0.01
    assert result["characters_seen_in_training"] == ["一"]
    assert "teacher-forced" in result["split"]
    assert "not closed-loop" in result["scope"]
    indices = result["indices"]
    assert len(set(indices)) == 5
    states = torch.cat([batch["observation.state"] for batch in model.inputs]).numpy()
    np.testing.assert_allclose(states[:, 0], np.array(indices) / 2, atol=1e-5)
    images = torch.cat([batch["observation.images.camera1"] for batch in model.inputs]).numpy()
    np.testing.assert_allclose(images[:, 0, 0, 0], np.array(indices) / 10)
    first_noise = torch.cat(model.noises)
    model.inputs.clear()
    model.noises.clear()
    repeated = score_adapter("adapter", "data", samples=5, batch_size=2, seed=17)
    assert repeated["indices"] == indices
    torch.testing.assert_close(torch.cat(model.noises), first_noise)
    all_rows = score_adapter("adapter", "data", samples=99)
    assert all_rows["prediction"]["samples"] == len(data)


@pytest.mark.parametrize(
    "contract", ["state_contract", "action_contract", "image_preprocessing", "observation_contract"]
)
def test_contract_mismatch_rejected_before_model_load(scoring, contract):
    data, _, _, loads = scoring
    data.manifest[contract] = {"different": True}
    with pytest.raises(ValueError, match=contract):
        score_adapter("adapter", "data")
    assert not loads


def test_episode_overlap_rejected_before_model_load(scoring):
    data, _, _, loads = scoring
    data.manifest["episodes"][0]["sha256"] = "training-episode"
    with pytest.raises(ValueError, match="separate recording"):
        score_adapter("adapter", "data")
    assert not loads


@pytest.mark.parametrize("kind", ["shape", "nonfinite"])
def test_malformed_predicted_chunks_rejected(scoring, kind):
    model = scoring[2]
    model.predict_action_chunk = lambda batch, noise: (
        torch.zeros(len(noise), 2, 6)
        if kind == "shape"
        else torch.full((len(noise), 4, 6), float("nan"))
    )
    with pytest.raises(ValueError, match="finite.*action chunks"):
        score_adapter("adapter", "data")


def test_action_metrics_known_error():
    result = action_metrics(np.ones((2, 6)), np.zeros((2, 6)))
    assert result["rmse"] == result["mae"] == result["max_absolute_error"] == 1
    assert result["per_axis_rmse"] == [1] * 6


@pytest.mark.parametrize("objective,steps", [("flow_matching", 10), ("action_regression", 1)])
def test_objective_controls_inference_noise_and_default_steps(scoring, objective, steps):
    _, report, model, _ = scoring
    report["objective"] = objective
    result = score_adapter("adapter", "data", samples=3)
    assert result["objective"] == objective
    assert result["denoise_steps"] == steps
    noise = torch.cat(model.noises)
    if objective == "action_regression":
        assert torch.count_nonzero(noise) == 0
    else:
        assert torch.count_nonzero(noise) > 0


def test_action_regression_rejects_incompatible_steps_before_model_load(scoring):
    _, report, _, loads = scoring
    report["objective"] = "action_regression"
    with pytest.raises(ValueError):
        score_adapter("adapter", "data", denoise_steps=2)
    assert not loads


def test_ancestor_episode_overlap_rejected_before_model_load(scoring):
    _, report, _, loads = scoring
    report["training_episode_sha256"] = ["training-episode", "evaluation-episode"]
    with pytest.raises(ValueError, match="separate recording"):
        score_adapter("adapter", "data")
    assert not loads


def test_legacy_ancestry_is_verified_and_unioned(tmp_path):
    ancestor = {
        "dataset": {"episodes": [{"sha256": "earlier-episode"}]},
        "adapter_sha256": "parent-weights",
        "training_episode_sha256": ["earlier-episode", "oldest-episode"],
    }
    parent_path = tmp_path / "training.json"
    parent_path.write_text(json.dumps(ancestor))
    child = {
        "dataset": {"episodes": [{"sha256": "current-episode"}]},
        "warm_start": {
            "directory": str(tmp_path),
            "training_manifest_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "adapter_sha256": "parent-weights",
        },
    }
    expected = {"current-episode", "earlier-episode", "oldest-episode"}
    assert smolvla.training_episode_hashes(child) == expected
    child["training_episode_sha256"] = sorted(expected)
    parent_path.unlink()
    assert smolvla.training_episode_hashes(child) == expected  # self-contained new artifacts
    child.pop("training_episode_sha256")
    with pytest.raises(ValueError, match="ancestry unavailable"):
        smolvla.training_episode_hashes(child)


@pytest.mark.parametrize("mutation", ["hash", "adapter"])
def test_legacy_ancestry_tampering_rejected(tmp_path, mutation):
    parent_path = tmp_path / "training.json"
    parent_path.write_text(
        json.dumps({"dataset": {"episodes": [{"sha256": "parent"}]}, "adapter_sha256": "actual"})
    )
    child = {
        "dataset": {"episodes": [{"sha256": "child"}]},
        "warm_start": {
            "directory": str(tmp_path),
            "training_manifest_sha256": "wrong"
            if mutation == "hash"
            else hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "adapter_sha256": "wrong" if mutation == "adapter" else "actual",
        },
    }
    with pytest.raises(ValueError, match="mismatch"):
        smolvla.training_episode_hashes(child)


def test_inherited_episode_union_cannot_omit_current_training_episodes():
    with pytest.raises(ValueError, match="inherited"):
        smolvla.training_episode_hashes(
            {
                "dataset": {"episodes": [{"sha256": "current"}]},
                "training_episode_sha256": ["parent-only"],
            }
        )


def test_unavailable_ancestry_rejects_scoring_before_model_load(scoring, tmp_path):
    _, report, _, loads = scoring
    report["warm_start"] = {
        "directory": str(tmp_path / "missing"),
        "training_manifest_sha256": "expected",
        "adapter_sha256": "parent",
    }
    with pytest.raises(ValueError, match="ancestry unavailable"):
        score_adapter("adapter", "data")
    assert not loads
