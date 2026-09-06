import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shodo import vla_benchmark
from shodo.vla_benchmark import _denoising_sweep, _replay, latency_summary
from shodo.vla_inference import FrameCache, PromptCache, cached_vision


def test_prompt_cache_keys_tasks_and_tokenizer_options():
    calls = []

    def tokenize(tasks, **kwargs):
        calls.append((tasks, kwargs))
        return {"input_ids": torch.tensor([[len(calls)]])}

    cached = PromptCache(tokenize)
    first = cached(["Draw 一"], max_length=48, padding="max_length")
    assert cached(["Draw 一"], padding="max_length", max_length=48) is first
    assert len(calls) == 1
    assert cached(["Draw 二"], max_length=48, padding="max_length") is not first
    cached(["Draw 二"], max_length=32, padding="max_length")
    assert len(calls) == 3


def test_prompt_trim_only_removes_globally_masked_suffix_without_mutation():
    ids = torch.arange(12).reshape(2, 6)
    mask = torch.tensor([[0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    original = {"input_ids": ids, "attention_mask": mask}
    cache = PromptCache(lambda tasks, **kwargs: original, trim_padding=True)
    actual = cache(["first", "second"])
    assert actual["input_ids"].shape == (2, 4)
    torch.testing.assert_close(actual["input_ids"], ids[:, :4])
    torch.testing.assert_close(actual["attention_mask"], mask[:, :4])
    # State and suffix positions depend on valid counts, not physical padding width.
    torch.testing.assert_close(actual["attention_mask"].sum(1), mask.sum(1))
    assert original["input_ids"].shape == original["attention_mask"].shape == (2, 6)
    assert cache(["first", "second"]) is actual
    cache.clear()
    assert cache.tokens is None


def test_prompt_trim_handles_all_masked_batch_and_validates_shape():
    cache = PromptCache(
        lambda tasks, **kwargs: {
            "input_ids": torch.zeros(1, 4),
            "attention_mask": torch.zeros(1, 4),
        },
        trim_padding=True,
    )
    assert cache([""])["input_ids"].shape == (1, 1)
    with pytest.raises(ValueError, match="boolean"):
        PromptCache(None, trim_padding=1)
    bad = PromptCache(
        lambda tasks, **kwargs: {
            "input_ids": torch.zeros(1, 4),
            "attention_mask": torch.zeros(2, 4),
        },
        trim_padding=True,
    )
    with pytest.raises(ValueError, match="matching"):
        bad(["x"])


@torch.no_grad()
def test_frame_cache_only_reuses_one_immutable_acquisition():
    calls = []

    def encode(image):
        calls.append(image.clone())
        return image * 2

    cache = FrameCache(encode)
    with pytest.raises(ValueError, match="acquisition"):
        cache(torch.ones(2))
    cache.frame = 0
    first = cache(torch.ones(2))
    assert cache(torch.ones(2)) is first
    cache.frame = 1
    torch.testing.assert_close(cache(torch.full((2,), 3)), torch.full((2,), 6))
    cache.frame = 0
    torch.testing.assert_close(cache(torch.ones(2)), first)
    assert (cache.hits, cache.misses) == (1, 3)
    cache.clear()
    assert cache.value is None and cache.frame is None
    assert cache.hits == cache.misses == 0


@torch.no_grad()
def test_scoped_vision_cache_preserves_changing_state_and_restores_after_error():
    class Vision:
        def embed_image(self, image):
            return image * 2

    vision = Vision()
    original = vision.embed_image
    model = SimpleNamespace(model=SimpleNamespace(vlm_with_expert=vision))
    image = torch.tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="test"), cached_vision(model) as cache:
        cache.frame = 0
        for state in (torch.tensor([3.0, 4.0]), torch.tensor([-1.0, 5.0])):
            # Current state participates after vision encoding, never in its cache.
            torch.testing.assert_close(vision.embed_image(image) + state, original(image) + state)
        assert cache.hits == 1
        raise RuntimeError("test")
    assert vision.embed_image == original


def test_vision_cache_refuses_autograd_and_training_mode():
    class Encoder(torch.nn.Module):
        def forward(self, image):
            return image * 2

    encoder = Encoder().eval()
    cache = FrameCache(encoder.forward)
    cache.frame = 0
    with pytest.raises(RuntimeError, match="evaluation"):
        cache(torch.ones(2))
    with torch.no_grad():
        torch.testing.assert_close(cache(torch.ones(2)), torch.full((2,), 2.0))
        encoder.train()
        with pytest.raises(RuntimeError, match="evaluation"):
            cache(torch.ones(2))


def test_latency_summary_reports_percentiles_without_dropping_outliers():
    result = latency_summary([0.001, 0.002, 0.003, 0.1])
    assert result["samples"] == 4
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] == pytest.approx(85.45)
    assert result["seconds"][-1] == 0.1
    assert result["calls_over_20ms"] == 1
    for invalid in ([], [-1], [np.nan], [[0.1]]):
        with pytest.raises(ValueError, match="Timings"):
            latency_summary(invalid)


@pytest.mark.parametrize("objective", ["flow_matching", "action_regression"])
def test_replay_cache_matches_changing_state_fresh_images_and_fixed_noise(objective):
    class Vision(torch.nn.Module):
        def embed_image(self, image):
            return image.mean()

    class Policy:
        def __init__(self):
            self.model = SimpleNamespace(vlm_with_expert=Vision().eval())

        def reset(self):
            pass

        def predict_action_chunk(self, inputs, noise):
            assert bool(torch.count_nonzero(noise)) == (objective == "flow_matching")
            return (
                noise[:, :, :6]
                + inputs["observation.state"][:, None, :6]
                + self.model.vlm_with_expert.embed_image(inputs["observation.images.camera1"])
            )

    def tokenize(tasks, **kwargs):
        return {"input_ids": torch.ones(1, 2), "attention_mask": torch.ones(1, 2)}

    observations = np.arange(6 * 39, dtype=np.float32).reshape(6, 39) / 100
    episode = SimpleNamespace(
        metadata={"char": "一"},
        arrays={
            "observations": observations,
            "observation_times_s": np.arange(6) * 0.02,
            "camera_times_s": np.array([0.0, 0.1]),
            "camera_frames": np.stack(
                [np.zeros((3, 4, 3), dtype=np.uint8), np.full((3, 4, 3), 255, dtype=np.uint8)]
            ),
        },
    )
    report = {
        "objective": objective,
        "chunk_size": 4,
        "dataset": {
            "sensors": {"history": 1},
            "image_preprocessing": {"size": 4},
            "stats": {
                "state": {"mean": [0.0] * 32, "std": [1.0] * 32},
                "actions": {"mean": [0.0] * 6, "std": [1.0] * 6},
            },
        },
    }
    model, indices = Policy(), np.array([0, 1, 5])
    _, _, expected, fresh = _replay(model, tokenize, report, episode, indices, "cpu")
    with cached_vision(model) as cache:
        total, inference, actual, other_fresh = _replay(
            model, PromptCache(tokenize), report, episode, indices, "cpu", cache=cache
        )
        assert (cache.hits, cache.misses) == (1, 2)
    np.testing.assert_array_equal(actual, expected)
    assert fresh == other_fresh == [True, False, True]
    assert all(whole >= inner > 0 for whole, inner in zip(total, inference, strict=True))
    assert not np.array_equal(actual[0], actual[1])


@pytest.mark.parametrize(
    "report,requested,expected",
    [
        ({}, None, (1, 2, 4, 10)),
        ({"objective": "flow_matching"}, None, (1, 2, 4, 10)),
        ({"objective": "flow_matching"}, [2, 4], (2, 4)),
        ({"objective": "action_regression"}, None, (1,)),
        ({"objective": "action_regression"}, [1], (1,)),
    ],
)
def test_denoising_sweep_defaults_follow_adapter_objective(report, requested, expected):
    assert _denoising_sweep(report, requested) == expected


@pytest.mark.parametrize("requested", [[2], [1, 2], [10], [0], [True], [None], [], 1])
def test_invalid_action_regression_sweep_rejected_before_episode_or_model_loading(
    tmp_path, monkeypatch, requested
):
    (tmp_path / "training.json").write_text(json.dumps({"objective": "action_regression"}))

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must reject the sampler before loading")

    monkeypatch.setattr(vla_benchmark, "load_adapter", forbidden)
    monkeypatch.setattr(vla_benchmark, "load_episode", forbidden)
    with pytest.raises(ValueError):
        vla_benchmark.benchmark(tmp_path, "unused.npz", denoise_steps=requested)
