from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shodo import smolvla
from shodo.contracts import SensorConfig
from shodo.vla_inference import cached_static_prefix


class Prefix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix_length = 0
        self.config = SimpleNamespace(prefix_length=0)
        self.add_image_special_tokens = False
        self.state_proj = torch.nn.Linear(2, 4, bias=False)
        self.calls = 0

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        self.calls += 1
        image = images[0] * 2
        language = lang_tokens[:, :, None].expand(-1, -1, 4).float() * 3
        measured = self.state_proj(state)
        if measured.ndim == 2:
            measured = measured[:, None, :]
        prefix = torch.cat((image, language, measured), dim=1)
        pad = torch.cat(
            (
                img_masks[0][:, None].expand(-1, image.shape[1]),
                lang_masks,
                torch.ones(measured.shape[:2], dtype=torch.bool),
            ),
            dim=1,
        )
        attention = torch.zeros_like(pad)
        attention[:, -1] = True
        return prefix, pad, attention


@pytest.fixture
def prefix_case():
    flow = Prefix().eval()
    policy = SimpleNamespace(model=flow)
    inputs = (
        [torch.arange(8.0).reshape(1, 2, 4)],
        [torch.tensor([True])],
        torch.tensor([[2, 3, 0]]),
        torch.tensor([[True, True, False]]),
    )
    return policy, inputs


@torch.no_grad()
@pytest.mark.parametrize("state_axis", [False, True])
def test_static_prefix_keeps_current_state_exact_and_masks_isolated(prefix_case, state_axis):
    policy, inputs = prefix_case
    original = policy.model.embed_prefix
    states = [torch.tensor([[1.0, 2.0]]), torch.tensor([[4.0, -3.0]])]
    if state_axis:
        states = [state[:, None, :] for state in states]
    expected = [original(*inputs, state=state) for state in states]
    with cached_static_prefix(policy) as cache:
        cache.frame, cache.task = 0.0, "Draw 一"
        first = policy.model.embed_prefix(*inputs, state=states[0])
        for actual, reference in zip(first, expected[0], strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        # Fresh-call outputs cannot mutate the retained static prefix or masks.
        first[0].fill_(99)
        first[1].fill_(False)
        first[2].fill_(False)
        actual = policy.model.embed_prefix(*inputs, state=states[1])
        for value, reference in zip(actual, expected[1], strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert (cache.hits, cache.misses) == (1, 1)
        # Cache-hit outputs are isolated too, including the state barrier masks.
        actual[0].zero_()
        actual[1].zero_()
        actual[2].zero_()
        again = policy.model.embed_prefix(*inputs, state=states[1])
        for value, reference in zip(again, expected[1], strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert policy.model.embed_prefix == original


@torch.no_grad()
def test_static_prefix_invalidates_for_acquisition_task_signature_and_reset(prefix_case):
    policy, inputs = prefix_case
    state = torch.zeros(1, 2)
    with cached_static_prefix(policy) as cache:
        cache.frame, cache.task = 0.0, "Draw 一"
        policy.model.embed_prefix(*inputs, state=state)
        cache.frame = 0.1
        policy.model.embed_prefix(*inputs, state=state)
        cache.task = "Draw 二"
        changed = (*inputs[:2], inputs[2] + 1, inputs[3])
        policy.model.embed_prefix(*changed, state=state)
        trimmed = (*inputs[:2], changed[2][:, :2], changed[3][:, :2])
        policy.model.embed_prefix(*trimmed, state=state)
        policy.model.add_image_special_tokens = True
        policy.model.embed_prefix(*trimmed, state=state)
        assert cache.misses == 5 and cache.hits == 0
        cache.clear()
        assert cache.static is None and cache.pad_mask is None and cache.att_mask is None
        assert cache.frame is None and cache.task is None
        assert cache.hits == cache.misses == 0
        cache.frame, cache.task = 0.0, "Draw 一"
        policy.model.embed_prefix(*inputs, state=state)
        assert cache.misses == 1


@pytest.mark.parametrize("attribute", ["prefix_length", "config"])
@torch.no_grad()
def test_static_prefix_rejects_padding_before_encoding(prefix_case, attribute):
    policy, inputs = prefix_case
    if attribute == "prefix_length":
        policy.model.prefix_length = 128
    else:
        policy.model.config.prefix_length = 128
    with cached_static_prefix(policy) as cache:
        cache.frame, cache.task = 0.0, "Draw 一"
        with pytest.raises(ValueError, match="prefix_length=0"):
            policy.model.embed_prefix(*inputs, state=torch.zeros(1, 2))
    assert policy.model.calls == 0


@torch.no_grad()
@pytest.mark.parametrize("state", [None, torch.zeros(2), torch.zeros(1, 2, 2)])
def test_static_prefix_rejects_missing_or_multiple_state_tokens(prefix_case, state):
    policy, inputs = prefix_case
    with cached_static_prefix(policy) as cache:
        cache.frame, cache.task = 0.0, "Draw 一"
        with pytest.raises(ValueError, match="one current state token"):
            policy.model.embed_prefix(*inputs, state=state)
    assert policy.model.calls == 0


def test_static_prefix_rejects_gradients_and_training_and_restores_on_error(prefix_case):
    policy, inputs = prefix_case
    original = policy.model.embed_prefix
    with (
        pytest.raises(RuntimeError, match="frozen evaluation"),
        cached_static_prefix(policy) as cache,
    ):
        cache.frame, cache.task = 0.0, "Draw 一"
        policy.model.embed_prefix(*inputs, state=torch.zeros(1, 2))
    assert policy.model.embed_prefix == original
    with torch.no_grad(), cached_static_prefix(policy) as cache:
        cache.frame, cache.task = 0.0, "Draw 一"
        policy.model.train()
        with pytest.raises(RuntimeError, match="frozen evaluation"):
            policy.model.embed_prefix(*inputs, state=torch.zeros(1, 2))


@torch.no_grad()
def test_static_prefix_requires_explicit_frame_and_task(prefix_case):
    policy, inputs = prefix_case
    with cached_static_prefix(policy) as cache:
        with pytest.raises(ValueError, match="frame and task"):
            policy.model.embed_prefix(*inputs, state=torch.zeros(1, 2))
        cache.frame = 0.0
        with pytest.raises(ValueError, match="frame and task"):
            policy.model.embed_prefix(*inputs, state=torch.zeros(1, 2))


def test_controller_static_cache_requires_optimization_before_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid cache settings must fail before model loading")

    monkeypatch.setattr(smolvla, "load_adapter", forbidden)
    with pytest.raises(ValueError, match="optimize_inference"):
        smolvla.SmolController("unused", cache_static_inputs=True)


def test_controller_static_cache_preserves_live_state_and_clears_each_episode(monkeypatch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Prefix().eval()
            self.model.state_proj = torch.nn.Linear(32, 4, bias=False)
            with torch.no_grad():
                self.model.state_proj.weight.zero_()
                self.model.state_proj.weight[0, 0] = 0.1
            self.eval()

        def merge_and_unload(self, *, safe_merge):
            assert safe_merge
            return self

        def reset(self):
            pass

        def predict_action_chunk(self, inputs, noise):
            image = inputs["observation.images.camera1"].mean(dim=(1, 2, 3))[:, None, None]
            prefix, _, _ = self.model.embed_prefix(
                [image.expand(-1, 2, 4)],
                [torch.ones(1, dtype=torch.bool)],
                inputs["observation.language.tokens"],
                inputs["observation.language.attention_mask"],
                state=inputs["observation.state"],
            )
            return torch.cat((prefix[:, -1], torch.zeros(1, 2)), dim=1)[:, None, :]

    def tokenize(tasks, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}

    report = {
        "objective": "action_regression",
        "chunk_size": 1,
        "rank": 8,
        "adapter_sha256": "test",
        "dataset": {
            "sensors": SensorConfig(history=1).to_dict(),
            "image_preprocessing": {"size": 4},
            "episodes": [{"camera": {"every_control_steps": 5}}],
            "stats": {
                "state": {"mean": [0.0] * 32, "std": [1.0] * 32},
                "actions": {"mean": [0.0] * 6, "std": [1.0] * 6},
            },
        },
    }
    monkeypatch.setattr(
        smolvla, "load_adapter", lambda *args, **kwargs: (Policy(), tokenize, report)
    )
    reference = smolvla.SmolController("unused", execute_steps=1)
    cached = smolvla.SmolController(
        "unused", execute_steps=1, optimize_inference=True, cache_static_inputs=True
    )
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    try:
        for controller in (reference, cached):
            controller.observe_camera(image, 0.0, "一")
        observation = np.zeros(39, dtype=np.float32)
        np.testing.assert_array_equal(cached(observation), reference(observation))
        observation[0] = 0.2
        changed = cached(observation)
        np.testing.assert_array_equal(changed, reference(observation))
        assert changed[0] > 0
        cache = cached._vision_cache
        assert cache.hits == cache.misses == 1
        assert cache.task == "Draw 一 on paper following the supplied stroke reference."
        cached.observe_camera(image, 0.1, "二")
        cached(observation)
        assert cache.frame == 0.1 and "二" in cache.task
        assert cache.misses == 2
        cached.reset()
        assert cached.image is None and cached.camera_time is None
        assert cache.static is None and cache.frame is None and cache.task is None
        assert cache.hits == cache.misses == 0
        cached.observe_camera(image, 0.0, "一")
        cached(observation)
        assert cache.misses == 1
        original = cache.encode
        cached.close()
        assert cached.model.model.embed_prefix == original
        assert cached._vision_cache is None
    finally:
        reference.close()
        cached.close()


@pytest.mark.parametrize(
    "flag", ["optimize_inference", "trim_language_padding", "cache_static_inputs"]
)
@pytest.mark.parametrize("value", [1, "true", None])
def test_controller_inference_flags_require_booleans_before_loading(monkeypatch, flag, value):
    monkeypatch.setattr(
        smolvla, "load_adapter", lambda *a, **k: pytest.fail("Loaded invalid options")
    )
    with pytest.raises(TypeError, match="booleans"):
        smolvla.SmolController("unused", **{flag: value})
