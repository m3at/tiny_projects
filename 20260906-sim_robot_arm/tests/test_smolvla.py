"""Optional-model adapter invariants without network access or LeRobot dependencies."""

import hashlib
import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shodo import smolvla
from shodo.contracts import SensorConfig


def test_action_regression_noise_is_zero_and_does_not_consume_rng():
    # Teacher actions must not enter the model's regression input.
    left = smolvla.regression_inputs({"action": torch.zeros(2, 3, 6)})
    right = smolvla.regression_inputs({"action": torch.full((2, 3, 6), 17.0)})
    for key in left:
        torch.testing.assert_close(left[key], right[key])
    generator = torch.Generator().manual_seed(7)
    initial = generator.get_state().clone()
    noise = smolvla.inference_noise(
        {"objective": "action_regression", "chunk_size": 3}, 2, "cpu", generator=generator
    )
    assert noise.shape == (2, 3, 32) and torch.count_nonzero(noise) == 0
    torch.testing.assert_close(initial, generator.get_state())
    flow = smolvla.inference_noise({"chunk_size": 3}, 2, "cpu", generator=generator)
    assert torch.count_nonzero(flow) > 0


def test_action_regression_rejects_incompatible_sampler_before_model_load(
    saved_adapter, monkeypatch
):
    path, report = saved_adapter
    report["objective"] = "action_regression"
    (path / "training.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        smolvla,
        "_dependencies",
        lambda: pytest.fail("Sampler validation must precede model loading"),
    )
    with pytest.raises(ValueError, match="exactly one"):
        smolvla.load_adapter(path, denoise_steps=2)


def test_action_regression_controller_uses_zero_noise_and_auto_one_step(adapter):
    model, _, report = adapter
    report["objective"] = "action_regression"
    controller = smolvla.SmolController("unused", execute_steps=1, device="cpu")
    assert controller.denoise_steps == 1
    controller.observe_camera(np.zeros((2, 4, 3), np.uint8), 0.0, "一")
    controller(np.zeros(78, dtype=np.float32))
    assert torch.count_nonzero(model.calls[0][1]) == 0


def _warm_start_fixture(saved_adapter, monkeypatch):
    path, report = saved_adapter
    report.update(rank=8, alpha=16)
    report["dataset"]["chars"] = ["一"]
    config = b'{"r":8,"lora_alpha":16}'
    (path / "adapter/adapter_config.json").write_bytes(config)
    report["adapter_config_sha256"] = hashlib.sha256(config).hexdigest()
    (path / "training.json").write_text(json.dumps(report))
    manifest = deepcopy(report["dataset"])
    monkeypatch.setattr(
        "shodo.vla_data.PreparedDataset", lambda *a, **k: SimpleNamespace(manifest=manifest)
    )
    return path, report, manifest


@pytest.mark.parametrize(
    "mismatch",
    [
        "rank",
        "alpha",
        "state_contract",
        "action_contract",
        "observation_contract",
        "image_preprocessing",
        "peft_rank",
        "forgotten_char",
        "missing_rank",
    ],
)
def test_warm_start_contract_rejected_before_model_load(saved_adapter, monkeypatch, mismatch):
    path, report, manifest = _warm_start_fixture(saved_adapter, monkeypatch)
    kwargs = {}
    if mismatch in ("rank", "alpha"):
        kwargs[mismatch] = 32
    elif mismatch == "forgotten_char":
        manifest["chars"] = ["二"]
    elif mismatch == "missing_rank":
        del report["rank"]
        (path / "training.json").write_text(json.dumps(report))
    elif mismatch == "peft_rank":
        config = b'{"r":32,"lora_alpha":16}'
        (path / "adapter/adapter_config.json").write_bytes(config)
        report["adapter_config_sha256"] = hashlib.sha256(config).hexdigest()
        (path / "training.json").write_text(json.dumps(report))
    else:
        manifest[mismatch] = {"different": True}
    monkeypatch.setattr(
        smolvla, "load_adapter", lambda *a, **k: pytest.fail("Loaded incompatible parent")
    )
    with pytest.raises(ValueError, match="Warm-start"):
        smolvla.train_lora("unused", path / "new", warm_start=path, **kwargs)
    assert not (path / "new").exists()


@pytest.mark.parametrize("trainable,unsafe", [(False, False), (True, False), (True, True)])
def test_adapter_trainability_is_explicit_and_base_parameters_remain_frozen(
    saved_adapter, monkeypatch, trainable, unsafe
):
    path, report = saved_adapter
    (path / "training.json").write_text(json.dumps(report))
    model = torch.nn.Module()
    model.register_parameter("base", torch.nn.Parameter(torch.ones(1), requires_grad=unsafe))
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1), requires_grad=trainable))
    calls = []

    def load(policy, directory, *, is_trainable):
        calls.append(is_trainable)
        return model

    monkeypatch.setattr(smolvla, "_dependencies", lambda: None)
    monkeypatch.setattr(smolvla, "load_base", lambda **k: (object(), Tokenizer()))
    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load))
    )
    if unsafe:
        with pytest.raises(RuntimeError, match="exclusively LoRA"):
            smolvla.load_adapter(path, is_trainable=trainable)
    else:
        result, _, _ = smolvla.load_adapter(path, is_trainable=trainable)
        assert not result.base.requires_grad
        assert result.lora_A.requires_grad is trainable
    assert calls == [trainable]


@pytest.mark.parametrize("objective", ["flow_matching", "action_regression"])
def test_warm_start_accepts_new_statistics_and_updates_both_chunk_configs(
    saved_adapter, monkeypatch, objective
):
    path, report, manifest = _warm_start_fixture(saved_adapter, monkeypatch)
    manifest["stats"]["state"]["mean"][0] = 1.5

    class Data(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"state": torch.zeros(32)}

    data = Data()
    data.manifest = manifest
    monkeypatch.setattr("shodo.vla_data.PreparedDataset", lambda *a, **k: data)
    policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=4),
        model=SimpleNamespace(config=SimpleNamespace(chunk_size=4)),
        reset=lambda: None,
    )
    model = torch.nn.Module()
    model.register_parameter("base", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
    model.get_base_model = lambda: policy
    calls = []

    def load(*a, **kwargs):
        calls.append(kwargs)
        return model, Tokenizer(), report

    def stop(parameters, **kwargs):
        trainable = list(parameters)
        assert len(trainable) == 1 and trainable[0] is model.lora_A
        raise RuntimeError("training boundary sentinel")

    monkeypatch.setattr(smolvla, "load_adapter", load)
    monkeypatch.setattr(torch.optim, "AdamW", stop)
    with pytest.raises(RuntimeError, match="training boundary sentinel"):
        smolvla.train_lora(
            "unused", path / "new", warm_start=path, chunk_size=1, device="cpu", objective=objective
        )
    assert calls == [{"device": "cpu", "is_trainable": True}]
    assert policy.config.chunk_size == policy.model.config.chunk_size == 1
    assert policy.config.n_action_steps == policy.model.config.n_action_steps == 1
    assert (
        policy.config.num_steps
        == policy.model.config.num_steps
        == (1 if objective == "action_regression" else 10)
    )
    assert not model.base.requires_grad


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, tasks, **kwargs):
        self.calls.append((tasks, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2, 0]] * len(tasks)),
            "attention_mask": torch.tensor([[1, 1, 0]] * len(tasks)),
        }


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, noise):
        self.calls.append(({key: value.clone() for key, value in batch.items()}, noise.clone()))
        return torch.tensor([[[0.0] * 6, [1.0] * 6, [10.0] * 6, [-10.0] * 6]])


@pytest.fixture
def adapter(monkeypatch):
    policy, tokenizer = Policy(), Tokenizer()
    report = {
        "chunk_size": 4,
        "rank": 8,
        "adapter_sha256": "example-checksum",
        "dataset": {
            "sensors": SensorConfig(history=2).to_dict(),
            "image_preprocessing": {"size": 4},
            "episodes": [{"camera": {"every_control_steps": 2}}],
            "chars": ["一"],
            "stats": {
                "state": {"mean": [2.0] * 32, "std": [2.0] * 32},
                "actions": {"mean": [0.25] * 6, "std": [0.5] * 6},
            },
        },
    }
    monkeypatch.setattr(
        smolvla, "load_adapter", lambda *args, **kwargs: (policy, tokenizer, report)
    )
    return policy, tokenizer, report


def test_shared_model_batch_maps_tokens_masks_and_optional_targets():
    tokenizer = Tokenizer()
    batch = {
        "state": torch.zeros(2, 32),
        "image": torch.zeros(2, 3, 4, 4),
        "task": ["Draw 一", "Draw 二"],
        "actions": torch.ones(2, 4, 6),
        "action_is_pad": torch.tensor([[False, False, True, True]] * 2),
        "privileged_secret": torch.ones(2, 4),
    }
    result = smolvla.model_batch(batch, tokenizer, "cpu")
    assert set(result) == {
        "observation.state",
        "observation.images.camera1",
        "observation.language.tokens",
        "observation.language.attention_mask",
        "action",
        "action_is_pad",
    }
    assert result["observation.language.attention_mask"].dtype == torch.bool
    assert tokenizer.calls[0][1] == {
        "padding": "max_length",
        "truncation": True,
        "max_length": 48,
        "return_tensors": "pt",
    }
    inference = smolvla.model_batch(
        {key: batch[key] for key in ("state", "image", "task")}, tokenizer, "cpu"
    )
    assert "action" not in inference
    torch.testing.assert_close(result["action"], batch["actions"])


def test_controller_normalization_queue_clipping_camera_and_reset(adapter):
    model, tokenizer, _ = adapter
    controller = smolvla.SmolController("unused", device="cpu", execute_steps=4)
    observation = np.r_[np.full(39, 1e8), np.arange(39)].astype(np.float32)
    with pytest.raises(ValueError, match="camera observation"):
        controller(observation)
    frame = np.full((2, 4, 3), 128, np.uint8)
    controller.observe_camera(frame, 0.0, "一")
    np.testing.assert_allclose(controller(observation), 0.25)
    captured, noise = model.calls[0]
    expected = (np.r_[np.arange(25), np.arange(32, 39)] - 2) / 2
    np.testing.assert_array_equal(captured["observation.state"][0], expected)
    assert captured["observation.images.camera1"][0, 0, 1, 0] == np.float32(128 / 255)
    assert captured["observation.images.camera1"][0, 0, 0, 0] == 0
    assert tokenizer.calls[0][0] == ["Draw 一 on paper following the supplied stroke reference."]
    np.testing.assert_allclose(controller(observation), 0.75)
    controller.observe_camera(np.full_like(frame, 64), 0.04, "一")
    np.testing.assert_allclose(controller(observation), 1)
    np.testing.assert_allclose(controller(observation), -1)
    assert len(model.calls) == 1
    controller(observation)
    assert len(model.calls) == 2
    assert model.calls[1][0]["observation.images.camera1"][0, 0, 1, 0] == np.float32(64 / 255)
    controller.reset()
    assert not controller.queue and controller.image is None and not controller.inference_seconds
    controller.observe_camera(frame, 0, "二")
    controller(observation)
    torch.testing.assert_close(model.calls[-1][1], noise)
    assert model.resets == 2


@pytest.mark.parametrize("timestamp", [0, float("nan")])
def test_controller_rejects_invalid_or_repeated_camera_time(adapter, timestamp):
    controller = smolvla.SmolController("unused", device="cpu")
    image = np.zeros((2, 2, 3), np.uint8)
    controller.observe_camera(image, 0, "一")
    with pytest.raises(ValueError, match="timestamps"):
        controller.observe_camera(image, timestamp, "一")


def test_controller_rejects_nonfinite_model_actions(adapter):
    model, _, _ = adapter
    model.predict_action_chunk = lambda *args, **kwargs: torch.full((1, 4, 6), float("nan"))
    controller = smolvla.SmolController("unused")
    controller.observe_camera(np.zeros((2, 2, 3), np.uint8), 0, "一")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        controller(np.zeros(78))


def test_controller_rejects_mixed_camera_cadence(adapter):
    _, _, report = adapter
    second = deepcopy(report["dataset"]["episodes"][0])
    second["camera"]["every_control_steps"] = 3
    report["dataset"]["episodes"].append(second)
    with pytest.raises(ValueError, match="cadence"):
        smolvla.SmolController("unused")


def test_training_refuses_existing_output_before_dataset_access(tmp_path):
    with pytest.raises(FileExistsError):
        smolvla.train_lora(tmp_path / "missing", tmp_path)


def test_evaluation_rejects_training_characters_and_empty_selection(adapter):
    with pytest.raises(ValueError, match="overlaps training"):
        smolvla.evaluate_adapter("unused", chars="一")
    with pytest.raises(ValueError, match="needs characters"):
        smolvla.evaluate_adapter("unused", chars="")


def test_rollout_delivers_causal_raw_frames_before_actions_and_closes(monkeypatch):
    from shodo import learning, runtime

    events = []

    class StopProbe(Exception):
        pass

    class Environment:
        def __init__(self, **kwargs):
            self.unwrapped = self
            self.index = 0
            self.data = SimpleNamespace(time=0.0)
            self.renderer = SimpleNamespace(camera_frame=self.camera_frame)

        def reset(self, **kwargs):
            return np.zeros(78), {}

        def camera_frame(self, env):
            events.append(("capture", self.index, self.data.time))
            return np.full((2, 2, 3), self.index, np.uint8)

        def close(self):
            events.append(("close",))

    class CameraPolicy:
        sensor_config = SensorConfig(history=2)
        camera_every = 2

        def reset(self):
            events.append(("reset",))

        def observe_camera(self, image, timestamp, char):
            events.append(("observe", int(image[0, 0, 0]), timestamp, char))

        def __call__(self, observation):
            return np.zeros(6)

    def execute(env, observation, policy, **kwargs):
        events.append(("act", env.index, env.data.time))
        if env.index == 3:
            raise StopProbe
        env.index += 1
        env.data.time = env.index * 0.02
        return SimpleNamespace(
            next_observation=observation, terminated=False, truncated=False, reward=0
        )

    monkeypatch.setattr(runtime, "SensorEnv", Environment)
    monkeypatch.setattr(runtime, "execute_step", execute)
    with pytest.raises(StopProbe):
        learning.rollout("一", CameraPolicy())
    assert events == [
        ("reset",),
        ("capture", 0, 0.0),
        ("observe", 0, 0.0, "一"),
        ("act", 0, 0.0),
        ("act", 1, 0.02),
        ("capture", 2, 0.04),
        ("observe", 2, 0.04, "一"),
        ("act", 2, 0.04),
        ("act", 3, 0.06),
        ("close",),
    ]


@pytest.fixture
def saved_adapter(tmp_path):
    from shodo.config import SimConfig
    from shodo.contracts import ActionContract, sensor_contract
    from shodo.vla_data import SCHEMA, STATE_CONTRACT, _image_contract

    directory = tmp_path / "adapter"
    directory.mkdir()
    weights = b"fake weights never loaded"
    config = b'{"r":8}'
    (directory / "adapter_model.safetensors").write_bytes(weights)
    (directory / "adapter_config.json").write_bytes(config)
    sensors = SensorConfig()
    report = {
        "schema_version": smolvla.FORMAT_VERSION,
        "base_model": smolvla.BASE_ID,
        "base_revision": smolvla.BASE_REVISION,
        "vlm_model": smolvla.VLM_ID,
        "vlm_revision": smolvla.VLM_REVISION,
        "chunk_size": 4,
        "adapter_sha256": hashlib.sha256(weights).hexdigest(),
        "adapter_config_sha256": hashlib.sha256(config).hexdigest(),
        "dataset": {
            "schema": SCHEMA,
            "version": 1,
            "stats_std_floor": 1e-6,
            "episodes": [{"sha256": "training-episode", "char": "一"}],
            "state_contract": STATE_CONTRACT,
            "sensors": sensors.to_dict(),
            "config": SimConfig().to_dict(),
            "observation_contract": sensor_contract(sensors),
            "action_contract": ActionContract().to_dict(),
            "image_preprocessing": _image_contract(4),
            "stats": {
                key: {"mean": [0] * width, "std": [1] * width}
                for key, width in (("state", 32), ("actions", 6))
            },
        },
    }
    return tmp_path, report


@pytest.mark.parametrize(
    "mutation",
    [
        "base_id",
        "vlm_id",
        "state_contract",
        "action_contract",
        "zero_std",
        "nonfinite_stats",
        "image_contract",
        "weights",
        "adapter_config",
        "chunk_size",
    ],
)
def test_adapter_contract_rejection_precedes_optional_dependencies(
    saved_adapter, monkeypatch, mutation
):
    path, report = saved_adapter
    if mutation == "base_id":
        report["base_model"] = "other/model"
    elif mutation == "vlm_id":
        report["vlm_model"] = "other/model"
    elif mutation == "state_contract":
        report["dataset"]["state_contract"] = {"privileged_inputs": True}
    elif mutation == "action_contract":
        report["dataset"]["action_contract"]["frame"] = "tool"
    elif mutation == "zero_std":
        report["dataset"]["stats"]["state"]["std"][0] = 0
    elif mutation == "nonfinite_stats":
        report["dataset"]["stats"]["actions"]["mean"][0] = float("nan")
    elif mutation == "image_contract":
        report["dataset"]["image_preprocessing"]["storage"] = "BGR"
    elif mutation == "weights":
        (path / "adapter/adapter_model.safetensors").write_bytes(b"changed")
    elif mutation == "adapter_config":
        (path / "adapter/adapter_config.json").write_bytes(b'{"r":16}')
    else:
        report["chunk_size"] = 0
    (path / "training.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        smolvla,
        "_dependencies",
        lambda: pytest.fail("Must validate before importing optional models"),
    )
    with pytest.raises(ValueError):
        smolvla.load_adapter(path)


def test_adapter_execution_horizon_checked_before_optional_dependencies(saved_adapter, monkeypatch):
    path, report = saved_adapter
    (path / "training.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        smolvla, "_dependencies", lambda: pytest.fail("Must validate horizon first")
    )
    with pytest.raises(ValueError, match="execute_steps"):
        smolvla.load_adapter(path, execute_steps=5)


def test_counterfactual_adapter_rejects_multi_step_chunk_before_load(saved_adapter, monkeypatch):
    from shodo.vla_data import NORMALIZATION_CONTRACT, _supervision_contract

    path, report = saved_adapter
    report["dataset"].update(
        version=2,
        normalization=NORMALIZATION_CONTRACT,
        supervision=_supervision_contract("expert"),
    )
    (path / "training.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        smolvla, "_dependencies", lambda: pytest.fail("Validate expert horizon before load")
    )
    with pytest.raises(ValueError, match="Counterfactual.*chunk_size=1"):
        smolvla.load_adapter(path)


def test_optimized_controller_merge_cache_equivalence_lifecycle_and_reset(adapter):
    model, tokenizer, _ = adapter

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def embed_image(self, image):
            self.calls += 1
            return image.mean()

    vision = Vision()
    model.model = torch.nn.Module()
    model.model.vlm_with_expert = vision
    merges = []

    def merge_and_unload(*, safe_merge):
        merges.append(safe_merge)
        return model

    def predict(batch, noise):
        return torch.zeros(1, 4, 6) + vision.embed_image(batch["observation.images.camera1"])

    model.merge_and_unload = merge_and_unload
    model.predict_action_chunk = predict
    observation = np.zeros(78, np.float32)
    frame = np.full((4, 4, 3), 128, np.uint8)
    plain = smolvla.SmolController("unused", execute_steps=1)
    plain.observe_camera(frame, 0, "一")
    expected = plain(observation)
    plain.close()
    original = vision.embed_image
    optimized = smolvla.SmolController("unused", execute_steps=1, optimize_inference=True)
    cache = optimized._vision_cache
    assert merges == [True] and not model.training
    optimized.observe_camera(frame, 0, "一")
    np.testing.assert_array_equal(optimized(observation), expected)
    np.testing.assert_array_equal(optimized(observation), expected)
    assert cache.misses == 1 and cache.hits == 1
    assert len(tokenizer.calls) == 2  # plain plus first optimized prompt
    optimized.observe_camera(frame, 0.02, "一")
    optimized(observation)
    assert cache.misses == 2
    optimized.reset()
    assert cache.frame is None and cache.value is None and cache.hits == cache.misses == 0
    optimized.observe_camera(frame, 0, "二")
    optimized(observation)
    assert cache.misses == 1 and tokenizer.calls[-1][0][0].startswith("Draw 二")
    optimized.close()
    assert vision.embed_image == original
    assert optimized._vision_context is None and optimized._vision_cache is None
    optimized.close()  # idempotent release


@pytest.mark.parametrize("failure", ["overlap", "rollout"])
def test_evaluation_closes_policy_on_all_exit_paths(tmp_path, monkeypatch, failure):
    from shodo import runtime
    from shodo.config import SimConfig
    from shodo.contracts import ActionContract

    policy = SimpleNamespace(
        report={
            "dataset": {
                "chars": ["一"],
                "config": SimConfig().to_dict(),
                "action_contract": ActionContract().to_dict(),
            }
        },
        checkpoint_provenance={"test": True},
        denoise_steps=10,
        inference_seconds=[0.01, 0.02],
        decision_seconds=[0.011, 0.021],
        planning_seconds=[0.011, 0.021],
    )
    closed = []
    policy.close = lambda: closed.append(True)
    monkeypatch.setattr(smolvla, "SmolController", lambda *args, **kwargs: policy)

    def rollout(*args, **kwargs):
        if failure == "rollout":
            raise RuntimeError("rollout sentinel")
        return {"ink_rmse_mm": None, "truncated": False}

    monkeypatch.setattr(runtime, "sensor_rollout", rollout)
    report_path = tmp_path if failure == "write" else tmp_path / "evaluation.json"
    chars = "一" if failure == "overlap" else "永"
    if failure == "none":
        report = smolvla.evaluate_adapter(tmp_path, chars=chars, report_path=report_path)
        assert report["rows"][0]["split"] == "held-out"
        assert report["rows"][0]["inference_latency"]["calls"] == 2
        assert report["rows"][0]["actor_latency"]["control_deadline_misses"] == 1
        assert report["rows"][0]["actor_latency"]["planning_horizon_misses"] == 0
        policy.inference_seconds.clear()
        assert report["rows"][0]["inference_seconds"] == [0.01, 0.02]
    else:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            smolvla.evaluate_adapter(tmp_path, chars=chars, report_path=report_path)
    assert closed == [True]


def test_evaluation_persists_latency_and_checkpoint(adapter, monkeypatch, tmp_path):
    from shodo.config import SimConfig
    from shodo.contracts import ActionContract

    _, _, report = adapter
    report["dataset"]["config"] = SimConfig().to_dict()
    report["dataset"]["action_contract"] = ActionContract().to_dict()

    def rollout(char, policy, **kwargs):
        policy.inference_seconds = [0.1, 0.2]
        return {"char": char, "ink_rmse_mm": None, "truncated": True}

    monkeypatch.setattr("shodo.runtime.sensor_rollout", rollout)
    result = smolvla.evaluate_adapter(tmp_path, chars="永", device="cpu")
    latency = result["rows"][0]["inference_latency"]
    assert latency["calls"] == 2
    assert latency["median_seconds"] == pytest.approx(0.15)
    assert latency["execution_horizon_seconds"] == 0.08
    assert result["checkpoint"]["sha256"] == report["adapter_sha256"]
    assert json.loads((tmp_path / "evaluation.json").read_text()) == result
