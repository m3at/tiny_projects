"""Optional, pinned SmolVLA LoRA development workflow over native shodo episodes.

Run in integrations/smolvla's locked environment. No Hub uploads, hardware driver,
or claim that synchronous foundation-model inference meets the 50 Hz deadline.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from filelock import FileLock

from shodo.device import resolve_device

BASE_ID = "lerobot/smolvla_base"
BASE_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
FORMAT_VERSION = 1


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _dependencies():
    try:
        import lerobot
        import peft
    except ImportError as error:
        raise RuntimeError(
            "Run make smolvla-setup and use the integrations/smolvla environment"
        ) from error
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("This adapter requires the locked LeRobot 0.6.1 integration")
    return lerobot, peft


def load_base(*, device="cpu", image_size=256, chunk_size=16, denoise_steps=10):
    """Strictly load the complete pretrained policy, without downloading redundant VLM weights."""
    for value, name in (
        (image_size, "image_size"),
        (chunk_size, "chunk_size"),
        (denoise_steps, "denoise_steps"),
    ):
        _positive_integer(value, name)
    if image_size % 64:
        raise ValueError("SmolVLA image_size must be a positive multiple of 64")
    _dependencies()
    from huggingface_hub import snapshot_download
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    base = snapshot_download(
        BASE_ID, revision=BASE_REVISION, allow_patterns=["config.json", "model.safetensors"]
    )
    vlm = snapshot_download(
        VLM_ID,
        revision=VLM_REVISION,
        allow_patterns=["*.json", "*.txt"],
        ignore_patterns=["onnx/*"],
    )
    config = SmolVLAConfig.from_pretrained(
        base, cli_overrides=[f"--device={resolve_device(device)}"]
    )
    config.device = str(resolve_device(device))
    config.vlm_model_name = vlm
    config.load_vlm_weights = False
    config.chunk_size = chunk_size
    config.n_action_steps = 1
    config.num_steps = denoise_steps
    config.resize_imgs_with_padding = (image_size, image_size)
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (32,)),
        "observation.images.camera1": PolicyFeature(
            FeatureType.VISUAL, (3, image_size, image_size)
        ),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (6,))}
    config.push_to_hub = False
    policy = SmolVLAPolicy.from_pretrained(base, config=config, strict=True)
    # Full policy weights have now loaded strictly; PEFT's from-scratch warning is inapplicable.
    config.load_vlm_weights = True
    config.pretrained_path = BASE_ID
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    return policy, tokenizer


def model_batch(batch, tokenizer, device):
    """A single shared preprocessing path for training and inference."""
    tokens = tokenizer(
        batch["task"], padding="max_length", truncation=True, max_length=48, return_tensors="pt"
    )
    output = {
        "observation.state": batch["state"].to(device),
        "observation.images.camera1": batch["image"].to(device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
    }
    if "actions" in batch:
        output["action"] = batch["actions"].to(device)
        output["action_is_pad"] = batch["action_is_pad"].to(device)
    return output


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def adapter_objective(report):
    objective = report.get("objective", "flow_matching")
    if objective not in ("flow_matching", "action_regression"):
        raise ValueError("Unsupported SmolVLA training objective")
    return objective


def resolve_denoise_steps(report, requested=None):
    objective = adapter_objective(report)
    steps = requested if requested is not None else (1 if objective == "action_regression" else 10)
    _positive_integer(steps, "denoise_steps")
    if objective == "action_regression" and steps != 1:
        raise ValueError("Action regression requires exactly one denoising step")
    return steps


def inference_noise(report, batch_size, device, generator=None):
    shape = (batch_size, report["chunk_size"], 32)
    if adapter_objective(report) == "action_regression":
        return torch.zeros(shape, device=device)
    return torch.randn(shape, generator=generator).to(device)


def regression_inputs(inputs):
    """Pinned flow forward at t=1: x_t=0, u_t=-action; targets enter only the loss."""
    batch, chunk = inputs["action"].shape[:2]
    return {
        "noise": torch.zeros((batch, chunk, 32), device=inputs["action"].device),
        "time": torch.ones(batch, device=inputs["action"].device),
    }


def training_episode_hashes(report, *, _ancestors=frozenset()):
    """All fine-tuning episode identities, including verified warm-start ancestors.

    New artifacts carry a self-contained union. Legacy artifacts require their
    recorded parent manifests with matching hashes; missing ancestry cannot
    establish an independent evaluation split.
    """
    try:
        own = {entry["sha256"] for entry in report["dataset"]["episodes"]}
    except (KeyError, TypeError) as exc:
        raise ValueError("Training episode identities are missing") from exc
    if not own or any(not isinstance(value, str) or not value for value in own):
        raise ValueError("Training episode identities must be nonempty strings")
    inherited = report.get("training_episode_sha256")
    if inherited is not None:
        if (
            not isinstance(inherited, list)
            or any(not isinstance(value, str) or not value for value in inherited)
            or not own.issubset(inherited)
        ):
            raise ValueError("Invalid inherited training episode identities")
        return set(inherited)
    parent = report.get("warm_start")
    if parent is None:
        return own
    try:
        path = Path(parent["directory"]) / "training.json"
        expected = parent["training_manifest_sha256"]
        if not isinstance(expected, str) or expected in _ancestors:
            raise ValueError("Invalid or cyclic warm-start ancestry")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Warm-start ancestor manifest checksum mismatch")
        ancestor = json.loads(source)
        if ancestor.get("adapter_sha256") != parent["adapter_sha256"]:
            raise ValueError("Warm-start ancestor adapter identity mismatch")
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Cannot establish independent episodes: warm-start ancestry unavailable"
        ) from exc
    return own | training_episode_hashes(ancestor, _ancestors=_ancestors | {expected})


def train_lora(
    dataset,
    output,
    *,
    steps=100,
    batch_size=1,
    rank=8,
    alpha=16,
    learning_rate=1e-3,
    chunk_size=16,
    seed=7,
    device="auto",
    warm_start=None,
    objective="flow_matching",
):
    from shodo.artifacts import provenance
    from shodo.vla_data import PreparedDataset

    adapter_objective({"objective": objective})

    if any(type(v) is not int or v < 1 for v in (steps, batch_size, rank, alpha, chunk_size)):
        raise ValueError("Steps, batch size, rank, alpha and chunk size must be positive integers")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or type(seed) is not int
        or not 0 <= seed < 2**32
    ):
        raise ValueError("Use a positive finite learning rate and nonnegative integer seed")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Adapter output exists; use a new directory")
    data = PreparedDataset(dataset, chunk_size=chunk_size)
    episode_hashes = training_episode_hashes({"dataset": data.manifest})
    parent = None
    if warm_start is not None:
        parent = _read_adapter_report(warm_start)
        episode_hashes |= training_episode_hashes(parent)
        if parent.get("rank") != rank or parent.get("alpha") != alpha:
            raise ValueError("Warm-start rank and alpha must match the parent adapter")
        previous_chars = parent["dataset"].get("chars")
        current_chars = data.manifest.get("chars")
        if (
            not isinstance(previous_chars, list)
            or not previous_chars
            or not all(isinstance(char, str) and len(char) == 1 for char in previous_chars)
            or not isinstance(current_chars, list)
            or not all(isinstance(char, str) and len(char) == 1 for char in current_chars)
            or not set(previous_chars).issubset(current_chars)
        ):
            raise ValueError("Warm-start dataset must retain every parent training character")
        for key in (
            "state_contract",
            "action_contract",
            "observation_contract",
            "image_preprocessing",
        ):
            if parent["dataset"].get(key) != data.manifest.get(key):
                raise ValueError(f"Warm-start dataset {key} differs from the parent adapter")
        parent_config = json.loads((Path(warm_start) / "adapter/adapter_config.json").read_text())
        if (
            not isinstance(parent_config, dict)
            or parent_config.get("r") != rank
            or parent_config.get("lora_alpha") != alpha
        ):
            raise ValueError("Warm-start PEFT rank and alpha disagree with training metadata")
    resolved = resolve_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if parent is None:
        policy, tokenizer = load_base(
            device=str(resolved),
            image_size=data.manifest["image_preprocessing"]["size"],
            chunk_size=chunk_size,
            denoise_steps=resolve_denoise_steps({"objective": objective}),
        )
        model = policy.wrap_with_peft(
            peft_cli_overrides={"r": rank, "lora_alpha": alpha, "lora_dropout": 0.0}
        )
    else:
        model, tokenizer, _ = load_adapter(warm_start, device=str(resolved), is_trainable=True)
        policy = model.get_base_model()
        for config in (policy.config, policy.model.config):
            config.chunk_size = chunk_size
            config.n_action_steps = 1
            config.num_steps = resolve_denoise_steps({"objective": objective})
        policy.reset()
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected exclusively LoRA parameters to be trainable")
    initial = {name: value.detach().cpu().clone() for name, value in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=1e-10)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    fixed = model_batch(next(iter(loader)), tokenizer, resolved)
    noise = inference_noise(
        {"objective": objective, "chunk_size": chunk_size},
        fixed["action"].shape[0],
        resolved,
        generator=torch.Generator().manual_seed(41),
    )
    flow_time = torch.full(
        (fixed["action"].shape[0],),
        1.0 if objective == "action_regression" else 0.5,
        device=resolved,
    )
    model.eval()
    with torch.no_grad():
        before = float(model(fixed, noise=noise, time=flow_time)[0])
    print(
        f"SmolVLA LoRA ({objective}): {sum(v.numel() for v in trainable.values()):,} trainable / {sum(v.numel() for v in model.parameters()):,} parameters; device={resolved}",
        flush=True,
    )
    losses, durations = [], []
    iterator = iter(loader)
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = model_batch(batch, tokenizer, resolved)
        model.train()
        synchronize(resolved)
        start = time.perf_counter()
        loss, _ = model(
            inputs, **(regression_inputs(inputs) if objective == "action_regression" else {})
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite LoRA training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        synchronize(resolved)
        durations.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % max(10, steps // 20) == 0 or step + 1 == steps:
            print(
                f"  step {step + 1}/{steps}: loss={losses[-1]:.5f}, update={durations[-1]:.3f}s",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        after = float(model(fixed, noise=noise, time=flow_time)[0])
        probe = model.predict_action_chunk(fixed, noise=noise).cpu().numpy()
    changed = sum(
        not torch.equal(initial[name], value.detach().cpu()) for name, value in trainable.items()
    )
    if not changed or not np.isfinite(probe).all():
        raise RuntimeError("LoRA parameters did not update or inference is nonfinite")
    report = {
        "schema_version": FORMAT_VERSION,
        "objective": objective,
        "objective_semantics": (
            "deterministic action regression via pinned flow endpoint t=1, noise=0; inference one Euler step from zero"
            if objective == "action_regression"
            else "standard conditional flow matching with sampled noise and time"
        ),
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "vlm_model": VLM_ID,
        "vlm_revision": VLM_REVISION,
        "steps": steps,
        "batch_size": batch_size,
        "rank": rank,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "chunk_size": chunk_size,
        "seed": seed,
        "device": {"requested": device, "resolved": str(resolved)},
        "trainable_parameters": sum(v.numel() for v in trainable.values()),
        "total_parameters": sum(v.numel() for v in model.parameters()),
        "changed_adapter_tensors": changed,
        "losses": losses,
        "update_seconds": durations,
        "fixed_batch_loss_before": before,
        "fixed_batch_loss_after": after,
        "fixed_batch_scope": "training minibatch, not held-out validation",
        "dataset": data.manifest,
        "training_episode_sha256": sorted(episode_hashes),
        "dataset_manifest_sha256": hashlib.sha256(
            (Path(dataset) / "manifest.json").read_bytes()
        ).hexdigest(),
        "provenance": provenance(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("lerobot", "peft", "transformers", "torch", "numpy")
        },
        "resume": "adapter artifact only; optimizer/RNG resume not implemented",
        "warm_start": None
        if parent is None
        else {
            "directory": str(Path(warm_start).resolve()),
            "adapter_sha256": parent["adapter_sha256"],
            "adapter_config_sha256": parent["adapter_config_sha256"],
            "training_manifest_sha256": hashlib.sha256(
                (Path(warm_start) / "training.json").read_bytes()
            ).hexdigest(),
            "parent_settings": {
                key: parent.get(key)
                for key in (
                    "steps",
                    "batch_size",
                    "rank",
                    "alpha",
                    "learning_rate",
                    "chunk_size",
                    "seed",
                    "dataset_manifest_sha256",
                    "warm_start",
                )
            },
            "normalization_changed": parent["dataset"]["stats"] != data.manifest["stats"],
            "parent_objective": adapter_objective(parent),
            "semantics": "adapter-weight initialization only; new optimizer and RNG; new dataset normalization; not exact training resume",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(output.parent / f".{output.name}.lock", timeout=0):
        if output.exists():
            raise FileExistsError("Adapter output exists")
        with TemporaryDirectory(prefix=".smolvla-", dir=output.parent) as temporary:
            temporary = Path(temporary)
            model.save_pretrained(temporary / "adapter", safe_serialization=True)
            report["adapter_sha256"] = hashlib.sha256(
                (temporary / "adapter" / "adapter_model.safetensors").read_bytes()
            ).hexdigest()
            report["adapter_config_sha256"] = hashlib.sha256(
                (temporary / "adapter" / "adapter_config.json").read_bytes()
            ).hexdigest()
            (temporary / "training.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            np.savez_compressed(
                temporary / "reload-probe.npz",
                noise=noise.cpu().numpy(),
                flow_time=flow_time.cpu().numpy(),
                actions=probe,
                **{key: value.cpu().numpy() for key, value in fixed.items()},
            )
            temporary.rename(output)
    print(
        f"Adapter: {output.resolve()} · fixed training probe {before:.5f} → {after:.5f}", flush=True
    )
    return report


def _read_adapter_report(directory, *, execute_steps=None):
    from shodo.vla_data import validate_manifest_contract

    directory = Path(directory)
    report = json.loads((directory / "training.json").read_text())
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != FORMAT_VERSION
        or report.get("base_model") != BASE_ID
        or report.get("vlm_model") != VLM_ID
        or report.get("base_revision") != BASE_REVISION
        or report.get("vlm_revision") != VLM_REVISION
    ):
        raise ValueError("Unsupported SmolVLA adapter/base contract")
    validate_manifest_contract(report.get("dataset"))
    adapter_objective(report)
    if type(report.get("chunk_size")) is not int or report["chunk_size"] < 1:
        raise ValueError("Adapter chunk_size must be a positive integer")
    if (
        report["dataset"].get("supervision", {}).get("source") == "expert"
        and report["chunk_size"] != 1
    ):
        raise ValueError("Counterfactual expert supervision requires chunk_size=1")
    if execute_steps is not None and (
        type(execute_steps) is not int or not 1 <= execute_steps <= report["chunk_size"]
    ):
        raise ValueError("execute_steps must fit inside the trained action chunk")
    actual = hashlib.sha256(
        (directory / "adapter" / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    if actual != report.get("adapter_sha256"):
        raise ValueError("Adapter checksum mismatch")
    config_hash = hashlib.sha256(
        (directory / "adapter" / "adapter_config.json").read_bytes()
    ).hexdigest()
    if config_hash != report.get("adapter_config_sha256"):
        raise ValueError("Adapter configuration checksum mismatch")
    return report


def load_adapter(
    directory, *, device="auto", denoise_steps=None, execute_steps=None, is_trainable=False
):
    if denoise_steps is not None:
        _positive_integer(denoise_steps, "denoise_steps")
    if type(is_trainable) is not bool:
        raise ValueError("is_trainable must be a boolean")
    directory = Path(directory)
    report = _read_adapter_report(directory, execute_steps=execute_steps)
    denoise_steps = resolve_denoise_steps(report, denoise_steps)
    _dependencies()
    from peft import PeftModel

    policy, tokenizer = load_base(
        device=device,
        image_size=report["dataset"]["image_preprocessing"]["size"],
        chunk_size=report["chunk_size"],
        denoise_steps=denoise_steps,
    )
    model = PeftModel.from_pretrained(policy, directory / "adapter", is_trainable=is_trainable)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if is_trainable and (not trainable or any("lora_" not in name for name in trainable)):
        raise RuntimeError("Expected exclusively LoRA parameters to be trainable")
    model.eval()
    return model, tokenizer, report


def verify_reload(directory, *, device="auto"):
    model, _, report = load_adapter(directory, device=device)
    resolved = next(model.parameters()).device
    with np.load(Path(directory) / "reload-probe.npz", allow_pickle=False) as archive:
        batch = {
            key: torch.from_numpy(archive[key]).to(resolved)
            for key in archive.files
            if key not in ("noise", "actions", "flow_time")
        }
        noise = torch.from_numpy(archive["noise"]).to(resolved)
        with torch.no_grad():
            actual = model.predict_action_chunk(batch, noise=noise).cpu().numpy()
        error = float(np.max(np.abs(actual - archive["actions"])))
    if not math.isfinite(error) or error > 1e-4:
        raise RuntimeError(f"Adapter reload action mismatch: {error}")
    print(f"Adapter reload verified: maximum normalized action difference {error:.3g}", flush=True)
    result = {
        "max_action_difference": error,
        "training_steps": report["steps"],
        "device": str(resolved),
    }
    (Path(directory) / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class SmolController:
    """Receding-horizon simulated controller; camera input never contains diagnostics.

    Inference blocks simulation time. Wall-clock latency is measured, not hidden
    as a claim that a hardware controller can execute this at 50 Hz.
    """

    def __init__(
        self,
        directory,
        *,
        device="auto",
        execute_steps=4,
        denoise_steps=None,
        seed=7,
        optimize_inference=False,
        trim_language_padding=False,
        cache_static_inputs=False,
    ):
        from shodo.contracts import SensorConfig

        if any(
            type(value) is not bool
            for value in (optimize_inference, trim_language_padding, cache_static_inputs)
        ):
            raise TypeError("Inference options must be booleans")
        if cache_static_inputs and not optimize_inference:
            raise ValueError("Static input caching requires optimize_inference")
        self.cache_static_inputs = cache_static_inputs
        _positive_integer(execute_steps, "execute_steps")
        if denoise_steps is not None:
            _positive_integer(denoise_steps, "denoise_steps")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.model, self.tokenizer, self.report = load_adapter(
            directory, device=device, denoise_steps=denoise_steps, execute_steps=execute_steps
        )
        self._vision_context = None
        self._vision_cache = None
        if type(execute_steps) is not int or not 1 <= execute_steps <= self.report["chunk_size"]:
            raise ValueError("execute_steps must fit inside the trained action chunk")
        self.execute_steps = execute_steps
        self.denoise_steps = resolve_denoise_steps(self.report, denoise_steps)
        self.seed = seed
        self.device = str(next(self.model.parameters()).device)
        self.requested_device = device
        self.sensor_config = SensorConfig(**self.report["dataset"]["sensors"])
        self.image_size = self.report["dataset"]["image_preprocessing"]["size"]
        cadences = {
            entry["camera"]["every_control_steps"] for entry in self.report["dataset"]["episodes"]
        }
        if len(cadences) != 1 or not all(type(c) is int and c > 0 for c in cadences):
            raise ValueError("Training cameras need one positive, fixed control cadence")
        self.camera_every = cadences.pop()
        self.stats = self.report["dataset"]["stats"]
        self.checkpoint_provenance = {
            "algorithm": "SmolVLA LoRA",
            "sha256": self.report["adapter_sha256"],
            "base_revision": BASE_REVISION,
            "rank": self.report["rank"],
        }
        if optimize_inference:
            from shodo.vla_inference import PromptCache, cached_static_prefix, cached_vision

            self.model = self.model.merge_and_unload(safe_merge=True)
            self.model.eval()
            self.tokenizer = PromptCache(self.tokenizer, trim_padding=trim_language_padding)
            cache_scope = cached_static_prefix if cache_static_inputs else cached_vision
            self._vision_context = cache_scope(self.model)
            self._vision_cache = self._vision_context.__enter__()
        elif trim_language_padding:
            from shodo.vla_inference import PromptCache

            self.tokenizer = PromptCache(self.tokenizer, trim_padding=True)
        self.reset()

    def reset(self):
        self.model.reset()
        self.queue = deque()
        self.image = None
        self.camera_time = None
        self.inference_seconds = []
        self.decision_seconds = []
        self.planning_seconds = []
        self._camera_preprocess_seconds = 0.0
        self.rng = torch.Generator().manual_seed(self.seed)
        if self._vision_cache is not None:
            self._vision_cache.clear()

    def close(self):
        if self._vision_context is not None:
            self._vision_context.__exit__(None, None, None)
            self._vision_context = None
            self._vision_cache = None

    def observe_camera(self, image, timestamp_s, char):
        from shodo.vla_data import preprocess_image

        if (
            not math.isfinite(timestamp_s)
            or timestamp_s < 0
            or (self.camera_time is not None and timestamp_s <= self.camera_time)
        ):
            raise ValueError("Camera timestamps must strictly increase")
        start = time.perf_counter()
        self.image = preprocess_image(image, self.image_size).astype(np.float32) / 255
        self.camera_time = timestamp_s
        if self._vision_cache is not None:
            self._vision_cache.frame = timestamp_s
        self.task = f"Draw {char} on paper following the supplied stroke reference."
        if self.cache_static_inputs:
            self._vision_cache.task = self.task
        self._camera_preprocess_seconds += time.perf_counter() - start

    def __call__(self, observation):
        start = time.perf_counter()
        replanning = not self.queue
        try:
            return self._act(observation)
        finally:
            elapsed = time.perf_counter() - start + self._camera_preprocess_seconds
            self._camera_preprocess_seconds = 0.0
            self.decision_seconds.append(elapsed)
            if replanning:
                self.planning_seconds.append(elapsed)

    def _act(self, observation):
        from shodo.vla_data import state_from_observation

        if self.image is None:
            raise ValueError("SmolVLA needs a raw camera observation before acting")
        if not self.queue:
            state = state_from_observation(observation, self.sensor_config.history)
            state = (
                state - np.asarray(self.stats["state"]["mean"], dtype=np.float32)
            ) / np.asarray(self.stats["state"]["std"], dtype=np.float32)
            inputs = model_batch(
                {
                    "state": torch.from_numpy(state[None]),
                    "image": torch.from_numpy(self.image[None]),
                    "task": [self.task],
                },
                self.tokenizer,
                self.device,
            )
            noise = inference_noise(self.report, 1, self.device, generator=self.rng)
            synchronize(self.device)
            start = time.perf_counter()
            with torch.no_grad():
                actions = self.model.predict_action_chunk(inputs, noise=noise)[0].cpu().numpy()
            synchronize(self.device)
            self.inference_seconds.append(time.perf_counter() - start)
            actions = actions * np.asarray(
                self.stats["actions"]["std"], dtype=np.float32
            ) + np.asarray(self.stats["actions"]["mean"], dtype=np.float32)
            if not np.isfinite(actions).all():
                raise RuntimeError("Nonfinite SmolVLA action chunk")
            self.queue.extend(np.clip(actions[: self.execute_steps], -1, 1).copy())
        return self.queue.popleft().copy()


def evaluate_adapter(
    directory,
    *,
    chars="永",
    device="auto",
    execute_steps=4,
    denoise_steps=None,
    seed=7,
    allow_training_chars=False,
    report_path=None,
    optimize_inference=False,
    trim_language_padding=False,
    cache_static_inputs=False,
):
    from shodo.artifacts import provenance
    from shodo.config import config_from_dict
    from shodo.runtime import sensor_rollout

    if not chars:
        raise ValueError("Evaluation needs characters")
    policy = SmolController(
        directory,
        device=device,
        execute_steps=execute_steps,
        denoise_steps=denoise_steps,
        seed=seed,
        optimize_inference=optimize_inference,
        trim_language_padding=trim_language_padding,
        cache_static_inputs=cache_static_inputs,
    )
    try:
        overlap = set(chars) & set(policy.report["dataset"]["chars"])
        if overlap and not allow_training_chars:
            raise ValueError(f"Evaluation overlaps training characters: {sorted(overlap)}")
        rows = []
        for char in chars:
            split = "training" if char in overlap else "held-out"
            print(f"Evaluating SmolVLA on {split} {char} …", flush=True)
            start = time.perf_counter()
            metrics = sensor_rollout(
                char, policy, seed=seed, config=config_from_dict(policy.report["dataset"]["config"])
            )
            metrics["wall_seconds"] = time.perf_counter() - start
            metrics["split"] = split
            metrics["inference_seconds"] = list(policy.inference_seconds)
            durations = np.asarray(policy.inference_seconds)
            metrics["inference_latency"] = {
                "calls": len(durations),
                "median_seconds": float(np.median(durations)) if len(durations) else None,
                "p95_seconds": float(np.quantile(durations, 0.95)) if len(durations) else None,
                "execution_horizon_seconds": execute_steps
                * policy.report["dataset"]["action_contract"]["dt"],
            }
            decisions = np.asarray(policy.decision_seconds)
            planning = np.asarray(policy.planning_seconds)
            period = policy.report["dataset"]["action_contract"]["dt"]
            metrics["actor_latency"] = {
                "scope": "camera preprocessing plus policy call; excludes image acquisition, physics and model loading",
                "decision_seconds": decisions.tolist(),
                "planning_seconds": planning.tolist(),
                "control_period_seconds": period,
                "control_deadline_misses": int(np.sum(decisions > period)),
                "planning_horizon_misses": int(np.sum(planning > execute_steps * period)),
                "planning_p95_seconds": float(np.quantile(planning, 0.95))
                if len(planning)
                else None,
            }
            rows.append(metrics)
            ink = metrics["ink_rmse_mm"]
            ink_text = "missing" if ink is None else f"{ink:.3f} mm"
            print(
                f"SmolVLA {char}: ink={ink_text}, truncated={metrics['truncated']}, wall={metrics['wall_seconds']:.1f}s",
                flush=True,
            )
        report = {
            "rows": rows,
            "provenance": provenance(),
            "execute_steps": execute_steps,
            "denoise_steps": policy.denoise_steps,
            "objective": adapter_objective(policy.report),
            "seed": seed,
            "allow_training_chars": allow_training_chars,
            "optimize_inference": optimize_inference,
            "trim_language_padding": trim_language_padding,
            "cache_static_inputs": cache_static_inputs,
            "checkpoint": policy.checkpoint_provenance,
            "timing": "synchronous simulation; not real-time hardware execution",
        }
        path = Path(report_path) if report_path is not None else Path(directory) / "evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Evaluation: {path.resolve()}")
        return report
    finally:
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "verify", "evaluate"))
    parser.add_argument("--episodes", type=Path, default=Path("runs/smolvla-recordings/episodes"))
    parser.add_argument("--dataset", type=Path, default=Path("runs/smolvla-data"))
    parser.add_argument("--output", type=Path, default=Path("runs/smolvla-adapter"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--objective",
        choices=("flow_matching", "action_regression"),
        default="flow_matching",
        help="Training loss: standard flow matching or explicit deterministic action regression",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Train from a checked parent adapter with a fresh optimizer and RNG",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--chars", default="永")
    parser.add_argument("--execute-steps", type=int, default=4)
    parser.add_argument(
        "--denoise-steps",
        type=int,
        help="Defaults to 10 for flow matching or 1 for action regression",
    )
    parser.add_argument(
        "--allow-training-chars",
        action="store_true",
        help="Explicit development rollout, not held-out evaluation",
    )
    parser.add_argument("--report", type=Path, help="Evaluation report destination")
    parser.add_argument("--supervision", choices=("applied", "expert"), default="applied")
    parser.add_argument(
        "--optimize-inference",
        action="store_true",
        help="Merge LoRA and cache unchanged prompt/vision embeddings",
    )
    parser.add_argument(
        "--trim-language-padding",
        action="store_true",
        help="Omit globally masked trailing language tokens at inference",
    )
    parser.add_argument(
        "--cache-static-inputs",
        action="store_true",
        help="Cache unchanged input embeddings, keeping current state live; requires --optimize-inference",
    )
    args = parser.parse_args()
    if args.warm_start is not None and args.command != "train":
        parser.error("--warm-start applies only to train")
    try:
        if args.command == "prepare":
            from shodo.vla_data import prepare_dataset

            path = prepare_dataset(
                sorted(args.episodes.glob("*.npz")),
                args.dataset,
                image_size=args.image_size,
                supervision=args.supervision,
            )
            print(f"Prepared dataset: {path.resolve()}")
        elif args.command == "train":
            train_lora(
                args.dataset,
                args.output,
                steps=args.steps,
                batch_size=args.batch_size,
                rank=args.rank,
                alpha=args.alpha,
                learning_rate=args.learning_rate,
                chunk_size=args.chunk_size,
                seed=args.seed,
                device=args.device,
                warm_start=args.warm_start,
                objective=args.objective,
            )
        elif args.command == "verify":
            verify_reload(args.output, device=args.device)
        else:
            evaluate_adapter(
                args.output,
                chars=args.chars,
                device=args.device,
                execute_steps=args.execute_steps,
                denoise_steps=args.denoise_steps,
                seed=args.seed,
                allow_training_chars=args.allow_training_chars,
                report_path=args.report,
                optimize_inference=args.optimize_inference,
                trim_language_padding=args.trim_language_padding,
                cache_static_inputs=args.cache_static_inputs,
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"SmolVLA: {error}\n")


if __name__ == "__main__":
    main()
