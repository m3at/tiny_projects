"""Optional SmolVLA replay latency audit; excludes loading, physics and image acquisition.

Run in the locked integrations/smolvla environment. Measured calls include raw RGB
preprocessing, state normalization, tokenization/cache lookup, host/device copies,
vision/action inference, output transfer, denormalization and clipping. This is a
synchronous component benchmark, not a hardware deadline or drawing-quality test.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from shodo.dataset import load_episode
from shodo.smolvla import (
    adapter_objective,
    inference_noise,
    load_adapter,
    model_batch,
    resolve_denoise_steps,
    synchronize,
)
from shodo.vla_data import preprocess_image, state_from_observation
from shodo.vla_inference import PromptCache, cached_static_prefix, cached_vision


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Timings must be a nonempty finite nonnegative vector")
    return {
        "samples": len(values),
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "calls_over_20ms": int(np.count_nonzero(values > 0.02)),
        "seconds": values.tolist(),
    }


def _denoising_sweep(report, requested):
    """Resolve the sampler sweep before loading model weights or replay arrays."""
    objective = adapter_objective(report)
    if requested is None:
        requested = (1,) if objective == "action_regression" else (1, 2, 4, 10)
    if (
        not isinstance(requested, (list, tuple))
        or not requested
        or any(type(value) is not int or value < 1 for value in requested)
    ):
        raise ValueError("Denoising counts must be a nonempty list of positive integers")
    return tuple(resolve_denoise_steps(report, value) for value in requested)


def _replay(model, tokenizer, report, episode, indices, device, *, cache=None, seed=41):
    stats = report["dataset"]["stats"]
    state_mean, state_std = (
        np.asarray(stats["state"][key], dtype=np.float32) for key in ("mean", "std")
    )
    action_mean, action_std = (
        np.asarray(stats["actions"][key], dtype=np.float32) for key in ("mean", "std")
    )
    arrays = episode.arrays
    camera_indices = (
        np.searchsorted(
            arrays["camera_times_s"], arrays["observation_times_s"][indices], side="right"
        )
        - 1
    )
    if np.any(camera_indices < 0):
        raise ValueError("Replay decision has no causal camera frame")
    model.reset()
    times, model_times, outputs, fresh = [], [], [], []
    previous_frame = None
    with torch.no_grad():
        for index, camera_index in zip(indices, camera_indices, strict=True):
            synchronize(device)
            start = time.perf_counter()
            image = (
                preprocess_image(
                    arrays["camera_frames"][camera_index],
                    report["dataset"]["image_preprocessing"]["size"],
                ).astype(np.float32)
                / 255
            )
            state = state_from_observation(
                arrays["observations"][index], report["dataset"]["sensors"]["history"]
            )
            state = (state - state_mean) / state_std
            task = (
                f"Draw {episode.metadata['char']} on paper following the supplied stroke reference."
            )
            inputs = model_batch(
                {
                    "state": torch.from_numpy(state[None]),
                    "image": torch.from_numpy(image[None]),
                    "task": [task],
                },
                tokenizer,
                device,
            )
            noise = inference_noise(
                report,
                1,
                device,
                generator=torch.Generator().manual_seed(seed + int(index)),
            )
            if cache is not None:
                cache.frame = int(camera_index)
                if hasattr(cache, "task"):
                    cache.task = task
            synchronize(device)
            model_start = time.perf_counter()
            actions = model.predict_action_chunk(inputs, noise=noise)[0].cpu().numpy()
            synchronize(device)
            model_times.append(time.perf_counter() - model_start)
            unbounded = actions * action_std + action_mean
            applied = np.clip(unbounded, -1, 1)
            synchronize(device)
            times.append(time.perf_counter() - start)
            if not np.isfinite(unbounded).all():
                raise RuntimeError("Nonfinite benchmark action")
            outputs.append((actions.copy(), unbounded.copy(), applied.copy()))
            fresh.append(camera_index != previous_frame)
            previous_frame = camera_index
    return times, model_times, np.asarray(outputs), fresh


def benchmark(
    adapter,
    episode_path,
    *,
    device="cpu",
    samples=20,
    warmup=2,
    stride=4,
    start=0,
    denoise_steps=None,
):
    """Compare one loaded adapter before/after merge, with identical replay noise."""
    from shodo.artifacts import provenance

    for value, name in ((samples, "samples"), (stride, "stride")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(warmup) is not int or warmup < 0 or type(start) is not int or start < 0:
        raise ValueError("warmup and start must be nonnegative integers")
    preflight = json.loads((Path(adapter) / "training.json").read_text())
    if not isinstance(preflight, dict):
        raise TypeError("Adapter training metadata must be a JSON object")
    denoise_steps = _denoising_sweep(preflight, denoise_steps)
    episode = load_episode(episode_path)
    indices = np.arange(start, start + samples * stride, stride)
    if indices[-1] >= len(episode):
        raise ValueError("Requested replay does not fit in the episode")
    if "camera_frames" not in episode.arrays:
        raise ValueError("Benchmark requires raw camera frames")
    torch.set_num_threads(1)
    model, tokenizer, report = load_adapter(adapter, device=device, denoise_steps=denoise_steps[0])
    resolved = str(next(model.parameters()).device)
    if episode.metadata["sensors"] != report["dataset"]["sensors"]:
        raise ValueError("Replay sensor settings differ from adapter training settings")
    rows, references, merged_references, trimmed_references = [], {}, {}, {}

    def run_variant(name, model, tokenizer, cache=None):
        for count in denoise_steps:
            # Both wrappers and merged policies expose the same underlying config.
            model.config.num_steps = count
            _replay(
                model, tokenizer, report, episode, indices[:1].repeat(warmup), resolved, cache=cache
            )
            if cache is not None:
                cache.clear()
            times, model_times, outputs, fresh = _replay(
                model, tokenizer, report, episode, indices, resolved, cache=cache
            )
            if name == "adapter":
                references[count] = outputs
            elif name == "merged":
                merged_references[count] = outputs
            elif name == "merged_prompt_trim_vision":
                trimmed_references[count] = outputs
            differences = np.max(np.abs(outputs - references[count]), axis=(0, 2, 3))
            row = {
                "variant": name,
                "denoise_steps": count,
                **latency_summary(times),
                "model_and_output_transfer": latency_summary(model_times),
                "max_action_difference": dict(
                    zip(
                        ("model_normalized", "command_unclipped", "command_clipped"),
                        differences.tolist(),
                        strict=True,
                    )
                ),
                "fresh_camera": latency_summary(np.asarray(times)[fresh]),
                "held_camera": latency_summary(np.asarray(times)[np.logical_not(fresh)])
                if not all(fresh)
                else None,
                "vision_cache_hits": cache.hits if cache is not None else 0,
                "static_prefix_cache_hits": cache.hits
                if name == "merged_prompt_trim_static_prefix"
                else 0,
                "max_difference_from_trimmed_vision": float(
                    np.max(np.abs(outputs - trimmed_references[count]))
                )
                if count in trimmed_references
                else None,
                "max_difference_from_merged": float(
                    np.max(np.abs(outputs - merged_references[count]))
                )
                if count in merged_references
                else None,
                "max_action_difference_from_merged": dict(
                    zip(
                        ("model_normalized", "command_unclipped", "command_clipped"),
                        np.max(np.abs(outputs - merged_references[count]), axis=(0, 2, 3)).tolist(),
                        strict=True,
                    )
                )
                if count in merged_references
                else None,
                "execution_budget_ms": float(
                    stride * report["dataset"]["action_contract"]["dt"] * 1000
                ),
                "calls_over_execution_budget": int(
                    np.count_nonzero(
                        np.asarray(times) > stride * report["dataset"]["action_contract"]["dt"]
                    )
                ),
            }
            rows.append(row)
            print(
                f"{resolved} {name} denoise={count}: p50={row['p50_ms']:.1f}ms p95={row['p95_ms']:.1f}ms, max model-action difference={differences[0]:.3g}",
                flush=True,
            )

    run_variant("adapter", model, tokenizer)
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    run_variant("merged", model, tokenizer)
    run_variant("merged_prompt", model, PromptCache(tokenizer))
    with cached_vision(model) as cache:
        run_variant("merged_prompt_vision", model, PromptCache(tokenizer), cache)
    with cached_vision(model) as cache:
        run_variant(
            "merged_prompt_trim_vision",
            model,
            PromptCache(tokenizer, trim_padding=True),
            cache,
        )
    with cached_static_prefix(model) as cache:
        run_variant(
            "merged_prompt_trim_static_prefix",
            model,
            PromptCache(tokenizer, trim_padding=True),
            cache,
        )
    with Path(episode_path).open("rb") as source:
        episode_hash = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "device": resolved,
        "objective": adapter_objective(report),
        "noise_sampler": "zeros"
        if adapter_objective(report) == "action_regression"
        else "CPU Gaussian, seed 41 + decision index",
        "adapter_sha256": report["adapter_sha256"],
        "base_revision": report["base_revision"],
        "provenance": provenance(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("lerobot", "peft", "torch", "transformers")
        },
        "episode": str(Path(episode_path).resolve()),
        "episode_sha256": episode_hash,
        "decision_indices": indices.tolist(),
        "camera_acquisition_indices": (
            np.searchsorted(
                episode.arrays["camera_times_s"],
                episode.arrays["observation_times_s"][indices],
                side="right",
            )
            - 1
        ).tolist(),
        "warmup_calls_per_case": warmup,
        "torch_threads": torch.get_num_threads(),
        "scope": __doc__,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("runs/smolvla-adapter"))
    parser.add_argument(
        "--episode", type=Path, default=Path("runs/smolvla-recordings/episodes/episode-000000.npz")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--denoise-steps",
        type=int,
        nargs="+",
        help="default: 1 for action regression; 1 2 4 10 for flow matching",
    )
    args = parser.parse_args()
    report = benchmark(
        args.adapter,
        args.episode,
        device=args.device,
        samples=args.samples,
        warmup=args.warmup,
        stride=args.stride,
        start=args.start,
        denoise_steps=args.denoise_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Benchmark: {args.output.resolve()}")


if __name__ == "__main__":
    main()
