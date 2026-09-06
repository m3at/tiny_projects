"""Headless SmolVLA recordings using the shared executed-motion MP4 exporter."""

import argparse
import time
from pathlib import Path

from filelock import FileLock

from shodo.artifacts import save_rollout
from shodo.config import config_from_dict
from shodo.learning import rollout
from shodo.smolvla import (
    SmolController,
    _read_adapter_report,
    adapter_objective,
    resolve_denoise_steps,
)
from shodo.video import require_ffmpeg

_SUFFIXES = (".mp4", ".png", ".npz", ".json", "-scene.png")


def demo_adapter(
    adapter,
    output_directory,
    *,
    chars="永",
    device="auto",
    execute_steps=None,
    denoise_steps=None,
    seed=7,
    optimize_inference=False,
    trim_language_padding=False,
    cache_static_inputs=False,
):
    """Save bounded presentation artifacts; existing destinations are never overwritten.

    The video follows simulation time. Slow synchronous inference does not appear
    as a playback pause and must not be interpreted as real-time hardware control.
    """
    if not isinstance(chars, str) or not chars or any(char.isspace() for char in chars):
        raise ValueError("Demo needs a nonempty string of characters without whitespace")
    if len(set(chars)) != len(chars):
        raise ValueError("Demo characters must be distinct")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if any(
        type(value) is not bool
        for value in (optimize_inference, trim_language_padding, cache_static_inputs)
    ):
        raise TypeError("Inference options must be booleans")
    if cache_static_inputs and not optimize_inference:
        raise ValueError("Static input caching requires optimize_inference")
    require_ffmpeg()
    output = Path(output_directory)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    targets = {char: output / f"smolvla-{ord(char):05x}" for char in chars}
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".smolvla-demo.lock", timeout=0):
        for base in targets.values():
            for suffix in _SUFFIXES:
                path = Path(f"{base}{suffix}")
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f"Demo output exists: {path}; use a new destination")
        report = _read_adapter_report(Path(adapter), execute_steps=execute_steps)
        execute_steps = min(4, report["chunk_size"]) if execute_steps is None else execute_steps
        denoise_steps = resolve_denoise_steps(report, denoise_steps)
        config = config_from_dict(report["dataset"]["config"])
        policy = SmolController(
            adapter,
            device=device,
            execute_steps=execute_steps,
            denoise_steps=denoise_steps,
            seed=seed,
            optimize_inference=optimize_inference,
            trim_language_padding=trim_language_padding,
            cache_static_inputs=cache_static_inputs,
        )
        results = []
        try:
            for char, base in targets.items():
                print(f"Drawing SmolVLA {char}…", flush=True)
                start = time.perf_counter()
                metrics, history, paper, frames = rollout(
                    char, policy, seed=seed, frames=True, config=config
                )
                metrics = {
                    **metrics,
                    "wall_seconds": time.perf_counter() - start,
                    "smolvla": {
                        "objective": adapter_objective(policy.report),
                        "execute_steps": execute_steps,
                        "denoise_steps": denoise_steps,
                        "optimize_inference": optimize_inference,
                        "trim_language_padding": trim_language_padding,
                        "cache_static_inputs": cache_static_inputs,
                        "training_character": char in policy.report["dataset"]["chars"],
                        "timing": "simulation-timed presentation; synchronous inference, not real-time hardware",
                    },
                    "inference_seconds": list(policy.inference_seconds),
                    "decision_seconds": list(policy.decision_seconds),
                    "planning_seconds": list(policy.planning_seconds),
                }
                print("Encoding MP4…", flush=True)
                save_rollout(base, (metrics, history, paper, frames))
                ink = metrics["ink_rmse_mm"]
                ink_label = "missing" if ink is None else f"{ink:.3f} mm"
                print(
                    f"SmolVLA {char}: ink={ink_label} · peak force={metrics['max_force_n']:.3f} N · truncated={metrics['truncated']}",
                    flush=True,
                )
                for label, suffix in (
                    ("Video", ".mp4"),
                    ("Paper", ".png"),
                    ("Metrics", ".json"),
                    ("Trajectory", ".npz"),
                    ("Scene", "-scene.png"),
                ):
                    print(f"  {label}: {Path(f'{base}{suffix}').resolve()}", flush=True)
                results.append({"base": str(base.resolve()), "metrics": metrics})
        finally:
            policy.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("runs/smolvla-adapter"))
    parser.add_argument("--output-directory", type=Path, default=Path("runs/smolvla-demo"))
    parser.add_argument("--chars", default="永")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--execute-steps", type=int, help="default: min(4, trained chunk size)")
    parser.add_argument("--denoise-steps", type=int, help="default: adapter objective's sampler")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--optimize-inference", action="store_true")
    parser.add_argument("--trim-language-padding", action="store_true")
    parser.add_argument("--cache-static-inputs", action="store_true")
    args = parser.parse_args()
    try:
        demo_adapter(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"SmolVLA demo: {error}\n")


if __name__ == "__main__":
    main()
