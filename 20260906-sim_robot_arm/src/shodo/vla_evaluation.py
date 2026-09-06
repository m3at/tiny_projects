"""Teacher-forced action diagnostics on separate native recording episodes.

Inputs use the adapter's training normalization, never the evaluation dataset's
statistics. This measures action prediction on recorded states, not closed-loop
recovery, real-time execution, or held-out character generalization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def action_metrics(predicted, targets):
    predicted, targets = np.asarray(predicted), np.asarray(targets)
    if (
        predicted.ndim != 2
        or predicted.shape != targets.shape
        or predicted.shape[1] != 6
        or not len(predicted)
        or not np.isfinite(predicted).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Expected matching nonempty finite [N, 6] commands")
    error = predicted - targets
    return {
        "samples": len(predicted),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "per_axis_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "mae": float(np.mean(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def score_adapter(
    adapter,
    dataset,
    *,
    device="auto",
    samples=256,
    batch_size=8,
    denoise_steps=None,
    seed=107,
    optimize_inference=False,
):
    from shodo.smolvla import (
        _read_adapter_report,
        adapter_objective,
        inference_noise,
        load_adapter,
        model_batch,
        resolve_denoise_steps,
        training_episode_hashes,
    )
    from shodo.vla_data import PreparedDataset
    from shodo.vla_inference import PromptCache

    for value in (samples, batch_size):
        if type(value) is not int or value < 1:
            raise ValueError("Samples, batch size and denoising steps must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Use a nonnegative 32-bit integer seed")
    data = PreparedDataset(dataset, chunk_size=1)
    report = _read_adapter_report(adapter)
    denoise_steps = resolve_denoise_steps(report, denoise_steps)
    trained = report["dataset"]
    for key in ("state_contract", "action_contract", "image_preprocessing", "observation_contract"):
        if data.manifest[key] != trained[key]:
            raise ValueError(f"Evaluation {key} differs from adapter training")
    train_hashes = training_episode_hashes(report)
    if train_hashes & {episode["sha256"] for episode in data.manifest["episodes"]}:
        raise ValueError("Action evaluation requires separate recording episodes")
    model, tokenizer, loaded_report = load_adapter(
        adapter, device=device, denoise_steps=denoise_steps
    )
    if loaded_report != report:
        raise ValueError("Adapter metadata changed after evaluation preflight")
    if optimize_inference:
        model = model.merge_and_unload(safe_merge=True)
        model.eval()
        tokenizer = PromptCache(tokenizer, trim_padding=True)
    resolved = next(model.parameters()).device
    mean, std = (np.asarray(trained["stats"]["state"][key], np.float32) for key in ("mean", "std"))
    action_mean, action_std = (
        np.asarray(trained["stats"]["actions"][key], np.float32) for key in ("mean", "std")
    )
    indices = np.random.default_rng(seed).choice(len(data), min(samples, len(data)), replace=False)
    generator = torch.Generator().manual_seed(seed)
    predictions, targets = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            items = [data[int(index)] for index in indices[start : start + batch_size]]
            states = np.stack([item["state"] for item in items]) * data.state_std + data.state_mean
            batch = model_batch(
                {
                    "state": torch.from_numpy((states - mean) / std),
                    "image": torch.from_numpy(np.stack([item["image"] for item in items])),
                    "task": [item["task"] for item in items],
                },
                tokenizer,
                resolved,
            )
            noise = inference_noise(report, len(items), resolved, generator=generator)
            chunks = model.predict_action_chunk(batch, noise=noise).cpu().numpy()
            if (
                chunks.shape != (len(items), report["chunk_size"], 6)
                or not np.isfinite(chunks).all()
            ):
                raise ValueError("Model must predict finite [batch, chunk_size, 6] action chunks")
            predictions.append(np.clip(chunks[:, 0] * action_std + action_mean, -1, 1))
            targets.append(
                np.stack([item["actions"][0] for item in items]) * data.action_std
                + data.action_mean
            )
    predicted, target = np.concatenate(predictions), np.concatenate(targets)
    return {
        "scope": __doc__,
        "split": "separate recording episodes; teacher-forced first-action prediction",
        "adapter_sha256": report["adapter_sha256"],
        "training_episode_sha256": sorted(train_hashes),
        "dataset": str(Path(dataset).resolve()),
        "episodes": [
            {key: episode[key] for key in ("sha256", "char", "source")}
            for episode in data.manifest["episodes"]
        ],
        "characters_seen_in_training": sorted(set(data.manifest["chars"]) & set(trained["chars"])),
        "supervision": data.manifest.get("supervision", {"source": "applied"}),
        "seed": seed,
        "indices": indices.tolist(),
        "device": str(resolved),
        "batch_size": batch_size,
        "denoise_steps": denoise_steps,
        "objective": adapter_objective(report),
        "optimize_inference": optimize_inference,
        "units": "normalized effective command increments",
        "prediction": action_metrics(predicted, target),
        "training_mean_baseline": action_metrics(
            np.broadcast_to(action_mean, target.shape), target
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--denoise-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--optimize-inference", action="store_true")
    args = parser.parse_args()
    result = score_adapter(
        args.adapter,
        args.dataset,
        device=args.device,
        samples=args.samples,
        batch_size=args.batch_size,
        denoise_steps=args.denoise_steps,
        seed=args.seed,
        optimize_inference=args.optimize_inference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Recorded-state action RMSE: {result['prediction']['rmse']:.5f}; training-mean baseline {result['training_mean_baseline']['rmse']:.5f}"
    )
    print(f"Action diagnostic: {args.output.resolve()}")


if __name__ == "__main__":
    main()
