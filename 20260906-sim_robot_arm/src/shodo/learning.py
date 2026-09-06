"""Behavior cloning baseline and reproducible held-out evaluation."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch import nn

from shodo.artifacts import provenance, snapshot_source
from shodo.config import SimConfig
from shodo.data import TEST, TRAIN
from shodo.env import ACTIONS, OBSERVATION_VERSION, OBSERVATIONS, ShodoEnv


def network():
    return nn.Sequential(
        nn.Linear(OBSERVATIONS, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, ACTIONS),
        nn.Tanh(),
    )


def load_policy(file="runs/v2/bc.pt"):
    model = network()
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    if checkpoint.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("Checkpoint contract is obsolete; retrain into runs/v2")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def predict(obs):
        with torch.no_grad():
            return model(torch.as_tensor(obs)).numpy()

    return predict


def train(episodes=28, epochs=35, seed=7, output="runs/v2/bc.pt", config=None, chars=TRAIN):
    if episodes <= 0 or epochs <= 0 or not chars:
        raise ValueError("Positive episode/epoch counts and training characters are required")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = replace(config or SimConfig(randomize=True), record=False)
    with FileLock(path.parent / ".training.lock", timeout=0):
        metadata = {
            "provenance": provenance(),
            "source_snapshot": snapshot_source(path.parent, "bc"),
        }
        return _train(episodes, epochs, seed, path, config, chars, metadata)


def _train(episodes, epochs, seed, path, config, chars, metadata):
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = ShodoEnv(chars=chars, config=config)
    observations, actions = [], []
    truncated_episodes = 0
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode, options={"char": chars[episode % len(chars)]})
            while True:
                expert = env.expert()
                observations.append(obs)
                actions.append(expert)
                noisy = expert + rng.normal(0, 0.12 if episode % 2 else 0.03, ACTIONS)
                obs, _, terminated, truncated, _ = env.step(noisy)
                if terminated or truncated:
                    truncated_episodes += int(truncated)
                    break
            if (episode + 1) % 7 == 0:
                print(f"episodes={episode + 1}/{episodes} labels={len(actions)}", flush=True)
    finally:
        env.close()
    x, y = torch.tensor(np.asarray(observations)), torch.tensor(np.asarray(actions))
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(epochs):
        indices = torch.randperm(len(x))
        for batch in indices.split(256):
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch + 1} mse={loss.item():.6f}", flush=True)
    temporary = path.with_suffix(".pending.pt")
    torch.save(
        {"state_dict": model.state_dict(), "observation_version": OBSERVATION_VERSION}, temporary
    )
    temporary.replace(path)
    metadata = {
        **metadata,
        "seed": seed,
        "episodes": episodes,
        "epochs": epochs,
        "samples": len(x),
        "train_chars": chars,
        "held_out_chars": TEST,
        "algorithm": "behavior cloning",
        "observation_version": OBSERVATION_VERSION,
        "config": env.config.to_dict(),
        "final_batch_mse": loss.item(),
        "truncated_episodes": truncated_episodes,
        "seconds": time.perf_counter() - start,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {path} from {len(x)} expert labels")
    return metadata


def rollout(char, policy="expert", seed=17, frames=False, config=None, material=None):
    env = ShodoEnv(chars=char, config=replace(config or SimConfig(), record=True))
    images = []
    total_reward = 0
    try:
        obs, _ = env.reset(seed=seed, options={"material": material or {}})
        while True:
            if policy == "expert":
                action = env.expert()
            elif policy == "zero":
                action = np.zeros(ACTIONS)
            else:
                action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if frames and (env.index % 5 == 0 or terminated):
                images.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break
        history = np.asarray(env.history)
        errors = np.linalg.norm(np.c_[history[:, 20:22], history[:, 2]] - history[:, 3:6], axis=1)
        down = history[:, 13] >= 0
        contact = down & (history[:, 17] > 0)
        lifted = history[:, 5] > 0.015
        metrics = {
            "char": char,
            "seed": seed,
            "steps": len(history),
            "rmse_mm": float(np.sqrt(np.mean(errors**2)) * 1000),
            "tip_rmse_mm": float(
                np.sqrt(np.mean(np.sum((history[:, :3] - history[:, 3:6]) ** 2, axis=1))) * 1000
            ),
            "ink_rmse_mm": float(
                np.sqrt(
                    np.mean(np.sum((history[contact, 20:22] - history[contact, 3:5]) ** 2, axis=1))
                )
                * 1000
            )
            if contact.any()
            else None,
            "p95_mm": float(np.quantile(errors, 0.95) * 1000),
            "mean_reward": total_reward / len(history),
            "draw_contact_fraction": float(np.mean(history[down, 14] > 0.01))
            if down.any()
            else None,
            "lift_clear_fraction": float(np.mean(history[lifted, 2] > 0.01))
            if lifted.any()
            else None,
            "truncated": bool(truncated),
            "force_rmse_n": float(np.sqrt(np.mean((history[down, 14] - history[down, 15]) ** 2)))
            if down.any()
            else None,
            "max_force_n": env.peak_force,
            "max_penetration_mm": getattr(env.brush, "max_penetration", 0.0) * 1000,
            "orientation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(history[:, 16] ** 2)))),
            "max_torque_fraction": env.peak_torque_fraction,
            "force_impulse_n_s": env.force_impulse.tolist(),
            "pigment_mass_error": float(
                abs(env.paper.mobile.sum() + env.paper.fixed.sum() - env.paper.deposited_pigment)
            ),
            **raster_metrics(env.paper, env.path[env.stroke_ids >= 0, :2]),
            "config": env.config.to_dict(),
            "actual_brush": asdict(env.brush.config),
        }
        return metrics, history, env.paper.image(), images
    finally:
        env.close()


def raster_metrics(paper, target_xy):
    """Geometric ink checks, not an aesthetic/calligraphy score.

    A pixel counts as visibly painted at optical depth >= 0.3. Coverage permits a
    1 mm centerline gap; spill is visible ink farther than 6 mm from the centerline.
    """
    density = (paper.mobile + paper.fixed) / paper.area_mm2
    painted = paper.config.optical_absorption * density >= 0.3
    n = paper.config.resolution
    pixels = np.floor(
        (target_xy - [paper.center_x - paper.config.extent / 2, -paper.config.extent / 2])
        / paper.dx
    ).astype(int)
    inside = ((pixels >= 0) & (pixels < n)).all(axis=1)
    pixels = pixels[inside]
    target = np.zeros_like(painted)
    target[pixels[:, 1], pixels[:, 0]] = True
    coverage = 0.0
    spill = None
    if painted.any():
        distances = distance_transform_edt(~painted) * paper.dx
        coverage = float(np.sum(distances[pixels[:, 1], pixels[:, 0]] <= 0.001) / len(target_xy))
        spill = float(np.mean((distance_transform_edt(~target) * paper.dx)[painted] > 0.006))
    return {
        "raster_coverage_fraction": coverage,
        "raster_spill_fraction": spill,
        "painted_area_mm2": float(painted.sum() * paper.area_mm2),
        "pigment_mass_ug": float((paper.mobile + paper.fixed).sum()),
    }


def load_ppo(directory="runs/v2"):
    from shodo.rl import load_controller

    return load_controller(directory)


def evaluate(
    output="runs/v2/evaluation.json",
    checkpoint="runs/v2/bc.pt",
    algorithm="learned",
    config=None,
    chars=TEST,
    seed=17,
):
    if not chars:
        raise ValueError("Evaluation needs at least one character")
    if algorithm == "learned":
        policy = load_policy(checkpoint)
    elif algorithm == "ppo":
        policy = load_ppo(Path(checkpoint).parent)
    elif algorithm in ("expert", "zero"):
        policy = algorithm
    else:
        raise ValueError(f"Unknown policy: {algorithm}")
    results = {}
    policies = {"expert": "expert", algorithm: policy, "zero": "zero"}
    for name, controller in policies.items():
        results[name] = []
        for char in chars:
            row = rollout(char, controller, config=config, seed=seed)[0]
            results[name].append(row)
            if config and config.brush.backend == "cable":
                print(
                    f"{name} {char}: {row['steps']} steps, RMSE {row['rmse_mm']:.3f} mm, "
                    f"peak {row['max_force_n']:.3f} N, truncated={row['truncated']}",
                    flush=True,
                )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    for name, rows in results.items():
        print(name, "mean held-out RMSE mm:", round(np.mean([r["rmse_mm"] for r in rows]), 3))
    return results
