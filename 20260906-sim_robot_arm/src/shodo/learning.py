"""Behavior cloning baseline and reproducible held-out evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from shodo.artifacts import provenance
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


def load_policy(file="runs/bc.pt"):
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


def train(episodes=28, epochs=35, seed=7, output="runs/bc.pt"):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = ShodoEnv(config=SimConfig(randomize=True))
    observations, actions = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode, options={"char": TRAIN[episode % len(TRAIN)]})
        while True:
            expert = env.expert()
            observations.append(obs)
            actions.append(expert)
            noisy = expert + rng.normal(0, 0.12 if episode % 2 else 0.03, ACTIONS)
            obs, _, terminated, truncated, _ = env.step(noisy)
            if terminated or truncated:
                break
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
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "observation_version": OBSERVATION_VERSION}, path)
    metadata = {
        "seed": seed,
        "episodes": episodes,
        "epochs": epochs,
        "samples": len(x),
        "train_chars": TRAIN,
        "held_out_chars": TEST,
        "algorithm": "behavior cloning",
        "observation_version": OBSERVATION_VERSION,
        "config": env.config.to_dict(),
        "final_batch_mse": loss.item(),
        "provenance": provenance(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {path} from {len(x)} expert labels")


def rollout(char, policy="expert", seed=17, frames=False, config=None):
    from dataclasses import replace

    env = ShodoEnv(chars=char, config=replace(config or SimConfig(), record=True))
    obs, _ = env.reset(seed=seed)
    images = []
    total_reward = 0
    try:
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
        errors = np.linalg.norm(history[:, :3] - history[:, 3:6], axis=1)
        down = history[:, 13] >= 0
        lifted = history[:, 5] > 0.015
        metrics = {
            "char": char,
            "steps": len(history),
            "rmse_mm": float(np.sqrt(np.mean(errors**2)) * 1000),
            "p95_mm": float(np.quantile(errors, 0.95) * 1000),
            "mean_reward": total_reward / len(history),
            "draw_contact_fraction": float(np.mean(history[down, 14] > 0.01)),
            "lift_clear_fraction": float(np.mean(history[lifted, 2] > 0.01)),
            "truncated": bool(truncated),
            "force_rmse_n": float(np.sqrt(np.mean((history[down, 14] - history[down, 15]) ** 2))),
            "max_force_n": float(history[:, 14].max()),
            "orientation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(history[:, 16] ** 2)))),
            "max_torque_fraction": float(history[:, 18].max()),
            "pigment_mass_error": float(
                abs(env.paper.mobile.sum() + env.paper.fixed.sum() - env.paper.deposited_pigment)
            ),
            "config": env.config.to_dict(),
        }
        return metrics, history, env.paper.image(), images
    finally:
        env.close()


def load_ppo(directory="runs"):
    from stable_baselines3 import PPO

    model = PPO.load(Path(directory) / "ppo", device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def evaluate(output="runs/evaluation.json", checkpoint="runs/bc.pt", algorithm="learned"):
    if algorithm == "learned":
        policy = load_policy(checkpoint)
    elif algorithm == "ppo":
        policy = load_ppo(Path(checkpoint).parent)
    elif algorithm in ("expert", "zero"):
        policy = algorithm
    else:
        raise ValueError(f"Unknown policy: {algorithm}")
    results = {
        name: [rollout(char, p)[0] for char in TEST]
        for name, p in [("expert", "expert"), (algorithm, policy), ("zero", "zero")]
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    for name, rows in results.items():
        print(name, "mean held-out RMSE mm:", round(np.mean([r["rmse_mm"] for r in rows]), 3))
    return results
