"""PPO experiments, including a frozen imitation controller plus learned residual."""

import hashlib
import json
import shutil
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from filelock import FileLock
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from shodo.artifacts import provenance, snapshot_source
from shodo.config import SimConfig, config_from_dict
from shodo.data import TRAIN
from shodo.device import resolve_device
from shodo.env import ACTIONS, INK_LOAD_THRESHOLD, OBSERVATION_VERSION, OBSERVATIONS, ShodoEnv
from shodo.learning import load_policy


class InkObjective(gym.Wrapper):
    """Extra dense credit for loaded ink-center accuracy during drawing only.

    This changes training rewards, never deposition, observations or dynamics.
    Missing contact earns no ink bonus. Air tracking retains the base reward.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        env = self.unwrapped
        i = env.index - 1
        loads = env.brush.ink_loads
        if env.stroke_ids[i] >= 0 and loads.sum() > INK_LOAD_THRESHOLD:
            center = np.average(env.brush.ink_positions[:, :2], axis=0, weights=loads)
            error = float(np.linalg.norm(center - env.path[i, :2]))
            reward += 0.65 * np.exp(-((error / 0.002) ** 2))
            info["ink_error_m"] = error
        return observation, float(reward), terminated, truncated, info


class ResidualControl(gym.Wrapper):
    def __init__(self, env, base, scale=0.3):
        super().__init__(env)
        self.base = base
        self.scale = scale
        self.observation = None

    def reset(self, **kwargs):
        self.observation, info = self.env.reset(**kwargs)
        return self.observation, info

    def step(self, action):
        combined = np.clip(self.base(self.observation) + self.scale * np.asarray(action), -1, 1)
        self.observation, reward, terminated, truncated, info = self.env.step(combined)
        return self.observation, reward, terminated, truncated, info


class Progress(BaseCallback):
    def __init__(self, directory, metadata):
        super().__init__()
        self.directory = directory
        self.metadata = metadata
        self.start = time.perf_counter()
        self.errors = deque(maxlen=10000)

    def save(self, status):
        temporary = self.directory / "ppo.pending.zip"
        self.model.save(temporary)
        temporary.replace(self.directory / "ppo.zip")
        report = {
            **self.metadata,
            "actual_steps": self.num_timesteps,
            "status": status,
            "seconds": time.perf_counter() - self.start,
        }
        (self.directory / "ppo.json").write_text(json.dumps(report, indent=2) + "\n")

    def _on_step(self):
        self.errors.append(self.locals["infos"][0].get("error_m", 0.0))
        if self.num_timesteps % 10000 == 0:
            progress = {
                "steps": self.num_timesteps,
                "seconds": time.perf_counter() - self.start,
                "recent_tracking_rmse_mm": float(np.sqrt(np.mean(np.square(self.errors))) * 1000),
            }
            (self.directory / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
        if self.num_timesteps % 65536 == 0:
            self.save("running")
        return True


def train_ppo(
    directory,
    *,
    steps=32768,
    seed=7,
    config=None,
    base_checkpoint=None,
    resume=False,
    chars=TRAIN,
    objective="tracking",
    device="cpu",
):
    if steps <= 0 or not chars:
        raise ValueError("Positive step count and training characters are required")
    if objective not in ("tracking", "ink"):
        raise ValueError("PPO objective must be tracking or ink")
    resolved = resolve_device(device)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = replace(config or SimConfig(randomize=True), record=False)
    with FileLock(directory / ".training.lock", timeout=0):
        metadata = {
            "seed": seed,
            "requested_steps": steps,
            "train_chars": chars,
            "observation_version": OBSERVATION_VERSION,
            "config": config.to_dict(),
            "residual_scale": 0.3 if base_checkpoint is not None else 0.0,
            "objective": objective,
            "provenance": provenance(),
            "device": {"requested": device, "resolved": str(resolved)},
        }
        if base_checkpoint is not None:
            base_checkpoint = Path(base_checkpoint)
            metadata["base_sha256"] = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
        if resume:
            previous = json.loads((directory / "ppo.json").read_text())
            previous["config"] = config_from_dict(previous["config"]).to_dict()
            previous.setdefault("train_chars", TRAIN)
            previous.setdefault("objective", "tracking")
            for key in (
                "observation_version",
                "residual_scale",
                "base_sha256",
                "config",
                "seed",
                "train_chars",
                "objective",
            ):
                if previous.get(key) != metadata.get(key):
                    raise ValueError(f"Cannot resume after changing {key}; use a new run directory")
            metadata["resumed_from_steps"] = previous["actual_steps"]
            metadata["previous_source_snapshot"] = previous["source_snapshot"]
            metadata["resume_semantics"] = (
                "optimizer/checkpoint continuation; environment and RNG are not a bitwise replay"
            )
        metadata["source_snapshot"] = snapshot_source(directory, "ppo")
        env = ShodoEnv(chars=chars, config=config)
        if objective == "ink":
            env = InkObjective(env)
        if base_checkpoint is not None:
            local_base = directory / "bc.pt"
            if base_checkpoint.resolve() != local_base.resolve():
                shutil.copyfile(base_checkpoint, local_base)
            env = ResidualControl(env, load_policy(local_base, device=device))
        env = Monitor(env, str(directory / "ppo-monitor.csv"), override_existing=not resume)
        try:
            if resume:
                model = PPO.load(directory / "ppo", env=env, device=resolved)
            else:
                model = PPO(
                    "MlpPolicy",
                    env,
                    seed=seed,
                    device=resolved,
                    verbose=0,
                    n_steps=2048,
                    batch_size=256,
                    learning_rate=3e-4,
                    policy_kwargs={"net_arch": [128, 128], "activation_fn": torch.nn.SiLU},
                )
            callback = Progress(directory, metadata)
            model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=not resume)
            callback.save("complete")
        finally:
            env.close()


def load_controller(directory, *, device="cpu"):
    resolved = resolve_device(device)
    directory = Path(directory)
    metadata = json.loads((directory / "ppo.json").read_text())
    if metadata.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("PPO observation contract is incompatible; retrain the policy")
    model = PPO.load(directory / "ppo", device=resolved)
    if model.observation_space.shape != (OBSERVATIONS,) or model.action_space.shape != (ACTIONS,):
        raise ValueError("PPO shape mismatch")
    scale = metadata.get("residual_scale", 0.0)
    base = None
    if scale:
        checkpoint = directory / "bc.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["base_sha256"]:
            raise ValueError("Residual base controller does not match the training checkpoint")
        base = load_policy(checkpoint, device=device)

    def predict(observation):
        action = model.predict(observation, deterministic=True)[0]
        return np.clip(base(observation) + scale * action, -1, 1) if base else action

    predict.device = str(resolved)
    predict.requested_device = device
    return predict
