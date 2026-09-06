"""Behavior cloning baseline and reproducible held-out evaluation."""

import hashlib
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
from shodo.contracts import SENSOR_FEATURES, SensorConfig, sensor_contract
from shodo.data import TEST, TRAIN
from shodo.device import resolve_device
from shodo.env import ACTIONS, INK_LOAD_THRESHOLD, OBSERVATION_VERSION, OBSERVATIONS, ShodoEnv


def network(observations=OBSERVATIONS):
    return nn.Sequential(
        nn.Linear(observations, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, ACTIONS),
        nn.Tanh(),
    )


def load_policy(file="runs/bc.pt", *, device="cpu"):
    resolved = resolve_device(device)
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    if checkpoint.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("Checkpoint observation contract is incompatible; retrain the policy")
    sensors = SensorConfig(**checkpoint["sensors"]) if checkpoint.get("sensors") else None
    if checkpoint.get("sensor_contract") != (sensor_contract(sensors) if sensors else None):
        raise ValueError("Checkpoint sensor contract is incompatible; retrain the policy")
    model = network(SENSOR_FEATURES * sensors.history if sensors else OBSERVATIONS).to(resolved)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def predict(obs):
        with torch.no_grad():
            return model(torch.as_tensor(obs, device=resolved)).cpu().numpy()

    predict.device = str(resolved)
    predict.requested_device = device
    predict.sensor_config = sensors
    predict.checkpoint_provenance = {
        "path": str(Path(file).resolve()),
        "sha256": hashlib.sha256(Path(file).read_bytes()).hexdigest(),
        "algorithm": "behavior cloning",
    }
    return predict


def train(
    episodes=28,
    epochs=35,
    seed=7,
    output="runs/bc.pt",
    config=None,
    chars=TRAIN,
    device="cpu",
    sensors=None,
):
    if episodes <= 0 or epochs <= 0 or not chars:
        raise ValueError("Positive episode/epoch counts and training characters are required")
    resolved = resolve_device(device)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = replace(config or SimConfig(randomize=True), record=False)
    with FileLock(path.parent / ".training.lock", timeout=0):
        metadata = {
            "provenance": provenance(),
            "source_snapshot": snapshot_source(path.parent, "bc"),
            "device": {"requested": device, "resolved": str(resolved)},
        }
        return _train(episodes, epochs, seed, path, config, chars, metadata, resolved, sensors)


def _train(episodes, epochs, seed, path, config, chars, metadata, device, sensors=None):
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if sensors is not None:
        from shodo.runtime import SensorEnv

        env = SensorEnv(chars=chars, config=config, sensors=sensors)
    else:
        env = ShodoEnv(chars=chars, config=config)
    observations, actions = [], []
    truncated_episodes = 0
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode, options={"char": chars[episode % len(chars)]})
            while True:
                # Privileged teacher labels are training supervision, never student inputs.
                expert = env.unwrapped.expert()
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
    x = torch.tensor(np.asarray(observations), device=device)
    y = torch.tensor(np.asarray(actions), device=device)
    model = network(env.observation_space.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(epochs):
        indices = torch.randperm(len(x), device=device)
        for batch in indices.split(256):
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch + 1} mse={loss.item():.6f}", flush=True)
    temporary = path.with_suffix(".pending.pt")
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "observation_version": OBSERVATION_VERSION,
            "sensor_contract": sensor_contract(sensors) if sensors else None,
            "sensors": sensors.to_dict() if sensors else None,
        },
        temporary,
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
        "config": env.unwrapped.config.to_dict(),
        "sensor_contract": sensor_contract(sensors) if sensors else None,
        "sensors": sensors.to_dict() if sensors else None,
        "teacher": "privileged contact-state expert",
        "final_batch_mse": loss.item(),
        "truncated_episodes": truncated_episodes,
        "seconds": time.perf_counter() - start,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {path} from {len(x)} expert labels")
    return metadata


def rollout(
    char,
    policy="expert",
    seed=17,
    frames=False,
    config=None,
    material=None,
    sensors=None,
    recorder=None,
    camera_every=0,
    expert_noise=None,
):
    from shodo.runtime import SensorEnv, execute_step

    if type(camera_every) is not int or camera_every < 0:
        raise ValueError("camera_every must be a nonnegative integer")
    if camera_every and recorder is None:
        raise ValueError("Raw camera collection requires a training-data recorder")
    if expert_noise is not None and (
        not np.isfinite(expert_noise)
        or expert_noise < 0
        or not isinstance(policy, str)
        or policy != "oracle"
        or recorder is None
    ):
        raise ValueError(
            "Expert noise requires a recorder, oracle policy and finite nonnegative value"
        )
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53484F44]))
    sensors = sensors or getattr(policy, "sensor_config", None)
    if isinstance(policy, str) and policy == "classical":
        sensors = sensors or SensorConfig()
    if callable(policy):
        trained = getattr(policy, "sensor_config", None)
        if (trained is None) != (sensors is None):
            raise ValueError("Policy and environment observation contracts differ")
        if trained and trained.history != sensors.history:
            raise ValueError("Policy and environment sensor history differ")
    cfg = replace(config or SimConfig(), record=True)
    env = (
        SensorEnv(chars=char, config=cfg, sensors=sensors)
        if sensors
        else ShodoEnv(chars=char, config=cfg)
    )
    physical = env.unwrapped
    images = []
    frame_times = []
    total_reward = 0
    try:
        obs, _ = env.reset(seed=seed, options={"material": material or {}})
        if hasattr(policy, "reset"):
            policy.reset()
        if recorder is not None:
            recorder.start(
                obs,
                timestamp_s=float(physical.data.time),
                inputs=env.input_channels() if isinstance(env, SensorEnv) else None,
            )

        def record_camera():
            if physical.renderer is None:
                from shodo.rendering import Renderer

                physical.renderer = Renderer(physical.model)
            recorder.add_camera_frame(
                physical.renderer.camera_frame(physical), timestamp_s=float(physical.data.time)
            )
            recorder.metadata["camera_calibration"] = physical.renderer.camera_metadata()

        if recorder is not None and camera_every:
            record_camera()
        if frames:
            images.append(Image.fromarray(env.render()))
            frame_times.append(float(physical.data.time))
        while True:
            if hasattr(policy, "observe_camera") and physical.index % policy.camera_every == 0:
                if physical.renderer is None:
                    from shodo.rendering import Renderer

                    physical.renderer = Renderer(physical.model)
                policy.observe_camera(
                    physical.renderer.camera_frame(physical), float(physical.data.time), char
                )
            transition = execute_step(
                env, obs, policy, expert_noise=expert_noise, noise_rng=noise_rng
            )
            obs = transition.next_observation
            terminated, truncated = transition.terminated, transition.truncated
            total_reward += transition.reward
            if recorder is not None:
                privileged = {"history": np.asarray(physical.history[-1])}
                if transition.expert_requested_action is not None:
                    privileged.update(
                        expert_requested_action=transition.expert_requested_action,
                        expert_applied_action=transition.expert_applied_action,
                    )
                recorder.append(
                    transition.requested_action,
                    transition.applied_action,
                    obs,
                    timestamp_s=transition.next_timestamp_s,
                    reward=transition.reward,
                    terminated=terminated,
                    truncated=truncated,
                    privileged=privileged,
                    inputs=env.input_channels() if isinstance(env, SensorEnv) else None,
                )
                if camera_every and (physical.index % camera_every == 0 or terminated or truncated):
                    record_camera()
            if frames and (physical.index % 5 == 0 or terminated or truncated):
                images.append(Image.fromarray(env.render()))
                frame_times.append(float(physical.data.time))
            if terminated or truncated:
                break
        env = physical
        history = np.asarray(env.history)
        errors = np.linalg.norm(np.c_[history[:, 20:22], history[:, 2]] - history[:, 3:6], axis=1)
        down = history[:, 13] >= 0
        contact = down & (history[:, 14] > INK_LOAD_THRESHOLD)
        lifted = history[:, 5] > 0.015
        metrics = {
            "char": char,
            "policy_device": {
                "requested": getattr(policy, "requested_device", "cpu"),
                "resolved": getattr(policy, "device", "cpu"),
            },
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
            "terminated": bool(terminated),
            "expected_steps": len(env.path),
            "sensors": sensors.to_dict() if sensors else None,
            "sensor_contract": sensor_contract(sensors) if sensors else None,
            "checkpoint_provenance": getattr(policy, "checkpoint_provenance", None),
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
        if frames:
            metrics["frame_times_s"] = frame_times
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


def load_ppo(directory="runs", *, device="cpu"):
    from shodo.rl import load_controller

    return load_controller(directory, device=device)


def evaluate(
    output="runs/evaluation-learned.json",
    checkpoint="runs/bc.pt",
    algorithm="learned",
    config=None,
    chars=TEST,
    seed=17,
    device="cpu",
    sensors=None,
    observation_mode=None,
):
    if not chars:
        raise ValueError("Evaluation needs at least one character")
    if algorithm == "learned":
        policy = load_policy(checkpoint, device=device)
    elif algorithm == "ppo":
        policy = load_ppo(Path(checkpoint).parent, device=device)
    elif algorithm in ("expert", "classical", "oracle", "zero"):
        policy = algorithm
    else:
        raise ValueError(f"Unknown policy: {algorithm}")
    results = {}
    if algorithm == "classical":
        if observation_mode == "privileged":
            raise ValueError("Classical baseline requires sensor observations")
        sensors = sensors or SensorConfig()
    if observation_mode == "privileged" and getattr(policy, "sensor_config", None):
        raise ValueError("Sensor checkpoint cannot use privileged observations")
    if (
        observation_mode == "sensor"
        and callable(policy)
        and not getattr(policy, "sensor_config", None)
    ):
        raise ValueError("Privileged checkpoint cannot use sensor observations")
    sensors = sensors or getattr(policy, "sensor_config", None)
    policies = {"expert": "expert", algorithm: policy, "zero": "zero"}
    if sensors:
        policies = {"classical": "classical", algorithm: policy, "oracle": "oracle", "zero": "zero"}
    for name, controller in policies.items():
        results[name] = []
        for char in chars:
            row = rollout(char, controller, config=config, seed=seed, sensors=sensors)[0]
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
        completed = sum(row["terminated"] and not row["truncated"] for row in rows)
        truncated = sum(row["truncated"] for row in rows)
        missing = sum(row["ink_rmse_mm"] is None for row in rows)
        print(
            f"{name}: mean held-out RMSE {np.mean([r['rmse_mm'] for r in rows]):.3f} mm; "
            f"complete {completed}/{len(rows)}, truncated {truncated}, missing ink {missing}"
        )
    print(f"Evaluation report: {path.resolve()}", flush=True)
    return results
