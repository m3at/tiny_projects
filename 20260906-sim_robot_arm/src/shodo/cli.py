import argparse
import json
import time
from pathlib import Path

import numpy as np

from shodo.artifacts import provenance, save_rollout
from shodo.data import fetch
from shodo.env import ShodoEnv
from shodo.learning import evaluate, load_policy, load_ppo, rollout, train
from shodo.robot import fetch_robot


def main():
    parser = argparse.ArgumentParser(description="Headless shodo arm experiments")
    parser.add_argument("command", choices=["data", "train", "evaluate", "demo", "ppo", "validate"])
    parser.add_argument("--chars", default="永")
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--policy", default="learned", choices=["learned", "expert", "zero", "ppo"])
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v2"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--episodes", type=int, default=28)
    args = parser.parse_args()
    if args.steps <= 0 or args.epochs <= 0 or args.episodes <= 0:
        parser.error("steps, epochs and episodes must be positive")
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "bc.pt"
    if args.command == "data":
        fetch()
        fetch_robot()
    elif args.command == "train":
        train(episodes=args.episodes, epochs=args.epochs, seed=args.seed, output=checkpoint)
    elif args.command == "evaluate":
        filename = (
            "evaluation.json" if args.policy == "learned" else f"evaluation-{args.policy}.json"
        )
        evaluate(directory / filename, checkpoint, algorithm=args.policy)
    elif args.command == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(ShodoEnv(), str(directory / "ppo-monitor.csv"))
        try:
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                device="cpu",
                verbose=1,
                n_steps=1024,
                batch_size=256,
                policy_kwargs={"net_arch": [64, 64]},
            )
            model.learn(total_timesteps=args.steps)
            model.save(directory / "ppo")
            (directory / "ppo.json").write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "requested_steps": args.steps,
                        "actual_steps": model.num_timesteps,
                        "provenance": provenance(),
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            env.close()
    elif args.command == "demo":
        policy = load_policy(checkpoint) if args.policy == "learned" else args.policy
        if args.policy == "ppo":
            policy = load_ppo(directory)
        for char in args.chars:
            result = rollout(char, policy, seed=args.seed, frames=True)
            save_rollout(directory / f"{ord(char):05x}-{args.policy}", result)
            print(json.dumps(result[0], ensure_ascii=False))
    elif args.command == "validate":
        from gymnasium.utils.env_checker import check_env

        start = time.perf_counter()
        (directory / "validation.json").write_text('{"passed": false, "status": "running"}\n')
        env = ShodoEnv(chars="一")
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
        if not checkpoint.exists():
            train(episodes=args.episodes, epochs=args.epochs, seed=args.seed, output=checkpoint)
        results = evaluate(directory / "evaluation.json", checkpoint)
        for name in ("expert", "learned"):
            for row in results[name]:
                assert not row["truncated"], row
                assert row["rmse_mm"] < 4, row
                assert row["draw_contact_fraction"] > 0.95, row
                assert row["lift_clear_fraction"] > 0.98, row
        learned = np.mean([r["rmse_mm"] for r in results["learned"]])
        zero = np.mean([r["rmse_mm"] for r in results["zero"]])
        assert learned < zero * 0.2
        result = rollout("永", load_policy(checkpoint), frames=True)
        frames = result[3]
        assert frames and np.asarray(frames[-1]).std() > 10
        save_rollout(directory / "validation-rollout", result)
        (directory / "validation.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "seconds": time.perf_counter() - start,
                    "held_out": results,
                    "render_frames": len(frames),
                    "provenance": provenance(),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "Validation passed: API, held-out tracking, brush contact, lifts, baseline comparison, rendering"
        )


if __name__ == "__main__":
    main()
