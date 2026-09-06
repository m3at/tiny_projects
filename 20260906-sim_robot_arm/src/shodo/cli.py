import argparse
import json
from dataclasses import replace
from pathlib import Path

from shodo.artifacts import save_rollout
from shodo.config import config_from_dict, load_config
from shodo.data import TEST, TRAIN, fetch
from shodo.learning import evaluate, load_policy, load_ppo, rollout, train
from shodo.robot import fetch_robot


def main():
    parser = argparse.ArgumentParser(description="Headless shodo arm experiments")
    parser.add_argument(
        "command", choices=["data", "train", "evaluate", "demo", "ppo", "validate", "benchmark"]
    )
    parser.add_argument(
        "--chars", help="Training, evaluation or demo characters (command-specific defaults)"
    )
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--policy", default="learned", choices=["learned", "expert", "zero", "ppo"])
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v2"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--episodes", type=int, default=28)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ink-objective", action="store_true", help="PPO: extra loaded-ink accuracy reward"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.chars is not None and not args.chars:
        parser.error("--chars must not be empty")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.command != "ppo" and (
        args.residual or args.base_policy or args.resume or args.ink_objective
    ):
        parser.error("residual/base-policy/resume/ink-objective options apply only to ppo")
    try:
        config = load_config(args.config) if args.config else None
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"Invalid experiment configuration: {error}")
    if args.steps <= 0 or args.epochs <= 0 or args.episodes <= 0:
        parser.error("steps, epochs and episodes must be positive")
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "bc.pt"
    if (
        config is None
        and args.command in ("evaluate", "demo", "validate")
        and args.policy in ("learned", "ppo")
    ):
        metadata = directory / ("bc.json" if args.policy == "learned" else "ppo.json")
        if metadata.exists():
            try:
                values = json.loads(metadata.read_text())["config"]
                config = replace(config_from_dict(values), randomize=False, record=False)
            except (OSError, ValueError, TypeError, KeyError) as error:
                parser.error(f"Invalid checkpoint configuration metadata: {error}")
            print(
                "Using checkpoint configuration with nominal materials; --config overrides it.",
                flush=True,
            )
    if args.command == "data":
        fetch()
        fetch_robot()
    elif args.command == "train":
        train(
            episodes=args.episodes,
            epochs=args.epochs,
            seed=args.seed,
            output=checkpoint,
            config=config,
            chars=args.chars or TRAIN,
        )
    elif args.command == "evaluate":
        filename = (
            "evaluation.json" if args.policy == "learned" else f"evaluation-{args.policy}.json"
        )
        evaluate(
            directory / filename,
            checkpoint,
            algorithm=args.policy,
            config=config,
            chars=args.chars or TEST,
            seed=args.seed,
        )
    elif args.command == "ppo":
        from shodo.rl import train_ppo

        base = args.base_policy or checkpoint if args.residual else None
        if args.base_policy and not args.residual:
            parser.error("--base-policy requires --residual")
        train_ppo(
            directory,
            steps=args.steps,
            seed=args.seed,
            config=config,
            base_checkpoint=base,
            resume=args.resume,
            chars=args.chars or TRAIN,
            objective="ink" if args.ink_objective else "tracking",
        )
    elif args.command == "demo":
        policy = load_policy(checkpoint) if args.policy == "learned" else args.policy
        if args.policy == "ppo":
            policy = load_ppo(directory)
        for char in args.chars or "永":
            result = rollout(char, policy, seed=args.seed, frames=True, config=config)
            save_rollout(directory / f"{ord(char):05x}-{args.policy}", result)
            print(json.dumps(result[0], ensure_ascii=False))
    elif args.command == "validate":
        from shodo.validation import validate

        validate(
            directory,
            config=config,
            chars=args.chars or TEST,
            seed=args.seed,
            algorithm=args.policy,
            episodes=args.episodes,
            epochs=args.epochs,
        )
    elif args.command == "benchmark":
        from shodo.benchmark import benchmark

        benchmark(
            directory / "benchmark.json",
            config=config,
            char=args.chars or "永",
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
