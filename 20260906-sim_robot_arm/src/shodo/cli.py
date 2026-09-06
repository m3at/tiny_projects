import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from shodo.artifacts import save_rollout
from shodo.config import config_from_dict, load_config
from shodo.contracts import SensorConfig
from shodo.data import TEST, TRAIN, fetch
from shodo.device import resolve_device
from shodo.learning import evaluate, load_policy, load_ppo, rollout, train
from shodo.robot import fetch_robot
from shodo.video import require_ffmpeg


def main():
    parser = argparse.ArgumentParser(description="Headless shodo arm experiments")
    parser.add_argument(
        "command",
        choices=[
            "data",
            "train",
            "evaluate",
            "demo",
            "ppo",
            "validate",
            "benchmark",
            "record",
            "robustness",
        ],
    )
    parser.add_argument(
        "--chars", help="Training, evaluation or demo characters (command-specific defaults)"
    )
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument(
        "--policy",
        default="learned",
        choices=["learned", "expert", "classical", "oracle", "zero", "ppo"],
    )
    parser.add_argument(
        "--observation",
        choices=["privileged", "sensor"],
        help="Policy input contract; learned checkpoints select their own by default",
    )
    parser.add_argument(
        "--sensor-config", type=Path, help="JSON SensorConfig (implies sensor observations)"
    )
    parser.add_argument(
        "--camera-every",
        type=int,
        default=0,
        help="record: raw camera every N control steps, 0 disables",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="robustness: paired seeds (default 7 17 27)"
    )
    parser.add_argument(
        "--expert-noise",
        type=float,
        help="record with oracle: Gaussian behavior noise and separate pre-action expert labels (0 enables clean labels)",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Neural controller device; auto prefers CUDA, then MPS, then CPU",
    )
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
    if args.expert_noise is not None and (
        args.command != "record"
        or args.policy != "oracle"
        or not math.isfinite(args.expert_noise)
        or args.expert_noise < 0
    ):
        parser.error(
            "--expert-noise requires record --policy oracle and a finite nonnegative value"
        )
    if args.camera_every < 0 or (args.camera_every and args.command != "record"):
        parser.error("--camera-every must be nonnegative and applies only to record")
    if args.seeds is not None and args.command != "robustness":
        parser.error("--seeds applies only to robustness")
    if args.seeds is not None and (
        any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("--seeds must be unique nonnegative integers")
    if args.observation == "privileged" and args.policy == "classical":
        parser.error("Classical baseline requires sensor observations")
    if args.observation == "privileged" and (
        args.sensor_config or args.command in ("record", "robustness")
    ):
        parser.error("record, robustness and sensor-config require sensor observations")
    sensors = None
    if (
        args.observation == "sensor"
        or args.sensor_config
        or args.command in ("record", "robustness")
    ):
        try:
            values = json.loads(args.sensor_config.read_text()) if args.sensor_config else {}
            sensors = SensorConfig(**values)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"Invalid sensor configuration: {error}")
    if sensors and args.command in ("data", "validate", "benchmark"):
        parser.error(
            "Sensor policies use evaluate/robustness; canonical validate remains privileged"
        )
    if args.policy in ("classical", "oracle") and args.command not in (
        "evaluate",
        "demo",
        "record",
        "robustness",
    ):
        parser.error("classical/oracle policies apply to evaluate, demo, record and robustness")
    if args.policy == "classical" and not sensors:
        sensors = SensorConfig()
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
    if args.command == "benchmark" and (args.repeats < 1 or len(args.chars or "永") != 1):
        parser.error("benchmark requires positive --repeats and exactly one character")
    if args.base_policy and not args.residual:
        parser.error("--base-policy requires --residual")
    if args.command in ("demo", "validate"):
        try:
            require_ffmpeg()
        except RuntimeError as error:
            parser.error(str(error))
    if args.command in ("train", "ppo") or (
        args.command in ("evaluate", "demo", "validate", "record", "robustness")
        and args.policy in ("learned", "ppo")
    ):
        try:
            resolved = resolve_device(args.device)
        except ValueError as error:
            parser.error(str(error))
        print(f"Neural controller device: {args.device} -> {resolved}; physics remains on CPU.")
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "bc.pt"
    if (
        config is None
        and args.command in ("evaluate", "demo", "validate", "record", "robustness")
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
    controller = None
    # validate may train a missing legacy BC checkpoint; leave that workflow intact.
    if (
        args.command in ("demo", "record", "robustness", "validate")
        and args.policy in ("learned", "ppo")
        and (args.command != "validate" or checkpoint.exists() or args.policy == "ppo")
    ):
        try:
            controller = (
                load_policy(checkpoint, device=args.device)
                if args.policy == "learned"
                else load_ppo(directory, device=args.device)
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            parser.error(f"Cannot load checkpoint: {error}")
        trained = getattr(controller, "sensor_config", None)
        if args.observation == "privileged" and trained:
            parser.error("Sensor checkpoint cannot use privileged observations")
        if args.command == "validate" and trained:
            parser.error("Sensor checkpoints use evaluate/robustness, not canonical validate")
        if sensors and not trained:
            parser.error(
                "Privileged checkpoint cannot use sensor observations; train a sensor policy"
            )
        if trained and not args.sensor_config:
            sensors = trained
        if sensors and trained and sensors.history != trained.history:
            parser.error("Sensor history differs from the checkpoint contract")
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
            device=args.device,
            sensors=sensors,
        )
    elif args.command == "evaluate":
        try:
            evaluate(
                directory / f"evaluation-{args.policy}.json",
                checkpoint,
                algorithm=args.policy,
                observation_mode=args.observation,
                config=config,
                chars=args.chars or TEST,
                seed=args.seed,
                device=args.device,
                sensors=sensors
                if args.sensor_config or args.policy not in ("learned", "ppo")
                else None,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            parser.error(f"Evaluation failed: {error}")
    elif args.command == "ppo":
        from shodo.rl import train_ppo

        base = (args.base_policy or checkpoint) if args.residual else None
        train_ppo(
            directory,
            steps=args.steps,
            seed=args.seed,
            config=config,
            base_checkpoint=base,
            resume=args.resume,
            chars=args.chars or TRAIN,
            objective="ink" if args.ink_objective else "tracking",
            device=args.device,
            sensors=sensors,
        )
    elif args.command == "demo":
        policy = controller or args.policy
        chars = args.chars or "永"
        for index, char in enumerate(chars, 1):
            start = time.perf_counter()
            print(f"[{index}/{len(chars)}] Drawing {char} ({args.policy})…", flush=True)
            result = rollout(
                char, policy, seed=args.seed, frames=True, config=config, sensors=sensors
            )
            base = directory / f"{ord(char):05x}-{args.policy}"
            print("  Encoding MP4…", flush=True)
            try:
                save_rollout(base, result)
            except (OSError, RuntimeError, ValueError) as error:
                parser.exit(1, f"Demo export failed: {error}\n")
            metrics = result[0]
            ink = metrics["ink_rmse_mm"]
            ink_label = "no loaded ink" if ink is None else f"ink error {ink:.3f} mm"
            status = "TRUNCATED" if metrics["truncated"] else "Complete"
            simulated = (
                metrics["steps"] * metrics["config"]["timestep"] * metrics["config"]["substeps"]
            )
            video = Path(f"{base}.mp4").resolve()
            print(
                f"  {status} · {simulated:.2f}s simulated · {ink_label} · "
                f"peak force {metrics['max_force_n']:.3f} N · "
                f"{time.perf_counter() - start:.1f}s elapsed\n"
                f"  Video: {video} ({video.stat().st_size / 1024**2:.2f} MiB)\n"
                f"  Ink:   {Path(f'{base}.png').resolve()}\n"
                f"  Metrics: {base}.json\n"
                f"  Motion:  {base}.npz",
                flush=True,
            )
    elif args.command == "record":
        from shodo.runtime import record_episodes

        try:
            record_episodes(
                directory / "episodes",
                chars=args.chars or TRAIN,
                episodes=args.episodes,
                seed=args.seed,
                policy=controller or args.policy,
                config=config,
                sensors=sensors,
                camera_every=args.camera_every,
                expert_noise=args.expert_noise,
            )
        except (OSError, ValueError) as error:
            parser.exit(1, f"Recording failed: {error}\n")
    elif args.command == "robustness":
        from shodo.robustness import evaluate_robustness

        evaluate_robustness(
            directory / "robustness.json",
            chars=args.chars or TEST,
            seeds=tuple(args.seeds) if args.seeds else (7, 17, 27),
            config=config,
            sensors=sensors,
            policy=controller,
        )
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
            device=args.device,
        )
    elif args.command == "benchmark":
        from shodo.benchmark import benchmark

        benchmark(
            directory / "benchmark.json",
            config=config,
            char=args.chars or "永",
            repeats=args.repeats,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
