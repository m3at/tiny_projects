# SmolVLA LoRA development

This workflow now uses the B601-RS with a rigid vertical brush. Its new action contract and reserved joint slot reject earlier Panda datasets/adapters. The optional neural integration has regression coverage, but no B601 SmolVLA training or hardware qualification is claimed.

This optional integration adapts pretrained SmolVLA to Shodo's sensor inputs, raw camera view, and six Cartesian command increments. It uses LeRobot's SmolVLA implementation and PEFT LoRA, with a local Shodo dataset reader rather than LeRobotDataset or Hub publication. The default 100 updates are a development smoke run, not a policy-quality training budget. See [VALIDATION.md](../../VALIDATION.md) for measured results and limitations.

## Setup and first run

Run these commands from the repository root:

```sh
make setup
make data
make smolvla-setup

# One privileged-teacher demonstration per training glyph; actor inputs remain sensor-only.
make record POLICY=oracle RUN_DIR=runs/smolvla-recordings \
  CHARS=一二三十木大人 EPISODES=7 CAMERA_EVERY=5

make smolvla-prepare
make smolvla-train
make smolvla-verify
make smolvla-evaluate CHARS=永
```

`smolvla-setup` installs Python 3.13 dependencies into `integrations/smolvla/.venv` using the checked-in [uv.lock](uv.lock). It does not replace the root environment or add LeRobot to ordinary BC/PPO workflows. The integration pins `lerobot[smolvla,peft]==0.6.1`; all transitive versions are locked. Model loading downloads the pinned pretrained policy into the local Hugging Face cache on first use. Recording and preparation do not need pretrained weights. Everything runs headlessly; this workflow does not require FFmpeg.

Use new recording, prepared-data, and adapter destinations for new experiments: publication refuses overwrites. Verification reads the adapter; evaluation writes `evaluation.json` inside its directory by default, replacing that report if rerun. Supply `--report` with distinct experiment paths to retain comparisons. The seven demonstrations and 100 updates exercise the pipeline, not a recommended training budget or sufficient test of generalization. Default held-out glyphs 永水日山 are rejected during preparation, and evaluation rejects characters present in the adapter's training manifest unless explicitly marked as a development rollout with `--allow-training-chars`.

## Recovery demonstrations

Clean demonstrations cover a narrow state distribution. Oracle recovery recording perturbs executed commands while retaining separate, unperturbed expert targets at each decision. This supplies examples of corrections from disturbed states without giving privileged observations to the actor:

```sh
make record POLICY=oracle RUN_DIR=runs/smolvla-recovery-recordings \
  CHARS=一二三十木大人 EPISODES=28 CAMERA_EVERY=5 EXPERT_NOISE=0.08
make smolvla-prepare VLA_EPISODES=runs/smolvla-recovery-recordings/episodes \
  VLA_DATASET=runs/smolvla-recovery-compact-data VLA_ARGS="--supervision expert --image-size 128"

# Development budget only; choose duration from independent evaluation, not minibatch loss.
make smolvla-train VLA_DATASET=runs/smolvla-recovery-compact-data \
  VLA_OUTPUT=runs/smolvla-recovery-adapter VLA_STEPS=3000 \
  VLA_ARGS="--chunk-size 1 --rank 32 --alpha 64 --batch-size 8"
make smolvla-verify VLA_OUTPUT=runs/smolvla-recovery-adapter
make smolvla-evaluate VLA_OUTPUT=runs/smolvla-recovery-adapter CHARS=永水日山 \
  VLA_ARGS="--execute-steps 1 --report runs/smolvla-recovery-heldout.json"
```

`EXPERT_NOISE` is the Gaussian standard deviation in normalized command increments, using an independent, recorded per-episode RNG seed. It requires `POLICY=oracle`. Omit it for ordinary recordings; explicitly set `EXPERT_NOISE=0` to collect clean expert labels. The default configuration remains nominal: this noise perturbs behavior, not material parameters or sensor calibration.

Native `requested_actions` and `applied_actions` remain the actual noisy behavior. Separate `privileged_expert_requested_action` labels contain the unperturbed oracle command; `privileged_expert_applied_action` contains its counterfactual workspace-clipped increment from the same pre-action command. The latter is the target selected by `--supervision expert`; it is not executed motion. Successive labels follow disturbed behavior and do not form a coherent future oracle rollout, so expert-target datasets require `--chunk-size 1` and `--execute-steps 1`. Preparation rejects missing or incompatible expert-label metadata. Default `--supervision applied` continues using actual effective behavior commands.

## Settings and artifacts

SmolVLA has its own `VLA_DEVICE=auto` default: CUDA, then Apple Metal, then CPU according to availability. This is independent of the root `DEVICE=cpu` default for small BC/PPO policies; availability is not evidence of a speedup. An unavailable explicitly requested device fails. MuJoCo, sensing, and ink transport remain on CPU.

```sh
# Separate experiment, with explicit device and training settings.
make smolvla-train VLA_DEVICE=mps VLA_STEPS=200 \
  VLA_OUTPUT=runs/smolvla-experiment \
  VLA_ARGS="--rank 8 --alpha 16 --batch-size 1 --chunk-size 16 --learning-rate 0.001"
make smolvla-verify VLA_OUTPUT=runs/smolvla-experiment VLA_DEVICE=mps
make smolvla-evaluate VLA_OUTPUT=runs/smolvla-experiment CHARS=永水日山 \
  VLA_ARGS="--execute-steps 4 --denoise-steps 10"

# All CLI options; paths remain relative to the repository root.
uv run --project integrations/smolvla --locked python -m shodo.smolvla --help
```

`VLA_EPISODES` defaults to `runs/smolvla-recordings/episodes`, `VLA_DATASET` to `runs/smolvla-data`, and `VLA_OUTPUT` to `runs/smolvla-adapter`. `VLA_STEPS` defaults to 100 and `SEED` to 7. `VLA_ARGS` forwards additional command-specific options; for example, `make smolvla-prepare VLA_ARGS="--image-size 256"`. Default training uses rank 8, alpha 16, batch size 1, a 16-action chunk, and learning rate 0.001. Only LoRA tensors train; all pretrained base weights remain frozen. LoRA reduces trainable parameters and optimizer state, but still requires the base model and its forward/backward computation.

Prepared datasets contain read-only memory-mapped `.npy` arrays and a manifest with contracts, normalization statistics, source episode hashes, provenance, attribution, and camera calibration. Training saves `adapter/`, `training.json`, and `reload-probe.npz`: adapter weights/configuration, full settings and losses/timings, and a fixed-noise reload probe. It saves adapters only, not a standalone full model, optimizer state, or RNG state; training resume is not implemented. Keep the manifest and pinned base available with the adapter. `verify` reloads the base plus adapter and checks the saved action probe; it is a serialization check, not held-out validation. The reported before/after fixed-batch losses use a training minibatch, not an independent test set. No command uploads data, adapters, or metrics to the Hub.

## Input and action contract

The actor uses one raw perspective camera, a task string such as `Draw 一 on paper following the supplied stroke reference.`, and 32 state features. State selects indices `0:25` and `32:39` from the latest 39-feature `sensor-history` slice: measured pose/reference/command errors, preview, six joint positions and one reserved zero, force proxy, authored target force and drawing flag, sample age, and freshness. It omits joint velocities and earlier history slices. No contact-center, bristle-deflection, contact-fraction, or privileged ink arrays enter model inputs. The force proxy remains synthetic and uncalibrated; supplied references mean this is path-conditioned control, not autonomous stroke planning.

At every action decision, preparation selects the latest camera frame whose acquisition timestamp is at or before that decision. It never borrows a future frame. Recording every five 50 Hz control steps gives a 10 Hz camera stream; intervening decisions reuse the last acquired image. Frames have no diagnostic overlays or paper inset. They are bilinearly resized with preserved aspect ratio, centered on a black 256×256 canvas by default, and converted from uint8 RGB to float32 RGB divided by 255. The same preprocessing is used during inference.

Default targets are the six stored `applied_actions`: effective clipped normalized command increments, not requested pre-clipping actions, joint commands, torques, or measured robot displacement. Expert supervision instead selects the explicitly named counterfactual labels described above. Physical scales and the world-frame additive rotation-vector convention come from the episode's action contract. State/action statistics use only prepared training transitions, and the adapter uses those same statistics at inference. Prepared manifest version 2 gives state features with population standard deviation below `1e-6` a scale of `1.0`; other state scales use their population standard deviation. This prevents nominally constant channels such as sample age and freshness from being amplified by a tiny divisor. Action scales retain a minimum standard deviation of `1e-6`. Version 1 artifacts retain their recorded normalization unchanged; re-prepare and retrain to use the new rule. Action chunks never cross episode boundaries; padded suffixes carry a loss mask. Privileged teacher targets are permitted, but privileged diagnostics never enter model inputs. Preserve the recorded KanjiVG attribution and CC BY-SA 3.0 obligations when distributing derived datasets.

## Execution limits

Evaluation uses synchronous receding-horizon simulation: predict a chunk, execute its first four actions by default, then predict again. Camera acquisition follows the recorded control-step cadence. Inference blocks simulation advancement and reports wall-clock timings; this is not a real-time 50 Hz hardware controller, asynchronous inference implementation, or safety-qualified deployment. Changing execution horizon or denoising steps changes the experiment. A finite loss, a successful adapter reload, or a completed rollout does not demonstrate good brush control; compare ink accuracy, force, missing ink, and failures with the existing classical/oracle/sensor-policy baselines.

`--optimize-inference` opts into safely merging LoRA into the in-memory model, caching prompt tokenization, and caching image embeddings for the same acquired frame. It does not rewrite the saved adapter or reuse stale state/action predictions. Every new camera acquisition invalidates the image cache, and task or tokenizer-option changes invalidate the prompt cache. The state-dependent VLM prefix is always recomputed. These optimizations still need measurement on the target device. Fewer denoising steps or longer execution horizons are separate quality/latency tradeoffs, not equivalent computations.

```sh
# Explicit training-character diagnostic, kept separate from held-out reports.
make smolvla-evaluate VLA_OUTPUT=runs/smolvla-recovery-adapter CHARS=一 \
  VLA_ARGS="--execute-steps 1 --optimize-inference --allow-training-chars --report runs/smolvla-recovery-development.json"

# Replay audit compares baseline and optimized inference over recorded causal inputs.
uv run --project integrations/smolvla --locked python -m shodo.vla_benchmark \
  --adapter runs/smolvla-recovery-adapter \
  --episode runs/smolvla-recovery-recordings/episodes/episode-000000.npz \
  --output runs/smolvla-recovery-latency.json --device mps \
  --samples 20 --warmup 2 --stride 1 --denoise-steps 1 2 4 10
uv run --project integrations/smolvla --locked python -m shodo.vla_benchmark --help
```

The replay audit includes raw RGB preprocessing, state normalization, tokenization/cache lookup, device transfers, prediction, output transfer, denormalization and clipping. It excludes model loading, physics and camera acquisition, and is neither a closed-loop drawing test nor proof of meeting hardware deadlines. Keep warmup and steady-state scope explicit, and use distinct report paths for each device or adapter. Training-character development reports are explicitly marked and must not be presented as held-out results.

Image sizes used by SmolVLA must be positive multiples of 64. The compact recovery workflow uses 128×128 images; this changes visual information and requires a matching prepared dataset and adapter, rather than silently resizing an existing policy at deployment. `--trim-language-padding` is an independent inference option that removes only globally masked trailing language tokens. Cache reuse is exact in the recorded tests, but LoRA merging rounds FP32 adapter updates into some BF16 base weights, and padding removal changes numerical kernel shapes; both produce small nonzero action differences. Keep these options and denoising settings in evaluation reports.

For additional updates, `--warm-start runs/smolvla-recovery-adapter` reuses checked LoRA weights with a fresh optimizer and RNG. Rank, alpha, input/action contracts and image preprocessing must match; the new dataset must retain all parent training characters. Normalization may change, and the parent hashes/settings and that change are recorded. This is fine-tuning continuation, not exact optimizer-state resume. Always use a new output directory.

Independent episode scoring excludes recordings seen in every warm-start ancestor, not only the current training stage. New adapters retain the full training-episode hash union. Legacy continuations resolve their recorded parent manifests and verify checksums; keep those manifests available. Missing or unverifiable ancestry fails before model loading rather than silently claiming a clean split.

## Deterministic action-regression experiment

`--objective action_regression` is an explicit alternative to the default `flow_matching` objective. It retains the same pretrained SmolVLA model and LoRA modules, with all base parameters frozen. The pinned forward constructs `x_t = t * noise + (1 - t) * action` and learns velocity `noise - action`. Setting `t=1` and `noise=0` makes the action-token input identically zero regardless of the target, while the loss target becomes `-action`. One Euler step from zero then predicts an action directly. Targets remain supervision only; they are not supplied through the model's input prefix or suffix. This is deterministic action regression through the existing expert, not standard flow-matching training or a separate replacement policy.

```sh
# Continue the matching rank-32, alpha-64, 128px flow adapter with a new objective.
make smolvla-train VLA_DATASET=runs/smolvla-recovery-compact-data \
  VLA_OUTPUT=runs/smolvla-recovery-regression VLA_STEPS=1500 \
  VLA_ARGS="--warm-start runs/smolvla-recovery-adapter --objective action_regression --rank 32 --alpha 64 --chunk-size 1 --batch-size 8 --learning-rate 0.0003"
make smolvla-verify VLA_OUTPUT=runs/smolvla-recovery-regression

# Training-glyph development rollout, not held-out qualification.
make smolvla-evaluate VLA_OUTPUT=runs/smolvla-recovery-regression CHARS=一 \
  VLA_ARGS="--execute-steps 1 --allow-training-chars --optimize-inference --trim-language-padding --cache-static-inputs --report runs/smolvla-recovery-regression/development.json"
```

Action-regression inference always uses zero noise and exactly one denoising step. Omit `--denoise-steps` to select the adapter's objective-aware default: one for action regression, ten for flow matching. Explicit incompatible counts for regression are rejected. Reload verification, closed-loop evaluation, recorded-state scoring, and replay benchmarking share this sampler contract; regression benchmarks must not use the flow model's `1 2 4 10` sweep. Older artifacts without an objective field retain flow-matching semantics. Training records the objective, parent objective and hashes, and a zero-noise reload probe; warm-starting still creates a fresh optimizer and RNG. This recipe specifies a controlled development experiment, not a validated quality or latency improvement.

`--cache-static-inputs` requires `--optimize-inference`. It reuses unchanged image/language input embeddings for the current acquired frame and task, while recomputing the measured-state embedding and all state-dependent VLM processing at every decision. It does not cache a previous action, state token or state-conditioned attention cache. Reset and new acquisition/task keys invalidate reuse. `--trim-language-padding` remains a separate numerical-shape optimization; neither option changes the regression sampler or grants real-time guarantees.

## Independent episode diagnostics

Separate recording seeds on training glyphs provide development data without repeatedly tuning against held-out characters. For example, record seven recovery episodes with `SEED=107` in `runs/smolvla-recovery-validation-recordings`, then prepare expert targets at 128 pixels in `runs/smolvla-recovery-validation-data`. Score with:

```sh
uv run --project integrations/smolvla --locked python -m shodo.vla_evaluation \
  --adapter runs/smolvla-recovery-regression --dataset runs/smolvla-recovery-validation-data \
  --output runs/smolvla-recovery-regression/action-diagnostic.json \
  --device mps --samples 256 --batch-size 8
```

This diagnostic uses the adapter's training normalization and compares its first predicted action with recorded targets. It rejects reused training episodes. Recorded-state action error is not closed-loop drawing error, and new seeds on training glyphs are not held-out character generalization. Closed-loop reports separately count 20 ms actor deadline misses and action-horizon throughput misses; fitting within a chunk horizon does not demonstrate a synchronous 50 Hz policy.

## Headless policy demo

```sh
make smolvla-demo VLA_OUTPUT=runs/smolvla-recovery-regression CHARS=永 \
  VLA_ARGS="--optimize-inference --trim-language-padding --cache-static-inputs"
```

This exports actual executed motion to H.264 MP4, with final paper/scene PNGs, named trajectory NPZ and full metrics JSON. It prints compact quality summaries and artifact paths. FFmpeg is required, and existing demo outputs are protected; set `VLA_DEMO_OUTPUT` to another destination when needed. Execution defaults to the smaller of four actions and the trained chunk size, with objective-aware denoising. Video timing follows simulation time, so smooth playback does not demonstrate real-time inference. B601 recordings and adapters must be generated afresh; earlier Panda results are not carried forward. Current qualification scope is recorded in [VALIDATION.md](../../VALIDATION.md).

## Pinned models and primary sources

The loader strictly restores the complete [SmolVLA base policy](https://huggingface.co/lerobot/smolvla_base/tree/c83c3163b8ca9b7e67c509fffd9121e66cb96205) at revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`. Its VLM configuration/tokenizer comes from [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467), revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; separate VLM weights are not downloaded because the policy checkpoint already contains them. Model revisions and adapter hashes are checked on reload.

See the official [SmolVLA documentation](https://huggingface.co/docs/lerobot/main/en/smolvla) and [LeRobot PEFT guide](https://huggingface.co/docs/lerobot/main/en/peft_training) for the upstream model and LoRA mechanisms. These rolling guides may differ from the locked release. This repository's explicit preprocessing, dataset schema, and synchronous simulator adapter are local integration choices, not upstream deployment guarantees.
