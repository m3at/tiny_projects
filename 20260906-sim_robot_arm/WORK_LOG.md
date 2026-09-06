# Extended realism work

User requested at least four more hours, at most twelve, on 2026-09-06.
First `date` check after instruction: **03:02:08 JST**.
Do not conclude the session before **07:02:08 JST**. Finish by **15:02:08 JST**.
Check `date` periodically. Use the time for substantive engineering and live experiments.

The original v1 source is archived in `runs/baseline-v1-source.zip`; original outputs
remain in `runs/`. New default run directory is `runs/v2`; current checkpoint contract is 3.

Implemented so far: pinned Apache-2.0 Menagerie Panda without hand, inverse-dynamics
torque servos, six-axis pose control, elastic Coulomb bristle bundles coupled to the
arm, conservative water/mobile/fixed-pigment grid, paper texture in offscreen rendering,
pressure/tilt trajectories, 40-observation six-action policy, material randomization.
Five physical/integration tests passed before the latest optimization pass. Initial
Panda teacher 永: 1.177 mm RMSE, 0.026 N force RMSE. First v2 BC trained (28 episodes,
45 epochs) but held-out evaluation uncovered a floating-point endpoint issue in
the new pressure envelope; fixed with nonnegative sine clamping. Re-evaluate.

Remaining work: validate/refine contact mechanics and ink model; expose experiment
configuration; improve controller learning and test multiple seeds/materials; profile
and optimize with accuracy comparisons; run longer PPO experiments; broaden to more
held-out glyphs; add conservation/convergence/robustness reports and recordings;
consolidate README, AGENTS, VALIDATION and source attribution; audit all commands.
Do not claim physical calibration or sim-to-real validity without measured evidence.

## Progress around 03:40 JST

Added native MuJoCo cable backend and material-reference bundle factory. Early raw
stiff-rod parameters were numerically poor (large penetration despite zero warnings);
retain `runs/v2/rod-probe.json` as exploratory evidence, not a passing validation.
`free_hair_bundle()` uses 1,000 wet-hair analogues, derived bundle EI/mass and capsule
mass correction. `experiments/cable.toml` uses this reference at 0.1 ms physics step.
Initial one-stroke native recording is `runs/v2/cable-one.*` (before latest ink batching).
PHYSICS.md contains equations, primary sources and detailed limitations.

Controller now compensates contact offset. On 永, uncompensated ink-center RMSE was
5.368 mm; compensated teacher 1.267 mm. Tool-tip error increases intentionally while
the ink improves. Contract version 3, same 40 features / 6 actions. A 56-episode,
60-epoch BC checkpoint is in runs/v2/bc.pt and matches teacher held-out performance.
PPO now supports frozen-BC residual control, atomic checkpoints, source snapshots,
training locks and resume. Source snapshots protect long runs from later source edits.

Active long jobs at last observation (must poll tool handles; this log is not liveness):
- exec session **93902**: 1,000,000-step residual PPO, seed 7, runs/v2/residual-seed7.
- exec session **53981**: 1,000,000-step plain PPO, seed 7, runs/v2/ppo-seed7.

Latest ink code batches all physics contact samples per controller interval and uses
a physical Gaussian contact footprint, preserving deposited mass; transport stays
10 Hz. Five tests pass after that change. Need rerender updated canonical outputs.
Next: native mass/EI/cantilever/contact convergence checks, expanded glyph/material
evaluation, residual/plain PPO evaluation and multiple seeds, performance reporting,
configuration/CLI consolidation, final documentation and completion audit after 07:02.

## Progress around 04:25 JST

One-million-step seed-7 PPO experiments completed. Four held-out mean tracking RMSE:
teacher 1.224 mm, BC approximately 1.224 mm, plain PPO 1.102 mm, residual PPO 0.889 mm.
Residual ink-center errors are 0.813/0.724/0.678/0.687 mm on 永/水/日/山. Force error
is somewhat higher than teacher; report this tradeoff, not only improved tracking.
New residual recording and 99-check validation passed in runs/v2/residual-seed7.

Mechanics audit now settles rods for 3 s and checks discrete beam theory, preserved
mass, timestep agreement, and unloaded recovery. All checks passed. Rods with 3/6/12/24
segments agree with the discrete beam prediction within 0.23%; refining segments
approaches continuum theory. Short .3 s tests had not settled the finer rods.

Microstep peak-force monitoring uncovered >20 N touchdown spikes in straight native
rods hidden by 50 Hz logs. Completed 36-case curvature/approach/timestep sweep in
runs/v2/contact-sweep.json (includes intentionally failing stress cases). Native preset
now has 1 mm stress-free curvature and 1 mm/s touchdown. Full native 永 completed:
peak .439 N, penetration .0564 mm, ink-center error 2.137 mm, visible coverage 97.27%.
Its narrower trace is visibly different from reduced brush; not a calibrated match.

Nine tests passed after configuration, raster, beam, and minimal-IK checks. IK now
uses only kinematics/CoM stages; 100 random poses/Jacobians matched full forward
dynamics exactly for reduced and native models. Native randomization now modifies
actual geom friction; elastic moduli are not randomized. BC now has locks, atomic
checkpoint writes, source snapshots and cleanup. PPO resume/reload tested (4096→6144).
CLI --chars/config now affect relevant workflows; validation is explicit (not assert).

Ink convergence on 永 at 128/256/512 and requested .02/.05/.1 s transport: deposited
pigment agrees to roundoff, visible area differs <1.6%, coverage 100%. Note transport
cadence is rounded up to a controller boundary (.05 request executes at .06 s).

Currently active jobs (poll tool sessions to establish actual liveness):
- **26217**: residual seed17, 3,000,000 steps, runs/v2/residual-seed17.
- **32822**: residual seed27, 3,000,000 steps, runs/v2/residual-seed27.
- **80646**: 260 glyphs × 3 material seeds × 4 controllers, runs/v2/heldout-sweep.
  Exact source matching initial hashes archived at heldout-sweep/source/shodo.
- **38566**: native held-out audit across dt, bundles/segments and friction,
  runs/v2/native-audit; may take substantial time for 19-bundle/12-segment cases.
- **10483**: make validate default BC, last still rendering; poll once for completion.

Still required: review broad/native/seed results and refine from evidence; native
force/segment convergence conclusions; benchmark and optimization equivalence audit;
packaging/CLI/locking failure tests; final README/VALIDATION/AGENTS/Makefile refresh,
canonical final artifacts and full audit. README is now v2 with explicit audit-in-progress
notice; VALIDATION.md remains v1. Do not conclude before 07:02:08 JST.

## Progress around 05:00 JST

VALIDATION.md now contains v2 completed results and explicitly labels ongoing studies;
README/Makefile expose current commands. Existing .gitignore intentionally keeps
uv.lock local; preserved that preference. setup creates a missing lock and honors an
existing one. Make supports RUN_DIR/CONFIG/POLICY and residual/benchmark/mechanics/figures.
Removed unused src/shodo/arm.xml after verifying byte-exact preservation in baseline zip.
Isolated wheel install successfully imported packaged manifest and built the Panda.
Final wheel must be rebuilt after source cleanup.

**Native transfer validation completed:** runs/v2/native-transfer/validation.json,
107 checks passed for teacher/BC/zero on four held-out glyphs, with a native recording.
The BC was trained only on reduced physics. Native teacher ink errors are
2.137/1.961/1.561/1.655 mm; transferred BC 2.319/2.106/1.646/1.803 mm. All nonzero
controllers have >97% raster coverage, peaks below .452 N, no truncation. This is
numerical transfer between two uncalibrated models, NOT hardware transfer.

**Completed 300-episode material grid:** actual stiffness/friction scaled .5/.8/1/1.2/1.5
while nominal target force stays fixed. Default teacher/BC/residual mean ink errors
1.214/1.217/.758 mm; mean force errors .135/.135/.140 N. No truncation, coverage >=99.79%.
Stronger normal-force feedback exposed a better pressure tradeoff. Added separate
experiments/pressure.toml (gain .02 m/N, variation ±50%, reward force scale .1 N),
leaving defaults unchanged. Pressure BC trained 56 episodes/60 epochs, 35,840 labels.
Its completed 200-episode teacher+BC grid is runs/v2/material-pressure-bc/report.json:
pressure BC mean ink 1.193 mm, mean force error .0480 N, max force error .115 N,
max physical load .799 N. Default BC mean force error was .135 N.

All 6,702 glyph geometries pass exhaustive audit (150.6 s under load), maximum
trajectory length 4,537 steps (驪). Bounds stay inside the paper's nominal margin.
New contact ablation file reproduces ink 5.368→1.267 mm exactly; native optimization
reproduced complete history exactly (native-optimization.json). Ink convergence,
mechanics-summary.svg/png, training-curves and partial heldout-distribution plots exist.
Matplotlib 3.11.1 added to dev only. Ten tests + Ruff/format currently pass.

Provenance now captures source/lock hashes at module import, not after a long run;
snapshot_source verifies its copy matches those hashes. Older already-running jobs
use their initial parent report/snapshot as authoritative execution provenance;
individual old save_rollout JSONs can contain later save-time hashes. Final canonical
recordings should be regenerated after code freezes. Core default behavior has been
kept unchanged (new force/material knobs retain old defaults; IK/alloc optimization exact).

Live sessions at last poll (verify via handles, don't infer liveness from this log):
- **26217**: residual seed17, target 3M; ~1.39M at 05:00.
- **32822**: residual seed27, target 3M; ~1.39M at 05:00.
- **44493**: seed7 continuation to ~3M in runs/v2/residual-seed7-3m; original 1M
  reference preserved. ~1.28M at 05:00. Resumed at 1,001,472 for 2M additional steps.
- **89217**: pressure-focused residual seed7, target 1M, runs/v2/pressure-residual;
  ~330k at 05:00. After completion, evaluate and run material_sweep with policy
  pressure-residual into a separate directory, compare to completed baseline grids.
- **80646**: fixed 260-glyph/3-material-seed/4-policy cohort; 663 teacher episodes
  complete at 05:00, no truncation. Still must finish BC/plain/residual sections.
- **38566**: native audit; 0.2/0.1 ms cases complete; 0.05 ms cases in progress,
  then 7×12,19×6,19×12 bundles/segments and friction. Potentially hours for fine cases.
- **35755**: 48-episode reduced dt/bundle audit, just started around 04:58.

All other listed older sessions (including 94949 native validation, 9763 pressure BC
grid, 24122 dataset audit, 76954 baseline material grid) have completed.

Remaining: review/quantify native discretization and large cohort, 3-seed convergence,
pressure-residual tradeoff, final CPU benchmark without competing jobs, source/metadata
audit and final canonical validation/recordings/wheel/docs. Avoid gratuitous new features;
the long experiments now supply the evidence needed to refine/conclude. Minimum
completion time is still **07:02:08 JST**, maximum **15:02:08 JST**.

## Progress around 05:30 JST

Native fine-segment instability was traced to actual handle/third-segment overlap,
not cured by weakening the physical checks. Explicit collision categories now exclude
handle/bristles while retaining handle/scene and bristle/paper contact. Six-segment
full-history equivalence is exact (`native-root-fix-equivalence.json`); new 6/12-segment
air-contact tests pass. Stopped session 38566 was archived under native-pre-root-fix
with status/reason. Corrected session **92737** tests nominal and 7×12/19×6/19×12 on 一永.
The first 7×12 一 completed cleanly: peak .348 N, penetration .0251 mm, ink 1.264 mm.

Native full-永 integrated force at .2/.1/.05 ms is saved in native-impulse.json.
The .1/.05 ms impulse-vector difference is 0.132%; microstep peak differs 3.59%.
Default reduced 2 ms vs .5 ms mean tracking differs .0191 mm across four glyphs,
with peak .73677 vs .73655 N. Completed 48-case reduced audit is reduced-audit/report.json.

CLI evaluation/demo/validation now reuses checkpoint material/action configuration
at nominal materials unless --config explicitly overrides it. Four routing/error
tests were added; make check passes 16 tests plus Ruff/format. Pressure BC automatic
configuration evaluation passed. Both data fetchers use locks and atomic payload
replacement. Native performance optimizations remain exactly history-equivalent.

At 05:27, fresh seed17/27 were ~2M transitions, resumed seed7 ~1.9M, pressure PPO
~950k. The broad fixed cohort was ~960/3120 episodes (teacher complete, BC running).
All six live handles above remain active, verified via tool polling. Continue broad
cohort to completion even if later than the 07:02 lower bound; maximum is 15:02.

## Progress around 05:36 JST

Artifact audit found that Path.with_suffix truncated decimal experiment identifiers.
Fixed save_rollout to append suffixes, added regression coverage. Aggregate metrics
were intact; regenerated all 48 reduced cases with complete unique artifacts (session
23603 finished). Tracking/ink/force/peak/pigment metrics reproduce the old report
exactly. Previous outputs remain in reduced-pre-filename-fix. Native timestep/friction
regeneration is session **93174**, runs/v2/native-timestep-friction, 20 cases. Existing
refinement 92737 has nondotted remaining names and continues unaffected.
Moved the pre-root-fix MuJoCo warning log into native-pre-root-fix for preservation.

Pressure PPO finished and passed 99 nominal validation checks. Its 100-episode stress
grid finished (26214): mean ink .753746 mm, force .061305 N, worst force .112299 N,
max load .838874 N, coverage >=.99795, no truncation. Baseline residual ink .758322 mm,
force .140411 N. Pressure BC has lower force error (.0480 N) but higher ink error
(1.19260 mm). Summary is material-summary.json, regenerated by make figures.

The pressure residual now runs separately on the SAME fixed 260-glyph/3-material-seed
cohort: **83843**, runs/v2/heldout-pressure. This is an exploratory extension after
seeing the stress-grid result, not a predeclared fifth arm of the original study.
Do not merge it without retaining that distinction. Script now accepts --policies,
captures source snapshot/script hash, and validates count. Original running process
80646 retains its original code/checkpoints and four-policy cohort.

Twenty tests pass, including rotational covariance of brush force/moment and edge/CFL
ink conservation. Core package is provisionally frozen after artifact naming fix.
Final canonical BC/residual/native validation chain and wheel rebuild have started;
poll new handles from current tool context. Keep documenting numerical limitations.

## Progress around 05:54 JST

Wheel rebuilt after source freeze; isolated uv installation imports the cached wheel,
loads the manifest, and constructs nv=7 Panda. No obsolete arm.xml is packaged.
Frozen-source BC and fixed-1M residual each passed 99 checks with exact prior metrics;
their validation report source hashes match current package files.
**98646** is now on native transfer revalidation (teacher four glyphs complete, BC next).

Repaired decimal-name native references: preserved existing 永 files under their full
identifier; regenerated 一 (14390 completed) with exactly identical tracking, ink,
force, penetration, and pigment metrics. All six completed refinement cases have
JSON/NPZ/PNG files. Independent fine-timestep 一 reference also completed (73088):
0.05 ms, ink1.279294 mm, peak .453717 N, penetration .021853 mm, no truncation.
This enables equal-timestep comparisons with refined geometry.

Latest live polling: **26217/32822** ~2.5M; **44493** ~2.38M. **80646** ~1200 original
cohort episodes; **83843** ~144 pressure cohort episodes. **92737** finished 19×6 永
(ink2.003214 mm, peak .397178 N, penetration .026977 mm), now 19×12 cases remain.
**93174** finished all .2/.1 ms native timestep cases; fine dt/friction remain.
All seven long study handles plus 98646 verified active. The plotting script now
also creates separate heldout-pressure summary/plots; keep exploratory label.

Still required: finish these runs, evaluate/validate the three final seed checkpoints,
paired broad-cohort statistics and material identity checks, final uncontended speed
benchmarks, final docs/source/artifact audit and goal completion after 07:02:08 JST.

Source-archive audit note: all 15 hashes recorded by the original cohort match the
archive exactly. Its extra assets/panda_manifest.json matches the current manifest;
the old provenance format only hashed top-level source files, so a naive equality
against the newer recursive 16-file map reports a spurious mismatch. Do not discard
the extra required asset or rewrite historical hashes. New provenance is recursive.

## Progress around 06:26 JST

Frozen-source native transfer revalidation **98646 completed**, all107 checks pass,
505 rendered frames, source hashes match current package. Scene inspected headlessly.
Default BC/fixed1M residual/new native canonical validations are now finalized.

**26217/32822/44493 completed**: fresh17/27 at3,000,320 transitions; resumed7 near3M.
Seed17 validation (99702) and seed27 (50903) both passed99. Nominal means:
seed17 tracking1.082148/ink1.100296 mm, force.053030 N;
seed27 tracking.955403/ink.902422 mm, force.102234 N.
These are worse than fixed1M seed7 on those metrics; do not imply longer training
universally improves accuracy. All three must be reported, with seed7 resume protocol.
New seed7-3M validation just started; poll its handle from current tool context.

Native timestep grid all12 cases complete. Default .1ms vsfine .05ms across4glyphs:
ink error difference<.029mm, impulse vectorrelative<.326%, peakrelative<3.60%.
Penetration .046–.056mm→.023–.027mm. Friction cases in **93174** nearly finished.
**92737** still on19×12 永; its一 completedcleanly peak.341983N,ink1.198535mm.

An evidence-driven faster native variant was tested: 2mm reference curvature,
5mm/s approach. Full永 at.1/.05ms (76497 completed):1324steps/26.48ssim instead
2524/50.48; ink2.306905/2.235084mm, peaks.664183/.650657N, no truncation.
Coverage.96686/.97271, pressureerror~.091N. More timestep sensitivity and higher
load than slowreference; speed/accuracy tradeoff only. Added experiments/cable-fast.toml,
kept cable.toml unchanged. **87745** now full native-fast BC transfer validation.

Added scripts/ik_benchmark.py: equal positions/Jacobians verified before paired
prerequisite-stage timings. Current contended smoke:2.54× reduced/9.10×native
pipeline only, NOT full simulator. Rerun after competing jobs stop. Package source
remains frozen; only scripts/docs/new TOML were changed since artifact-name fix.

Latest broad polling: **80646**1488 original episodes, **83843**432 pressure episodes.
No observed truncation. Final seed validation, broad completed statistics/material
pairing, remaining nativecases, uncontended benchmarks and final docs still required.
Minimumcompletion07:02:08, maximum15:02:08 remainunchanged.

## Progress around 06:45 JST — current handoff state

All original native studies completed: **92737** eight refinement cases, no truncation;
**93174** twenty timestep/friction cases, no truncation; **87745** native-fast transfer,
all107 checks. 19×12 永: ink2.100221mm, peak.369979N, penetration.039761mm. The
fast preset and its speed/accuracy tradeoff are documented; slow reference unchanged.

**68286** seed7-3M validation completed, all99 checks. Mean tracking.814198mm,
ink.838738mm, force.059573N. All longer-run ink scores are worse than fixed1M seed7.
History decomposition on 永 identifies the metric mismatch: drawXY .813→.887mm,
airXY .799→.555mm, airZ .586→.425mm. Better composite tracking can mask worse ink.

Evidence justified ONE training-only extension after provisional freeze: **InkObjective**
in rl.py adds .65*exp(-(loaded inkXY error/.002)^2) during drawing, with no bonus
without contact. Default reward, observations, dynamics and deposition are unchanged.
CLI --ink-objective applies only to PPO; objective metadata is resume-checked, with
legacy missing objective normalized to tracking. All24 tests pass, including paired
state/pigment equivalence and objective routing/resume guards. No more package changes
are currently planned. Wheel rebuilt and isolated installed wrapper reset confirmed
40 observations/6 actions (51181 completed).

**54141**: new matched seed7 residual1M with --ink-objective, runs/v2/ink-residual;
~350k at06:44. After completion: validate nominally; run material_sweep --policies
ink-residual --output runs/v2/material-ink-residual; run heldout_sweep --policies
ink-residual --count256 --output runs/v2/heldout-ink. These are exploratory extensions,
not replacements for the original fixed four-controller study. Scripts support these
names and plotting includes separate ink summaries. New cohorts archive their exact
entrypoint script alongside package snapshot.

**92658**: canonical default BC/fixed1M residual/native reference validation chain
rerunning against latest training/CLI source; currently native teacher four cases done.
The earlier identical-physics native reference already passed107; source provenance
records its earlier training-code version. Native-fast/3M/pressure experimental reports
retain their actual captured source versions; do not relabel historical hashes.

**80646**: original3120-episode cohort, ~1776 completed (teacher780+BC780 done, PPO216).
**83843**: pressure780-episode cohort, ~672 completed. Neither has observed truncation.
All other listed sessions have completed. Remaining long work is these four handles,
then ink objective's evaluation/grid/cohort, final figures/statistics/material pairing,
standalone end-to-end and IK benchmarks after CPU jobs finish, final docs/source audit.
Minimum finish07:02:08 JST; maximum15:02:08. Likely finish after the original and new
cohorts complete, not merely at the lower time bound. Keep checking date.

## Progress around 07:23 JST — lower time bound satisfied

The four-hour minimum passed at07:02:08; continue until the remaining actual work
finishes, within15:02:08 maximum. Source is unchanged since the InkObjective/CLI
extension; all24 tests and packaging checks pass. **92658** latest-source canonical
BC/residual/native validation chain completed (99/99/107); **46554** plainPPO also
completed99 checks. No more model/package changes are planned.

**54141** ink objective1M completed. **54905** nominal99 checks and **81075** material
grid100 completed. Ink nominal means tracking.891380mm, ink.637807mm, force.143573N.
Grid: ink.650561mm, force.163522N, worst force.323697N, coverage>=.99637, no truncation.
It improves ink but sacrifices pressure; never promote it as an unqualified improvement.

One FINAL existing-option combination is running: **63420**, pressure profile plus
--ink-objective, base runs/v2/pressure/bc.pt,1M seed7, runs/v2/pressure-ink-residual.
This completes the four profile/objective combinations; do not add further tuning.
After completion, validate, run material_sweep --policies pressure-ink-residual
--output runs/v2/material-pressure-ink-residual, and (if numerically usable) the same
780-episode cohort: --policies pressure-ink-residual --count256
--output runs/v2/heldout-pressure-ink. Scripts and plots support these names.

Other LIVE handles: **80646** original3120 cohort, now final residual-policy section
(teacher/BC/plain each780 complete); **31470** ink cohort780, ~362 at07:20. **63420**
was~380k at07:20. All other earlier handles have completed.

Pressure cohort **83843 completed780**, no truncation/missing ink. Mean ink.762524mm,
p95.931566, force.040604N, max ink1.124374, max forceRMSE.056328, coverage>=.99789.
Original rows lack actual_brush fields; do not compare a nonexistent key. Replayed
archived/current reset code and verified identical full brush settings and paper
fiber hashes; all780 pressure rows match. Evidence: material-seed-pairing.json.
Three sampled stiffnesses231.008/250.367/237.401N/m are ALL above nominal; only the
separate25-condition material grid covers softer material. Docs state this limitation.

Still required: finish remaining cohorts/combined profile, final full statistics and
pairing/integrity checks, final figures, standalone benchmarks after project CPU jobs
stop (default/quality/native and scripts/ik_benchmark.py), final docs cleanup/source
archive and artifact integrity audit, final make check, then goal completion.

## Progress around 07:45 JST — final combination and provenance checks

Combined pressure/ink training **63420 completed** 1,001,472 transitions. Nominal
validation **86634 passed 99 checks**: tracking 0.895523 mm, ink 0.680782 mm,
force 0.043717 N. Material grid **98784 completed 100**: ink 0.685739 mm,
force 0.062219 N, worst force 0.147839 N, coverage >= 0.997951, peak 0.822897 N,
no truncation/missing ink. Versus pressure-only: 9% better ink, similar mean force,
but worse worst-case force. No further model changes/training are planned.

LIVE: **80646** original cohort ~2844/3120; **31470** ink cohort ~741/780;
**54506** combined cohort ~111/780. The pressure cohort is already complete.
All other sessions completed, including make check **26191** (24 tests) and figures.

All four cohort source archives match every recorded hash; latest canonical BC,
fixed residual, native reference and combined validations match current package
source exactly. Residual frozen BC bytes match the intended default/pressure bases.
Archived/current teacher 永 replay at all three seeds gives exact 21-column histories
and ink image bytes. Only torque history was excluded: original controller-boundary
sampling versus newer microstep peaks. All shared metrics except torque are exact.
Evidence saved in runs/v2/archived-rollout-equivalence.json. Do not claim identical
torque sampling across cohort generations. A failed diagnostic tried treating the
returned PIL image as a Paper; corrected replay completed, no artifact was changed.

Remaining: cohorts, final summaries/bootstrap/material pairing/checkpoint hashes,
standalone default/quality/native and IK benchmarks once CPU jobs finish, final docs,
source archive, artifact integrity evidence, and goal completion. Minimum time met;
maximum remains 15:02:08 JST. Keep headless and check date periodically.

## Final record — 08:22 JST, 2026-09-06

The extended work is complete. The user's additional time window began at 03:02:08;
more than five hours were used, satisfying the four-hour minimum and staying below
the twelve-hour maximum. All training, cohort, validation and benchmark processes
have finished. Earlier live-handle instructions above are historical, not pending work.

Original cohort: 3,120 complete episodes, zero truncation/missing ink. Mean ink error
teacher/BC/plain PPO/residual: 1.216243/1.218236/0.944693/0.743912 mm. Residual improves
38.9% over BC; paired mean improvement 0.474325 mm, glyph-bootstrap 95% interval
[0.469609, 0.478960] mm. This is conditional on the fixed checkpoint/material draws.

Pressure, ink-only and combined extensions each completed 780 episodes without
truncation/missing ink. Combined: ink 0.665459 mm, p95 0.829168 mm, force 0.035371 N,
coverage minimum 1.0, peak 0.788961 N. All four cohorts have exactly their intended
unique episode sets; shared config, source/checkpoint hashes and paired materials
passed the final read-only audit. Total broad episodes 5,460; material-grid episodes
800. Original fixed comparison and exploratory extensions remain clearly separated.

Standalone benchmarks ran sequentially after all project CPU jobs stopped, one warmup
plus three timed 永 episodes each. Median control steps/s: default 1041.384344,
quality 758.181785, native reference 36.031726 (20.8277×/15.1636×/0.720635× simulated
real time). Per-preset physical metrics are identical across repeats. IK prerequisite/
Jacobian-only speedups: reduced 2.69363×, native 7.52779×; not whole-simulator claims.

Final make check: all 24 tests pass, 30 files formatted, lint clean; git diff --check
also clean. Six GIFs decode fully and corresponding 22-column histories are finite.
The wheel matches all 17 current package/manifest files; isolated installation passed.
No current MuJoCo warning log was present. Package source stayed frozen after the
training-only ink objective/CLI extension; canonical BC/residual/native/combined
validation source hashes match the final package. Physics remains uncalibrated.

Final reports/figures are generated. Evidence is in runs/v2/final-artifact-audit.json;
README.md, PHYSICS.md and VALIDATION.md contain reproduction instructions, primary
sources and limitations. runs/v2/final-source.zip is the final source/lock archive,
excluding downloaded data, environments and training artifacts. Old v1 source and
recordings are preserved. No commit was made and no unrelated workspace was changed.
