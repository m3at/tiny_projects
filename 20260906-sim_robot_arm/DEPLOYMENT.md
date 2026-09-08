# B601-RS physical deployment preparation

This setup targets the **48 V RobStride reBot Arm B601-RS**, with a directly attached brush and no motorized gripper. It uses a public description pending the URDF for the delivered arm. No code in this project connects to or enables motors. Calibration and hardware bring-up remain separate from simulation validation.

## Public evidence and unresolved differences

Sources were inspected on 2026-09-08. The model pin is [Seeed-Projects/reBot-DevArm, commit ce074041cd1c26f67ce74c2ca6fca9af22f8aee5](https://github.com/Seeed-Projects/reBot-DevArm/tree/ce074041cd1c26f67ce74c2ca6fca9af22f8aee5/Rebot_Arm_description/RS). The 25-file SHA-256 manifest includes the original URDF, required arm meshes, README and repository license. `make data` verifies them; downloaded files remain unmodified and ignored by git.

| Evidence | Use here | Remaining uncertainty |
|---|---|---|
| Public URDF: six revolute arm joints and gripper descendants | Exact arm joint origins/axes, link COMs/inertias and arm meshes; gripper omitted | The delivered revision, encoder zero convention and inertial accuracy need confirmation |
| [Seeed RS control guide](https://github.com/Seeed-Studio/wiki-documents/blob/docusaurus-version/sites/zh-CN/docs/Robotics/Robot_Kits/reBot_Arm/B601_RS/cn_reBot_Arm_B601_RS_control_mit.md): RS06 ×3 and RS00 ×3 for arm | Joint motor types; rated torque caps 11/5 N·m | Rated motor torque is not a verified continuous arm operating envelope; thermal/firmware limits remain unmodeled |
| Same guide: six arm IDs 1–6, CAN 1 Mbps, 48 V | Hardware integration notes; no bus implementation | Verify installed IDs, adapter, protocol, firmware, errors and emergency stop locally; budget command plus feedback traffic for all six motors before selecting a bus rate |
| [Pinned Python SDK RS configuration](https://github.com/Seeed-Projects/reBotArm_control_py/blob/1bcd81b22c182ec257bf04f5746e1e3d556a5f1c/config/rebotarm_rs.yaml): MIT gains, 500 Hz setting | Initial gains: kp `[50,150,150,50,50,50]`, kd `[3,10,10,5,4,4]` | Starting simulation values, not hardware-tuned gains or proof of bus timing |
| RS guide specifies J1 ±150°; public URDF uses ±2.8 rad | Intersect the ranges, using ±150° | J2/J3 guide angles and URDF zeros differ; do not substitute degree ranges without establishing the zero/sign mapping |
| Guide: 587.5 mm reach without gripper, 754.7 mm with gripper | Context only; compute tool workspace from the chain | A reach radius alone cannot establish vertical-tool reachability or collision clearance |

Some proximal inertia tensors in the RS URDF match the DM export despite different masses and geometry. We retain and label them as public CAD estimates rather than inventing replacement measurements. Advertised whole-arm mass and reach include a different end effector; they are not used to rescale the arm-only model. The documentation's repeatability figure is not absolute accuracy or force-control performance.

The source hardware description is licensed under [CERN-OHL-W-2.0](https://github.com/Seeed-Projects/reBot-DevArm/blob/ce074041cd1c26f67ce74c2ca6fca9af22f8aee5/LICENSE), © Seeed Studio. The 2026-09-08 runtime conversion changes the scene, removes gripper descendants, adds a rigid brush, intersects J1 limits, adds estimated rotor armature and bounded MIT actuators, and uses MuJoCo convex hull collision geometry. Original meshes, notices and license remain available in `data/robot/rebot/`. Keep the pin, manifest, conversion source and original attribution with distributed derivatives.

## Installation assumptions

World +Z is upward. Paper center is `(0.50,0,0)` m. Base origin is `(0.20,0,-0.005)` m on the table, placing paper center 300 mm in front of the base. The paper geometry represents a 5 mm pad atop the table, not measured paper thickness. The broad table supports both arm and sheet. This placement passes a discrete command-volume/path audit; it does not certify swept-volume clearance.

The brush handle is a 40 g capsule, 130 mm long, with 30 mm exposed bristles; the nominal tip is 160 mm along tool +Z. Tool +Z points down during writing and lifts. The mount is a rigid transform from URDF `link6`, initially identity. `RobotConfig.mount_xyz` and `mount_rpy` describe an actual changed attachment, whereas `SensorConfig.tool_offset` describes an estimated calibration error without moving the plant. There is no gripper mass, jaw actuator, wrist force sensor mass or cable/hose model. Include a real bracket/sensor's mass, COM and inertia in the delivered tool model before relying on gravity compensation.

At 50 Hz, six action slots carry requested XYZ and rotation increments; the effective rotation increments are zero. Damped IK still solves all six pose constraints to maintain vertical orientation. At 500 Hz by default, a critically damped joint target filter limits target speed to 0.8 rad/s and acceleration to 4 rad/s², followed by MIT impedance and model bias compensation. These target bounds do not guarantee identical bounds on actual motion under load. Reports include microstep actual speed, torque and joint margins. Total torque uses published rated values, not 36/14 N·m peak values or the URDF's large generic velocity fields.

The 0.002 kg·m² reflected motor armature is an explicit, unmeasured assumption. Gear friction, backlash, structural flexure, thermal derating, drive quantization, motor-current estimation and CAN jitter are not identified. Native cable brush simulation uses finer physics substeps; it does not establish a corresponding hardware command rate.

## Offline checks and the hardware boundary

`make prepare` examines a 5×5×5 grid over the command volume plus every authored pose in the seven training and four held-out characters. It records numerical FK residual, orientation error, joint margin, static gravity torque, scaled Jacobian singular value and arm/handle collision counts. The NPZ retains all candidates and validity flags; JSON retains total/valid denominators and failed indices. These are scratch-data calculations, not physically executed robot motion. `make validate` separately checks actual torque-driven execution and records it.

The shared `JointGovernor` requires finite six-joint targets within limits, monotonic acquisition/decision times, and a 40 ms command lease. It initializes at measured joint position and latches rejection faults until reset. `JointCalibration` implements `q_motor = sign*q_URDF + zero`, with the same sign for velocity/torque; roundtrip and mechanical-power invariants are tested. Its identity defaults are an unverified numerical template. The simulation lease runs on simulation time; blocking inference freezes that clock and cannot qualify a wall-clock watchdog.

Simulation episodes stop on >2 N brush force, excessive actual joint speed (>1.5 times the experiment target speed), joint-limit violation, arm/handle collision, nonfinite state or solver warnings. The event is detected after a physics substep/control transaction; the reported peak is not a guaranteed physical force cap. Stopping a simulated episode does not define a real arm stop. A hardware transport must independently supervise feedback age, temperature/error flags, command deadlines and operator stop state. It must implement a tested, load-supporting hold/stop response; merely ceasing CAN messages or commanding zero torque can let a gravity-loaded arm fall.

## Bring-up sequence when the arm arrives

1. Confirm the RS revision, URDF, firmware, power supply, CAN adapter and physical emergency-stop behavior with Seeed's procedure. Keep the brush clear of the table during initial checks; begin with manufacturer tools and measured joint feedback.
2. Establish joint order, signs and zero offsets one joint at a time. Compare measured poses with FK over several low-speed configurations. An independent pose measurement is needed to distinguish correct visual alignment from a shared software convention error.
3. Measure base-to-paper registration and the full wrist-to-tool transform. Fit the tool tip from multiple orientations/contacts, then recheck the vertical working pose. Measure handle/bracket/sensor mass and COM. Update the model pin and RobotConfig; rerun `make prepare`, `make check` and `make validate`.
4. Identify unloaded gravity/torque residuals, joint friction/backlash and small-signal response at conservative gains. Add the measured effects before training with randomization distributions; a published motor limit is not measured usable bandwidth.
5. Choose and calibrate force sensing. The simulation's world contact-force proxy cannot be read directly from RobStride motor current. A real wrist transducer requires bias, tool gravity/inertia compensation, frame transforms, filtering and timestamps; any observer needs independent calibration and error characterization.
6. Measure brush force–compression, lateral drag, lift-off hysteresis and ink footprints on the intended paper. Fit the reduced brush first; use native rods as a reference only after mass, bending, contact force, penetration and refinement checks.
7. Add an isolated transport backend at the joint-command boundary using the actual measured mapping. Keep a fast state/command watchdog independent of policy inference, bound queued commands, expire late chunks, and validate start/hold/stop before permitting paper contact. Record requested commands, effective commands, measured feedback, acquisition times, faults and calibration hashes separately.
8. Begin with unloaded vertical moves and pen lifts, then low-force strokes, then held-out paths. Compare teacher/classical/learned/zero baselines on the same conditions; report failed trials and denominators. Physical trials, not tracking scores, establish the next permissible operating envelope.

These steps are outstanding engineering work, not prerequisites for running the headless simulator. BC and optional SmolVLA artifacts are simulation experiments until the sensor, timing and actuation boundary has been measured and tested on the arm.
