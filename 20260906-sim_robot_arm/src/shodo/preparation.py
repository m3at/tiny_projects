"""Headless geometric/load audit for the B601-RS setup; never connects hardware."""

import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from shodo.artifacts import provenance
from shodo.config import SimConfig
from shodo.contracts import ActionContract
from shodo.data import TEST, TRAIN
from shodo.env import reference
from shodo.robot import DOWN, HOME, RebotArm


def audit(directory, config=None):
    config = config or SimConfig()
    robot = RebotArm(config.timestep, config.brush, config.robot)
    limits = robot.model.jnt_range[:6]
    # Scratch FK/dynamics evaluate candidates, never move the executing robot.
    probe = mujoco.MjData(robot.model)
    jac = np.zeros((6, robot.model.nv))
    candidates = []
    bounds = ActionContract().to_dict()
    for point in itertools.product(
        *(
            np.linspace(lo, hi, 5)
            for lo, hi in zip(bounds["command_lower"][:3], bounds["command_upper"][:3], strict=True)
        )
    ):
        candidates.append(("workspace_grid", -1, np.asarray(point)))
    for char in TRAIN + TEST:
        path, _, _ = reference(char, config.paper_x, config.touchdown_speed)
        candidates.extend((char, index, point) for index, point in enumerate(path))
    rows = []
    for label, index, point in candidates:
        if index <= 0:
            robot.q_target = HOME.copy()
        q = robot.inverse(point, np.zeros(3), iterations=150 if index <= 0 else 12)
        robot.q_target = q  # seed the next numerical IK query, not an executing command
        probe.qpos[:6] = q
        mujoco.mj_forward(robot.model, probe)
        position_error = float(np.linalg.norm(probe.site_xpos[robot.site] - point))
        rotation = probe.site_xmat[robot.site].reshape(3, 3)
        angular_error = float(np.linalg.norm(Rotation.from_matrix(rotation @ DOWN.T).as_rotvec()))
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        mujoco.mj_jacSite(robot.model, probe, jac[:3], jac[3:], robot.site)
        scaled = jac[:, :6].copy()
        scaled[:3] /= 0.2  # Explicit 200 mm characteristic length for dimensionless conditioning.
        singular = float(np.linalg.svd(scaled, compute_uv=False)[-1])
        gravity = probe.qfrc_bias[:6]
        gravity_fraction = float(np.max(np.abs(gravity) / config.robot.torque_limits))
        contacts = robot.forbidden_contacts(probe)
        valid = (
            position_error < 0.001
            and angular_error < 0.01
            and contacts == 0
            and margin > 0.01
            and gravity_fraction < 1
        )
        rows.append(
            (
                label,
                index,
                point,
                q,
                gravity.copy(),
                position_error,
                angular_error,
                margin,
                singular,
                gravity_fraction,
                contacts,
                valid,
            )
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "workspace.npz",
        labels=np.array([r[0] for r in rows]),
        reference_index=np.array([r[1] for r in rows]),
        requested_xyz=np.array([r[2] for r in rows]),
        joint_positions=np.array([r[3] for r in rows]),
        static_torque_nm=np.array([r[4] for r in rows]),
        diagnostics=np.array([r[5:] for r in rows]),
        diagnostic_columns=np.array(
            [
                "position_error_m",
                "orientation_error_rad",
                "joint_margin_rad",
                "scaled_jacobian_min_singular",
                "gravity_torque_fraction",
                "forbidden_contacts",
                "valid",
            ]
        ),
    )
    groups = {}
    for label in dict.fromkeys(r[0] for r in rows):
        selected = [r for r in rows if r[0] == label]
        groups[label] = {
            "samples": len(selected),
            "valid": sum(r[-1] for r in selected),
            "max_position_error_mm": max(r[5] for r in selected) * 1000,
            "max_orientation_error_deg": float(np.rad2deg(max(r[6] for r in selected))),
            "minimum_joint_margin_rad": min(r[7] for r in selected),
            "minimum_scaled_jacobian_singular": min(r[8] for r in selected),
            "max_static_torque_fraction": max(r[9] for r in selected),
            "collision_samples": sum(r[10] > 0 for r in selected),
            "failed_reference_indices": [r[1] for r in selected if not r[-1]],
        }
    report = {
        "kind": "offline B601-RS workspace and static-load audit",
        "passed": all(r[-1] for r in rows),
        "hardware_qualified": False,
        "config": config.to_dict(),
        "provenance": provenance(),
        "groups": groups,
        "arm_mass_without_gripper_kg": float(
            sum(
                robot.model.body_mass[robot.model.body(name).id]
                for name in ["base_link", *[f"link{i}" for i in range(1, 7)]]
            )
        ),
        "tool_handle_mass_kg": config.robot.handle_mass,
        "jacobian_scaling": "translation rows / 0.2 m, rotation rows unchanged; diagnostic only",
        "limitations": [
            "Discrete candidates, not continuous collision certification",
            "Static gravity test omits dynamic acceleration and contact loads",
            "Public URDF inertias and collision convex hulls are unverified",
            "Encoder signs/zeros, tool transform, table registration and force sensing require measurement",
            "Host lease checks do not implement a motor firmware watchdog or emergency stop",
        ],
    }
    path = directory / "preparation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"B601-RS preparation: {sum(r[-1] for r in rows)}/{len(rows)} candidate poses valid")
    print(
        f"Report: {path.resolve()}\nFull workspace arrays: {(directory / 'workspace.npz').resolve()}"
    )
    if not report["passed"]:
        raise RuntimeError(f"Workspace/load audit failed; inspect {path}")
    return report
