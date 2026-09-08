"""B601 geometry, bounded actuation, and six-joint observation invariants."""

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shodo.actuation import JointCalibration, JointGovernor
from shodo.config import RobotConfig, SimConfig, free_hair_bundle
from shodo.contracts import SensorConfig
from shodo.rebot import JOINTS, source
from shodo.robot import RebotArm
from shodo.runtime import SensorEnv


def test_arm_fk_matches_published_urdf_and_has_no_gripper():
    robot = RebotArm()
    urdf = source()
    assert robot.model.nu == robot.model.nq == 6
    assert all("gripper" not in robot.model.body(i).name for i in range(robot.model.nbody))
    rng = np.random.default_rng(8)
    for q in rng.uniform(robot.model.jnt_range[:6, 0], robot.model.jnt_range[:6, 1], (12, 6)):
        transform = np.eye(4)
        transform[:3, 3] = robot.config.base_xyz
        for name, angle in zip(JOINTS, q, strict=True):
            joint = urdf.find(f"joint[@name='{name}']")
            origin = joint.find("origin")
            fixed = np.eye(4)
            fixed[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            fixed[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy"), sep=" ")
            ).as_matrix()
            moving = np.eye(4)
            moving[:3, :3] = Rotation.from_rotvec(
                np.fromstring(joint.find("axis").get("xyz"), sep=" ") * angle
            ).as_matrix()
            transform = transform @ fixed @ moving
        expected = transform @ np.array([0, 0, 0.16, 1])
        robot.data.qpos[:] = q
        mujoco.mj_forward(robot.model, robot.data)
        np.testing.assert_allclose(robot.tip, expected[:3], atol=1e-12)
        np.testing.assert_allclose(robot.rotation, transform[:3, :3], atol=1e-12)
    for i in range(1, 7):
        inertial = urdf.find(f"link[@name='link{i}']/inertial")
        assert robot.model.body(f"link{i}").mass[0] == pytest.approx(
            float(inertial.find("mass").get("value"))
        )
    assert RobotConfig(**json.loads(json.dumps(SimConfig().to_dict()["robot"]))) == RobotConfig()


def test_joint_governor_limits_acceleration_speed_and_latches_stale_commands():
    cfg = RobotConfig()
    governor = JointGovernor(np.tile([-2, 2], (6, 1)), cfg)
    governor.reset(np.zeros(6), 0.0)
    last_q, last_v = np.zeros(6), np.zeros(6)
    for step in range(1, 501):
        t = step * 0.002
        if step % 10 == 1:
            governor.submit(np.ones(6), (step - 1) * 0.002, (step - 1) * 0.002)
        q, v = governor.advance(t)
        assert np.max(np.abs(v)) <= cfg.joint_speed + 1e-12
        assert np.max(np.abs(v - last_v)) <= cfg.joint_acceleration * 0.002 + 1e-12
        np.testing.assert_allclose(q - last_q, v * 0.002, atol=1e-12)
        last_q, last_v = q, v
    before = governor.position.copy()
    with pytest.raises(RuntimeError, match="expired"):
        governor.advance(1.1)
    np.testing.assert_array_equal(governor.position, before)
    with pytest.raises(RuntimeError, match="expired"):
        governor.submit(np.zeros(6), 1.1, 1.1)
    governor.reset(last_q, 2.0)
    assert governor.fault is None
    with pytest.raises(RuntimeError, match="malformed"):
        governor.submit(np.full(6, np.nan), 2.0, 2.0)


def test_joint_calibration_roundtrip_preserves_mechanical_power():
    calibration = JointCalibration(signs=(-1, 1, -1, -1, 1, -1), zeros_rad=(0, 0.2, 0, 0, -0.1, 0))
    q, v, tau = np.arange(6) * 0.1, np.arange(6) * 0.01, np.arange(6) * 0.5
    motor = calibration.to_motor(q, v, tau)
    recovered = calibration.from_motor(*motor)
    for actual, expected in zip(recovered, (q, v, tau), strict=True):
        np.testing.assert_allclose(actual, expected)
    assert np.dot(v, tau) == pytest.approx(np.dot(motor[1], motor[2]))
    with pytest.raises(ValueError):
        JointCalibration(signs=(1, 1, 1, 1, 1, 0))


def test_native_brush_joints_never_enter_reserved_sensor_slots():
    config = SimConfig(
        timestep=0.0001,
        substeps=200,
        brush=replace(free_hair_bundle(segments=12), rod_tip_offset=0.001),
    )
    env = SensorEnv(
        "一", config=config, sensors=SensorConfig(joint_noise=0.001, velocity_noise=0.001)
    )
    try:
        obs, _ = env.reset(seed=7)
        assert env.unwrapped.model.nq > 6
        assert env.unwrapped.data.ncon == 0
        for sample in obs.reshape(-1, 39):
            assert sample[24] == sample[31] == 0
        assert env.unwrapped._obs()[24] == env.unwrapped._obs()[31] == 0
        np.testing.assert_array_equal(env.sample.joints[-1], 0)
        np.testing.assert_array_equal(env.env.tilts, 0)
        env.step(np.array([0, 0, 0, 1, -1, 1]))
        np.testing.assert_array_equal(env.env.last_applied_action[3:], 0)
        assert env.unwrapped.data.ncon == 0
        geoms = list(env.env.brush.geom_bundle)
        env.reset(seed=7, options={"material": {"friction": 0.3}})
        np.testing.assert_array_equal(env.env.model.geom_friction[geoms, 0], 0.3)
        env.reset(seed=7)
        np.testing.assert_array_equal(env.env.model.geom_friction[geoms, 0], config.brush.friction)
    finally:
        env.close()


def test_robot_setup_rejects_unrated_torque_and_nonfinite_mount():
    with pytest.raises(ValueError, match="rated"):
        replace(RobotConfig(), torque_limits=(36, 36, 36, 14, 14, 14))
    with pytest.raises(ValueError, match="finite"):
        replace(RobotConfig(), mount_xyz=(0, np.nan, 0))


@pytest.mark.parametrize("case", ["future_decision", "stalled_execution"])
def test_new_target_cannot_hide_a_clock_violation(case):
    governor = JointGovernor(np.tile([-2, 2], (6, 1)), RobotConfig())
    governor.reset(np.zeros(6), 0.0)
    if case == "future_decision":
        governor.submit(np.ones(6), 0.01, 0.01)
        now = 0.002
    else:
        governor.submit(np.ones(6), 1.0, 1.0)
        now = 1.002
    with pytest.raises(RuntimeError, match="expired command or nonmonotonic"):
        governor.advance(now)
    np.testing.assert_array_equal(governor.position, 0)
    governor.reset(np.zeros(6), 0.0)
    governor.submit(np.ones(6), 0.0, 0.01)
    with pytest.raises(RuntimeError, match="stale command"):
        governor.submit(np.ones(6), 0.0, 0.005)
    assert governor.fault is not None


def test_cached_models_do_not_share_mutable_physics_or_renderer_arrays():
    first, second = RebotArm(), RebotArm()
    for name in ("geom_friction", "geom_group", "tex_data", "body_mass"):
        original = getattr(second.model, name).copy()
        getattr(first.model, name).flat[0] += 1
        np.testing.assert_array_equal(getattr(second.model, name), original)
    first.data.qpos[0] = 1
    assert second.data.qpos[0] == 0
