from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shodo.brush import Brush
from shodo.config import InkConfig, SimConfig
from shodo.data import TEST, TRAIN, strokes, trajectory
from shodo.env import ACTIONS, ShodoEnv
from shodo.ink import Paper
from shodo.robot import DOWN, RebotArm


def test_rebot_pose_and_reaction():
    robot = RebotArm()
    for point in ([0.41, -0.09, 0.025], [0.59, 0.09, -0.004], [0.5, 0, 0.01]):
        robot.reset(np.array(point), np.zeros(3))
        np.testing.assert_allclose(robot.tip, point, atol=1e-5)
        np.testing.assert_allclose(robot.rotation, DOWN, atol=1e-4)
        assert robot.data.ncon == 0
    robot.step(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    assert np.linalg.norm(robot.data.qfrc_applied) > 0.01
    assert np.linalg.norm(robot.data.qvel) > 1e-5
    assert (np.abs(robot.data.actuator_force) <= robot.model.actuator_forcerange[:, 1]).all()


def test_ik_minimal_pipeline_matches_full_dynamics():
    robot = RebotArm()
    full, minimal = mujoco.MjData(robot.model), mujoco.MjData(robot.model)
    a, b = np.zeros_like(robot.jac), np.zeros_like(robot.jac)
    rng = np.random.default_rng(7)
    for q in rng.uniform(
        robot.model.jnt_range[:6, 0] + 0.01, robot.model.jnt_range[:6, 1] - 0.01, (20, 6)
    ):
        full.qpos[:6] = minimal.qpos[:6] = q
        mujoco.mj_forward(robot.model, full)
        mujoco.mj_kinematics(robot.model, minimal)
        mujoco.mj_comPos(robot.model, minimal)
        mujoco.mj_jacSite(robot.model, full, a[:3], a[3:], robot.site)
        mujoco.mj_jacSite(robot.model, minimal, b[:3], b[3:], robot.site)
        np.testing.assert_array_equal(full.site_xpos[robot.site], minimal.site_xpos[robot.site])
        np.testing.assert_array_equal(a, b)


def test_bristle_pressure_friction_and_memory():
    brush = Brush()
    loads, widths = [], []
    for depth in (0.0005, 0.002, 0.004):
        brush.reset()
        for _ in range(3):
            brush.update(np.array([0.5, 0, -depth]), DOWN, 0.002)
        loads.append(brush.force[2])
        widths.append(np.ptp(brush.contact[brush.touching, 0]))
    assert np.all(np.diff(loads) > 0)
    assert np.all(np.diff(widths) > 0)
    initial = brush.contact.copy()
    brush.update(np.array([0.5001, 0, -0.004]), DOWN, 0.002)
    np.testing.assert_allclose(brush.contact, initial, atol=1e-10)
    for x in np.linspace(0.5001, 0.52, 30):
        brush.update(np.array([x, 0, -0.004]), DOWN, 0.002)
        assert np.all(
            np.linalg.norm(brush.tangent, axis=1) <= brush.config.friction * brush.normal + 1e-12
        )
    assert brush.force[0] < 0
    old = brush.contact.copy()
    brush.update(np.array([0.5199, 0, -0.004]), DOWN, 0.002)
    np.testing.assert_allclose(brush.contact, old, atol=1e-10)
    brush.update(np.array([0.52, 0, 0.02]), DOWN, 0.002)
    np.testing.assert_array_equal(brush.force, np.zeros(3))
    assert not brush.touching.any()


def test_paper_conservation_drying_and_boundaries():
    paper = Paper(seed=2)
    paper.deposit(np.array([[0.5, 0, 0]]), np.ones(1), 1.0, 2.0)
    for _ in range(100):
        paper.advance(0.02)
    assert paper.mobile.min() >= 0 and paper.water.min() >= 0
    assert paper.mobile.sum() + paper.fixed.sum() == pytest.approx(2.0, abs=1e-12)
    assert paper.water.sum() == pytest.approx(np.exp(-0.15 * 2), rel=1e-10)
    assert paper.fixed.sum() > 1
    assert np.count_nonzero(paper.water > 1e-5) > 4
    before = paper.deposited_pigment
    paper.deposit(np.array([[0.8, 0, 0]]), np.ones(1), 1.0, 2.0)
    assert paper.deposited_pigment == before
    paper.water[:] = 0
    distribution = paper.mobile + paper.fixed
    paper.advance(1.0)
    np.testing.assert_allclose(paper.mobile + paper.fixed, distribution, atol=1e-14)
    with pytest.raises(ValueError):
        paper.advance(-0.1)
    with pytest.raises(ValueError):
        paper.deposit(np.array([[0.5, 0, 0]]), np.ones(1), -1.0, 1.0)


def test_bristle_force_and_moment_rotate_with_the_motion():
    original, transformed = Brush(), Brush()
    rotation = Rotation.from_rotvec([0, 0, 0.73]).as_matrix()
    origin = np.array([0.5, 0.0, 0.0])
    shift = np.array([0.02, -0.01, 0.0])
    for x in np.linspace(-0.01, 0.01, 30):
        tip = origin + [x, 0.003 * np.sin(100 * x), -0.003]
        original.update(tip, DOWN, 0.002)
        transformed.update(origin + shift + rotation @ (tip - origin), rotation @ DOWN, 0.002)
        np.testing.assert_allclose(transformed.force, rotation @ original.force, atol=1e-12)
        np.testing.assert_allclose(transformed.torque, rotation @ original.torque, atol=1e-12)


@pytest.mark.parametrize("sigma", [0, 0.006])
def test_paper_edge_accounting_and_large_transport_step(sigma):
    config = replace(
        InkConfig(),
        resolution=32,
        contact_sigma=sigma,
        water_diffusion=1e-4,
        pigment_diffusion=1e-4,
    )
    paper = Paper(config, seed=4)
    # One valid point near a corner, one wholly outside; do not renormalize loss.
    paper.deposit(np.array([[0.401, -0.099, 0], [0.8, 0, 0]]), np.ones(2), 2.0, 4.0)
    water, pigment = paper.deposited_water, paper.deposited_pigment
    assert 0 < water <= 1.0
    assert 0 < pigment <= 2.0
    paper.advance(2.0)
    assert min(paper.water.min(), paper.mobile.min(), paper.fixed.min()) >= 0
    assert paper.water.sum() == pytest.approx(water * np.exp(-0.3), rel=1e-12)
    assert (paper.mobile + paper.fixed).sum() == pytest.approx(pigment, rel=1e-12)


def test_ink_timestep_and_resolution():
    moments = []
    for resolution, dt in ((128, 0.02), (128, 0.01), (256, 0.02)):
        paper = Paper(replace(InkConfig(), resolution=resolution), seed=1)
        paper.fibers[:] = 1
        paper.deposit(np.array([[0.5, 0, 0]]), np.ones(1), 1.0, 1.0)
        coordinate = (np.arange(resolution) + 0.5) * paper.dx - 0.105
        radius2 = coordinate[:, None] ** 2 + coordinate[None, :] ** 2
        initial = np.sum(paper.water * radius2) / paper.water.sum()
        for _ in range(round(1 / dt)):
            paper.advance(dt)
        coordinate = (np.arange(resolution) + 0.5) * paper.dx - 0.105
        moment = (
            np.sum(paper.water * (coordinate[:, None] ** 2 + coordinate[None, :] ** 2))
            / paper.water.sum()
        )
        # Subtract the deposited footprint; diffusion adds 4*D*t in 2D.
        moments.append(moment - initial)
    np.testing.assert_allclose(moments, 4 * 2e-7, rtol=1e-6)


def test_strokes_api_and_clean_reset(short_stroke):
    for char in TRAIN + TEST + "書道愛龍風雨":
        path, ids = trajectory(char)
        assert np.isfinite(path).all(), char
        assert len(path) == len(ids)
        assert np.all(np.diff(ids[ids >= 0]) >= 0)
    assert len(strokes("永")) == 5
    path, ids = trajectory("永")
    assert set(ids) == {-1, 0, 1, 2, 3, 4}
    assert np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) < 0.00121
    for i in range(4):
        assert (
            path[np.flatnonzero(ids == i)[-1] : np.flatnonzero(ids == i + 1)[0], 2].max() >= 0.025
        )
    env = ShodoEnv(chars="一", config=SimConfig(record=True))
    try:
        observations, histories = [], []
        for _ in range(2):
            obs, _ = env.reset(seed=4)
            observations.append(obs)
            assert env.paper.mobile.sum() == env.paper.fixed.sum() == 0
            assert env.data.time == 0 and not env.history
            for _ in range(20):
                env.step(env.expert())
            assert env.paper.deposited_pigment > 0
            histories.append(np.array(env.history))
        np.testing.assert_array_equal(*observations)
        np.testing.assert_array_equal(*histories)
        with pytest.raises(ValueError):
            env.step(np.full(ACTIONS, np.nan))
    finally:
        env.close()


def test_raster_metric_detects_missing_and_displaced_ink():
    from shodo.learning import raster_metrics

    paper = Paper(seed=2)
    target = np.c_[np.linspace(0.46, 0.54, 100), np.zeros(100)]
    assert raster_metrics(paper, target)["raster_coverage_fraction"] == 0
    paper.deposit(np.c_[target, np.zeros(100)], np.ones(100), 1.0, 10.0)
    metrics = raster_metrics(paper, target)
    assert metrics["raster_coverage_fraction"] > 0.99
    assert metrics["raster_spill_fraction"] == 0
    displaced = raster_metrics(paper, target + [0, 0.02])
    assert displaced["raster_coverage_fraction"] == 0
    assert displaced["raster_spill_fraction"] == 1


def test_material_perturbation_keeps_authored_force_target():
    env = ShodoEnv(chars="一")
    try:
        env.reset(seed=1)
        target = env.target_force.copy()
        env.reset(seed=1, options={"material": {"normal_stiffness": 330.0, "friction": 0.4}})
        np.testing.assert_array_equal(target, env.target_force)
        assert env.brush.config.normal_stiffness == 330
        assert env.brush.config.friction == 0.4
    finally:
        env.close()
