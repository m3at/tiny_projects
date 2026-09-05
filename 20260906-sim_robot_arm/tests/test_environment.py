from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from shodo.brush import Brush
from shodo.config import InkConfig, SimConfig
from shodo.data import strokes, trajectory
from shodo.env import ACTIONS, ShodoEnv
from shodo.ink import Paper
from shodo.robot import DOWN, Panda


def test_panda_pose_and_reaction():
    robot = Panda()
    for point in ([0.41, -0.09, 0.025], [0.59, 0.09, -0.004], [0.5, 0, 0.01]):
        robot.reset(np.array(point), np.zeros(3))
        np.testing.assert_allclose(robot.tip, point, atol=1e-5)
        np.testing.assert_allclose(robot.rotation, DOWN, atol=1e-4)
        assert robot.data.ncon == 0
    robot.step(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    assert np.linalg.norm(robot.data.qfrc_applied) > 0.01
    assert np.linalg.norm(robot.data.qvel) > 1e-5
    assert (np.abs(robot.data.ctrl) <= robot.model.actuator_ctrlrange[:, 1]).all()


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


def test_ink_timestep_and_resolution():
    moments = []
    for resolution, dt in ((128, 0.02), (128, 0.01), (256, 0.02)):
        paper = Paper(replace(InkConfig(), resolution=resolution), seed=1)
        paper.fibers[:] = 1
        paper.deposit(np.array([[0.5, 0, 0]]), np.ones(1), 1.0, 1.0)
        for _ in range(round(1 / dt)):
            paper.advance(dt)
        coordinate = (np.arange(resolution) + 0.5) * paper.dx - 0.105
        moment = (
            np.sum(paper.water * (coordinate[:, None] ** 2 + coordinate[None, :] ** 2))
            / paper.water.sum()
        )
        # Subtract bilinear initialization variance; diffusion adds 4*D*t in 2D.
        moments.append(moment - paper.dx**2 / 2)
    np.testing.assert_allclose(moments, 4 * 2e-7, rtol=1e-6)


def test_strokes_api_and_clean_reset():
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
        check_env(env, skip_render_check=True)
        histories = []
        for _ in range(2):
            env.reset(seed=4)
            for _ in range(90):
                env.step(env.expert())
            histories.append(np.array(env.history))
        np.testing.assert_array_equal(*histories)
        assert env.paper.deposited_pigment > 0
        env.reset(seed=4)
        assert env.paper.mobile.sum() == 0 and env.paper.fixed.sum() == 0
        with pytest.raises(ValueError):
            env.step(np.full(ACTIONS, np.nan))
    finally:
        env.close()
