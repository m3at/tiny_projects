import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from shodo.brush import Brush
from shodo.data import strokes, trajectory
from shodo.env import ShodoEnv, reference
from shodo.learning import rollout


def test_svg_axes_map_to_paper_local_right_and_up(tmp_path):
    (tmp_path / "00041.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<path id="A-s1" d="M 0,27.25 L 109,27.25"/>'
        '<path id="A-s2" d="M 81.75,0 L 81.75,109"/>'
        "</svg>"
    )
    horizontal, vertical = strokes("A", tmp_path)
    np.testing.assert_allclose(horizontal[[0, -1]], [[-0.09, 0.045], [0.09, 0.045]])
    np.testing.assert_allclose(vertical[[0, -1]], [[0.045, 0.09], [0.045, -0.09]])
    assert (np.diff(horizontal[:, 0]) > 0).all()
    assert (np.diff(vertical[:, 1]) < 0).all()


@pytest.mark.parametrize("char,steps", [("永", 1064), ("水", 933), ("日", 1060), ("山", 723)])
def test_paper_local_trajectory_maps_once_to_world(char, steps):
    local, local_ids = trajectory(char)
    world, world_ids, _ = reference(char)
    assert len(local) == len(world) == steps
    assert np.max(np.abs(local[:, :2])) < 0.09
    np.testing.assert_array_equal(world_ids, local_ids)
    np.testing.assert_array_equal(
        world, gaussian_filter1d(local + [0.5, 0, 0], 1.2, axis=0, mode="nearest")
    )


def test_cached_geometry_cannot_leak_mutations_between_episodes():
    geometry = strokes("一")
    assert isinstance(geometry, tuple)
    for array in (*geometry, *trajectory("一"), *reference("一")):
        before = array.copy()
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0
        np.testing.assert_array_equal(array, before)


def test_unloaded_geometric_contact_is_not_reported_as_ink_accuracy(monkeypatch):
    update = Brush.update

    def unloaded(self, *args, **kwargs):
        update(self, *args, **kwargs)
        self.normal[:] = self.force[:] = self.torque[:] = 0
        return self.force, self.torque

    monkeypatch.setattr(Brush, "update", unloaded)
    metrics, history, _, _ = rollout("一")
    assert (history[:, 17] > 0).any()
    assert metrics["ink_rmse_mm"] is None
    assert metrics["pigment_mass_ug"] == 0
    assert metrics["draw_contact_fraction"] == 0


@pytest.mark.parametrize("reason", ["force", "warning"])
def test_truncation_flushes_pending_paper_transport(reason):
    env = ShodoEnv(chars="一")
    try:
        env.reset(seed=7)
        env.paper.deposit(np.array([[0.5, 0]]), np.ones(1), water=1, pigment=1)
        for _ in range(6):
            _, _, terminated, truncated, _ = env.step(env.expert())
            assert not terminated and not truncated
            assert env.paper.elapsed + env.ink_clock == pytest.approx(env.data.time)
        assert env.ink_clock == pytest.approx(0.02)
        if reason == "force":
            env.peak_force = 9
        else:
            env.data.warning.number[0] = 1
        _, _, terminated, truncated, _ = env.step(env.expert())
        assert truncated and not terminated
        assert env.ink_clock == 0
        assert env.paper.elapsed == pytest.approx(env.data.time)
        assert env.paper.elapsed == pytest.approx(0.14)
        assert env.paper.water.sum() == pytest.approx(np.exp(-0.15 * 0.14))
        assert (env.paper.mobile + env.paper.fixed).sum() == pytest.approx(1)
    finally:
        env.close()


@pytest.mark.parametrize("stop_step", [1, 5])
@pytest.mark.parametrize("terminated", [False, True])
def test_rollout_renders_reset_and_one_final_state(monkeypatch, stop_step, terminated):
    step = ShodoEnv.step

    def stop(self, action):
        obs, reward, _, _, info = step(self, action)
        done = self.index == stop_step
        return obs, reward, done and terminated, done and not terminated, info

    monkeypatch.setattr(ShodoEnv, "step", stop)
    monkeypatch.setattr(
        ShodoEnv, "render", lambda self: np.full((8, 8, 3), self.index, dtype=np.uint8)
    )
    metrics, history, _, frames = rollout("一", frames=True)
    assert metrics["truncated"] == (not terminated)
    assert len(history) == stop_step
    assert len(frames) == 2
    assert metrics["frame_times_s"] == pytest.approx([0, stop_step * 0.02])
    np.testing.assert_array_equal(np.asarray(frames[0]), 0)
    np.testing.assert_array_equal(np.asarray(frames[-1]), stop_step)
