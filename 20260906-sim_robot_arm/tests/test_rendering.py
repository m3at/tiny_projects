"""Headless checks of world, paper-grid, and displayed-image coordinates."""

import numpy as np
import pytest

from shodo.env import ShodoEnv


@pytest.mark.parametrize("position", [(0.55, 0.04)])
def test_deposited_ink_matches_world_coordinates_in_both_views(position):
    env = ShodoEnv(chars="一")
    try:
        env.reset(seed=7)
        env.paper.deposit(np.array([position]), np.ones(1), water=1, pigment=2)
        env.render()
        renderer = env.renderer
        # A calibrated top-down camera makes world projection independent of texture UVs.
        renderer.camera.lookat[:] = [0.5, 0, 0]
        renderer.camera.distance = 0.55
        renderer.camera.azimuth = 90
        renderer.camera.elevation = -90
        renderer.options.geomgroup[:] = 0
        renderer.options.geomgroup[5] = 1
        env.model.geom_group[env.model.geom("paper").id] = 5
        frame = renderer.frame(env)
        calibration = renderer.camera_metadata()
        camera_point = np.asarray(calibration["world_to_camera"]) @ [*position, 0, 1]
        projected = np.asarray(calibration["intrinsics"]) @ camera_point[:3]
        projected = projected[:2] / projected[2]
        scale = 480 / (2 * 0.55 * np.tan(np.deg2rad(env.model.vis.global_.fovy / 2)))
        world_pixel = np.array([320, 240]) + scale * np.array(
            [position[0] - env.paper.center_x, -position[1]]
        )
        np.testing.assert_allclose(projected, world_pixel - 0.5, atol=0.001)
        image_pixel = np.array([800, 245]) + 300 / env.paper.config.extent * np.array(
            [position[0] - env.paper.center_x, -position[1]]
        )
        for expected in (world_pixel, image_pixel):
            x, y = np.rint(expected).astype(int)
            patch = frame[y - 5 : y + 6, x - 5 : x + 6]
            weights = np.maximum(0, 100 - patch.mean(axis=2))
            assert weights.sum() > 100, "No ink at the projected physical contact"
            rows, columns = np.indices(weights.shape)
            centroid = np.array(
                [
                    x - 5 + np.average(columns, weights=weights),
                    y - 5 + np.average(rows, weights=weights),
                ]
            )
            np.testing.assert_allclose(centroid, expected - 0.5, atol=1)
        # Depositing must not move or mirror the independently displayed paper image.
        paper = np.asarray(env.paper.image())
        weights = np.maximum(0, 100 - paper.mean(axis=2))
        rows, columns = np.indices(weights.shape)
        row = np.average(rows, weights=weights)
        column = np.average(columns, weights=weights)
        recovered = np.array(
            [
                env.paper.center_x + (column + 0.5) * env.paper.dx - env.paper.config.extent / 2,
                env.paper.config.extent / 2 - (row + 0.5) * env.paper.dx,
            ]
        )
        np.testing.assert_allclose(recovered, position, atol=env.paper.dx)
    finally:
        env.close()
