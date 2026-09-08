from dataclasses import replace

import numpy as np
import pytest

from shodo.config import InkConfig
from shodo.ink import Paper


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("side", [-1, 1])
def test_contacts_cover_the_full_physical_paper(axis, side):
    paper = Paper(replace(InkConfig(), contact_sigma=0))
    half = paper.config.extent / 2
    for inset, retained in ((paper.dx / 4, 0.75), (0, 0.5), (-paper.dx / 4, 0)):
        position = np.array([paper.center_x, 0.0])
        position[axis] += side * (half - inset)
        before = paper.deposited_pigment
        paper.deposit(position[None], np.ones(1), water=1, pigment=2)
        assert paper.deposited_pigment - before == pytest.approx(2 * retained)
    assert paper.mobile.sum() == pytest.approx(paper.deposited_pigment)


def test_paper_grid_and_image_have_opposite_row_directions():
    paper = Paper()
    position = np.array([[0.54, 0.03]])
    paper.deposit(position, np.ones(1), water=1, pigment=0.001)
    rows, columns = np.indices(paper.mobile.shape)
    center = np.array(
        [
            paper.center_x
            - paper.config.extent / 2
            + (np.average(columns, weights=paper.mobile) + 0.5) * paper.dx,
            -paper.config.extent / 2 + (np.average(rows, weights=paper.mobile) + 0.5) * paper.dx,
        ]
    )
    np.testing.assert_allclose(center, position[0], atol=1e-12)
    assert np.asarray(paper.image())[: paper.config.resolution // 2].min() < 240


@pytest.mark.parametrize("sigma", [0, 0.0006])
@pytest.mark.parametrize(
    "offset, retained",
    [
        ([0.105, 0], 0.5),
        ([-0.105, 0], 0.5),
        ([0, 0.105], 0.5),
        ([0, -0.105], 0.5),
        ([0.105, -0.105], 0.25),
    ],
)
def test_symmetric_boundary_footprints_are_clipped_only_once(sigma, offset, retained):
    paper = Paper(replace(InkConfig(), contact_sigma=sigma))
    position = np.array([[paper.center_x, 0]]) + offset
    paper.deposit(position, np.ones(1), water=1, pigment=2)
    assert paper.deposited_water == pytest.approx(retained, abs=1e-12)
    assert paper.deposited_pigment == pytest.approx(2 * retained, abs=1e-12)
    assert paper.mobile.sum() == pytest.approx(paper.deposited_pigment)
