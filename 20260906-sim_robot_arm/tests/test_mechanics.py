"""Independent constitutive and discretization checks, without downloaded arm assets."""

import numpy as np
import pytest

from shodo.config import BrushConfig, InkConfig, SimConfig
from shodo.mechanics import cantilever


def test_native_beam_mass_timestep_and_recovery():
    results = []
    for dt in (0.0001, 0.00005):
        report, _ = cantilever(timestep=dt, duration=1.0, recovery=1.0)
        assert not any(report["warnings"])
        assert report["mass_kg"] == pytest.approx(report["expected_mass_kg"], rel=1e-6)
        assert report["discrete_relative_error"] < 0.005
        assert abs(report["recovered_displacement_m"]) < 1e-7
        results.append(report["displacement_m"])
    assert results[0] == pytest.approx(results[1], rel=1e-4)


def test_configuration_rejects_invalid_discretization_and_accepts_ablations():
    for factory, values in (
        (SimConfig, {"translation_step": np.nan}),
        (SimConfig, {"substeps": 10.0}),
        (SimConfig, {"touchdown_speed": 0}),
        (BrushConfig, {"rod_segments": 6.5}),
        (BrushConfig, {"rod_segments": 1}),
        (InkConfig, {"resolution": 256.0}),
        (InkConfig, {"water_diffusion": -1}),
    ):
        with pytest.raises(ValueError):
            factory(**values)
    BrushConfig(friction=0, normal_damping=0, water_flow=0, pigment_flow=0)
    InkConfig(water_diffusion=0, pigment_diffusion=0, adsorption=0, evaporation=0, contact_sigma=0)
