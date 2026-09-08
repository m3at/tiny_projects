"""Short physical traces for boundary tests; full glyph quality lives in make validate."""

import numpy as np
import pytest


@pytest.fixture
def short_stroke(monkeypatch):
    from shodo import env

    # Approach, loaded lateral motion, lift, and settle; no pose teleportation.
    lower = np.c_[np.full(6, 0.48), np.full(6, 0.01), np.linspace(0.002, -0.002, 6)]
    draw = np.c_[np.linspace(0.48, 0.484, 10), np.full(10, 0.01), np.full(10, -0.002)]
    lift = np.c_[np.full(14, 0.484), np.full(14, 0.01), np.linspace(-0.002, 0.016, 14)]
    path = np.vstack([lower, draw, lift, np.tile(lift[-1], (6, 1))])
    ids = np.r_[np.full(6, -1), np.zeros(10), np.full(20, -1)].astype(int)
    tilts = np.zeros_like(path)
    for value in (path, ids, tilts):
        value.flags.writeable = False
    monkeypatch.setattr(env, "reference", lambda *args, **kwargs: (path, ids, tilts))
    return len(path)
