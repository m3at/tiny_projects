"""Reduced elastic bristle bundles with unilateral contact and Coulomb memory.

This is an uncalibrated engineering model, not a discretized full hair/fluid solver.
Contact reactions are returned at the nominal tip for coupling to rigid arm dynamics.
"""

import numpy as np

from shodo.config import BrushConfig


class Brush:
    def __init__(self, config=None):
        config = config or BrushConfig()
        if config.bundles not in (7, 19, 37):
            raise ValueError("Use 7, 19 or 37 hexagonal bristle bundles")
        self.config = config
        rings = {7: 1, 19: 2, 37: 3}[config.bundles]
        offsets = [[0.0, 0.0]]
        for ring in range(1, rings + 1):
            angles = np.arange(6 * ring) * (2 * np.pi / (6 * ring))
            offsets.extend(np.c_[np.cos(angles), np.sin(angles)] * (ring / rings))
        self.offsets = np.asarray(offsets)
        self.radial = np.linalg.norm(self.offsets, axis=1)
        self.weights = np.ones(config.bundles) / config.bundles
        self.reset()

    def reset(self):
        n = self.config.bundles
        self.contact = np.zeros((n, 3))
        self.roots = np.zeros((n, 3))
        self.normal = np.zeros(n)
        self.tangent = np.zeros((n, 2))
        self.touching = np.zeros(n, dtype=bool)
        self.previous_depth = np.zeros(n)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def update(self, tip, rotation, dt, paper_z=0.0):
        cfg = self.config
        compression = max(0.0, paper_z - tip[2])
        spread = cfg.radius * (0.32 + min(compression / 0.004, 1.5))
        # Peripheral hairs are shorter, so the contact footprint grows with pressure.
        local = np.c_[self.offsets * spread, -(self.radial**2) * 0.0012]
        rest = local @ rotation.T + tip
        root_local = np.c_[self.offsets * cfg.radius, np.full(cfg.bundles, -0.03)]
        self.roots[:] = root_local @ rotation.T + tip
        depth = np.maximum(paper_z - rest[:, 2], 0.0)
        touching = depth > 0
        self.normal[:] = (
            np.maximum(
                0.0,
                cfg.normal_stiffness * depth
                + cfg.normal_damping * (depth - self.previous_depth) / dt,
            )
            * self.weights
        )
        self.normal[~touching] = 0.0
        fresh = touching & ~self.touching
        self.contact[fresh] = rest[fresh]
        kt = cfg.tangential_stiffness * self.weights
        trial = (rest[:, :2] - self.contact[:, :2]) * kt[:, None]
        magnitude = np.linalg.norm(trial, axis=1)
        limit = cfg.friction * self.normal
        ratio = np.minimum(1.0, limit / np.maximum(magnitude, 1e-12))
        self.tangent[:] = -trial * ratio[:, None]
        self.contact[:, :2] = rest[:, :2] + self.tangent / kt[:, None]
        self.contact[:, 2] = np.where(touching, paper_z, rest[:, 2])
        forces = np.c_[self.tangent, self.normal]
        self.force[:] = forces.sum(axis=0)
        self.torque[:] = np.cross(self.contact - tip, forces).sum(axis=0)
        self.touching[:] = touching
        self.previous_depth[:] = depth
        return self.force, self.torque

    @property
    def deflection(self):
        if not self.touching.any():
            return np.zeros(2)
        return -self.tangent.sum(axis=0) / self.config.tangential_stiffness
