"""Conservative water/pigment transport on a fibrous paper grid.

Water is in microliters, pigment in micrograms; fields store per-cell quantities.
No-flux boundaries conserve pigment. Water evaporates; mobile pigment adsorbs into
an immobile field. All coefficients are explicit, procedural material assumptions.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from shodo.config import InkConfig


class Paper:
    def __init__(self, config=None, center_x=0.5, seed=0):
        config = config or InkConfig()
        self.config = config
        self.center_x = center_x
        n = config.resolution
        self.dx = config.extent / n
        self.area_mm2 = (self.dx * 1000) ** 2
        self.water = np.zeros((n, n))
        self.mobile = np.zeros((n, n))
        self.fixed = np.zeros((n, n))
        rng = np.random.default_rng(seed)
        self.fibers = rng.uniform(0.75, 1.25, (n, n))
        self.deposited_pigment = 0.0
        self.deposited_water = 0.0
        self.elapsed = 0.0
        self.bounds = None

    def deposit(self, positions, weights, water, pigment):
        if not np.isfinite([water, pigment]).all() or water < 0 or pigment < 0:
            raise ValueError("Deposited water and pigment must be finite and nonnegative")
        positions, weights = np.asarray(positions), np.asarray(weights)
        if len(positions) and (
            positions.ndim != 2
            or positions.shape[1] < 2
            or weights.shape != (len(positions),)
            or not np.isfinite(positions).all()
            or not np.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("Expected finite contact positions and matching nonnegative weights")
        if not len(positions) or weights.sum() <= 0:
            return
        n = self.config.resolution
        # Use cell centers for bilinear splats. Reject off-paper contacts, never clamp.
        uv = (
            positions[:, :2] - [self.center_x - self.config.extent / 2, -self.config.extent / 2]
        ) / self.dx - 0.5
        inside = ((uv >= -0.5) & (uv <= n - 0.5)).all(axis=1)
        weights = weights / weights.sum()
        uv, weights = uv[inside], weights[inside]
        if not len(uv):
            return
        base = np.floor(uv).astype(int)
        fraction = uv - base
        margin = int(np.ceil(3 * self.config.contact_sigma / self.dx))
        # Build the complete footprint before clipping it to the paper. Clipping
        # the bilinear splat first would discard mass twice near an edge.
        footprint = np.array(
            [
                base[:, 1].min() - margin,
                base[:, 0].min() - margin,
                base[:, 1].max() + 2 + margin,
                base[:, 0].max() + 2 + margin,
            ]
        )
        y0, x0, y1, x1 = footprint
        patch = np.zeros((y1 - y0, x1 - x0))
        for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            w = (
                weights
                * (fraction[:, 0] if ox else 1 - fraction[:, 0])
                * (fraction[:, 1] if oy else 1 - fraction[:, 1])
            )
            index = (base[:, 1] - y0 + oy, base[:, 0] - x0 + ox)
            np.add.at(patch, index, w)
        patch = gaussian_filter(
            patch, self.config.contact_sigma / self.dx, mode="constant", truncate=3
        )
        box = np.clip(footprint, 0, n)
        by0, bx0, by1, bx1 = box
        patch = patch[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0]
        y0, x0, y1, x1 = box
        self.water[y0:y1, x0:x1] += water * patch
        self.mobile[y0:y1, x0:x1] += pigment * patch
        self.deposited_water += water * patch.sum()
        self.deposited_pigment += pigment * patch.sum()
        if self.bounds is None:
            self.bounds = box
        else:
            self.bounds[:2] = np.minimum(self.bounds[:2], box[:2])
            self.bounds[2:] = np.maximum(self.bounds[2:], box[2:])

    @staticmethod
    def _diffuse(field, diffusivity, dt_dx2):
        # Pairwise symmetric flux, so sum(field) is unchanged (including boundaries).
        horizontal = (
            0.5
            * (diffusivity[:, 1:] + diffusivity[:, :-1])
            * (field[:, 1:] - field[:, :-1])
            * dt_dx2
        )
        vertical = 0.5 * (diffusivity[1:] + diffusivity[:-1]) * (field[1:] - field[:-1]) * dt_dx2
        field[:, :-1] += horizontal
        field[:, 1:] -= horizontal
        field[:-1] += vertical
        field[1:] -= vertical

    def advance(self, dt):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("Transport duration must be finite and nonnegative")
        if dt == 0:
            return
        self.elapsed += dt
        if self.bounds is None:
            return
        cfg = self.config
        # CFL positivity bound covers the largest heterogeneous diffusion coefficient.
        substeps = max(
            1,
            int(
                np.ceil(
                    dt
                    * max(cfg.water_diffusion, cfg.pigment_diffusion)
                    * 1.25
                    / (0.24 * self.dx**2)
                )
            ),
        )
        h = dt / substeps
        self.bounds[:2] = np.maximum(0, self.bounds[:2] - substeps)
        self.bounds[2:] = np.minimum(cfg.resolution, self.bounds[2:] + substeps)
        y0, x0, y1, x1 = self.bounds
        water = self.water[y0:y1, x0:x1]
        mobile = self.mobile[y0:y1, x0:x1]
        fixed = self.fixed[y0:y1, x0:x1]
        fibers = self.fibers[y0:y1, x0:x1]
        for _ in range(substeps):
            self._diffuse(water, cfg.water_diffusion * fibers, h / self.dx**2)
            wetness = np.clip(water / (0.003 * self.area_mm2), 0.0, 1.0)
            self._diffuse(mobile, cfg.pigment_diffusion * fibers * wetness, h / self.dx**2)
            adsorbed = mobile * (1 - np.exp(-cfg.adsorption * fibers * h))
            mobile -= adsorbed
            fixed += adsorbed
            water *= np.exp(-cfg.evaporation * h)

    def image(self):
        density = (self.mobile + self.fixed) / self.area_mm2
        transmission = np.exp(-self.config.optical_absorption * density)
        texture = 1 + 0.02 * (self.fibers - 1)
        rgb = np.array([250, 248, 239])[None, None, :] * texture[..., None]
        rgb = 15 + (rgb - 15) * transmission[..., None]
        return Image.fromarray(np.flipud(np.clip(rgb, 0, 255).astype(np.uint8)))
