"""Small, immutable experiment configuration; SI units unless documented."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class BrushConfig:
    bundles: int = 19
    radius: float = 0.004
    normal_stiffness: float = 220.0
    normal_damping: float = 0.35
    tangential_stiffness: float = 65.0
    friction: float = 0.55
    water_flow: float = 0.5  # microliters / second, continuously fed
    pigment_flow: float = 8.0  # micrograms / second


@dataclass(frozen=True)
class InkConfig:
    resolution: int = 256
    extent: float = 0.21
    water_diffusion: float = 2e-7  # m² / second
    pigment_diffusion: float = 4e-8
    adsorption: float = 0.8  # 1 / second
    evaporation: float = 0.15  # 1 / second
    optical_absorption: float = 120.0  # mm² / microgram


@dataclass(frozen=True)
class SimConfig:
    timestep: float = 0.002
    substeps: int = 10
    translation_step: float = 0.004
    rotation_step: float = 0.03
    paper_x: float = 0.50
    paper_z: float = 0.0
    record: bool = False
    randomize: bool = False
    brush: BrushConfig = field(default_factory=BrushConfig)
    ink: InkConfig = field(default_factory=InkConfig)

    @property
    def dt(self):
        return self.timestep * self.substeps

    def to_dict(self):
        return asdict(self)
