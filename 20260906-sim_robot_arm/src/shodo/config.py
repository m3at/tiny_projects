"""Small, immutable experiment configuration; SI units unless documented."""

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BrushConfig:
    backend: str = "reduced"
    bundles: int = 19
    radius: float = 0.004
    normal_stiffness: float = 220.0
    normal_damping: float = 0.35
    tangential_stiffness: float = 65.0
    friction: float = 0.55
    water_flow: float = 0.5  # microliters / second, continuously fed
    pigment_flow: float = 8.0  # micrograms / second
    rod_segments: int = 6
    rod_radius: float = 0.00045
    young_modulus: float = 1e9
    shear_modulus: float = 3.4e8
    rod_damping: float = 1e-6
    rod_density: float = 1300.0
    rod_tip_offset: float = 0.0
    transfer_load: float = 0.3

    def __post_init__(self):
        if type(self.bundles) is not int or self.bundles not in (7, 19, 37):
            raise ValueError("bundles must be 7, 19 or 37")
        if self.backend not in ("reduced", "cable"):
            raise ValueError("Brush backend must be reduced or cable")
        if type(self.rod_segments) is not int or not 2 <= self.rod_segments <= 64:
            raise ValueError("Rod segments must be an integer in [2, 64]")
        nonnegative = {"normal_damping", "friction", "water_flow", "pigment_flow", "rod_damping"}
        for key, value in asdict(self).items():
            if key in ("bundles", "backend", "rod_tip_offset"):
                continue
            if not math.isfinite(value) or value < 0 or (value == 0 and key not in nonnegative):
                raise ValueError(f"Invalid brush coefficient: {key}")
        if not math.isfinite(self.rod_tip_offset) or not 0 <= self.rod_tip_offset <= 0.005:
            raise ValueError("Rod tip offset must be in [0, 0.005] m")


@dataclass(frozen=True)
class InkConfig:
    resolution: int = 256
    extent: float = 0.21
    water_diffusion: float = 2e-7  # m² / second
    pigment_diffusion: float = 4e-8
    adsorption: float = 0.8  # 1 / second
    evaporation: float = 0.15  # 1 / second
    optical_absorption: float = 120.0  # mm² / microgram
    transport_dt: float = 0.1
    contact_sigma: float = 0.0006

    def __post_init__(self):
        if type(self.resolution) is not int or not 32 <= self.resolution <= 1024:
            raise ValueError("Paper resolution must be in [32, 1024]")
        nonnegative = {
            "water_diffusion",
            "pigment_diffusion",
            "adsorption",
            "evaporation",
            "contact_sigma",
        }
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0 or (value == 0 and key not in nonnegative):
                raise ValueError(f"Invalid ink coefficient: {key}")


@dataclass(frozen=True)
class RobotConfig:
    """B601-RS setup assumptions, not manufacturer-qualified operating limits."""

    base_xyz: tuple = (0.20, 0.0, -0.005)
    mount_xyz: tuple = (0.0, 0.0, 0.0)
    mount_rpy: tuple = (0.0, 0.0, 0.0)
    handle_mass: float = 0.04
    armature: float = 0.002  # estimated reflected motor inertia, kg m²
    kp: tuple = (50.0, 150.0, 150.0, 50.0, 50.0, 50.0)
    kd: tuple = (3.0, 10.0, 10.0, 5.0, 4.0, 4.0)
    torque_limits: tuple = (11.0, 11.0, 11.0, 5.0, 5.0, 5.0)
    joint_speed: float = 0.8  # deliberately below published motor speeds, rad/s
    joint_acceleration: float = 4.0  # experiment limit, rad/s²

    def __post_init__(self):
        for name, size in (
            ("base_xyz", 3),
            ("mount_xyz", 3),
            ("mount_rpy", 3),
            ("kp", 6),
            ("kd", 6),
            ("torque_limits", 6),
        ):
            values = tuple(getattr(self, name))
            if len(values) != size or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Robot {name} requires {size} finite values")
            if size == 6 and any(v <= 0 for v in values):
                raise ValueError(f"Robot {name} must be positive")
            object.__setattr__(self, name, values)
        for name in ("handle_mass", "armature", "joint_speed", "joint_acceleration"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Robot {name} must be finite and positive")
        if any(v > limit for v, limit in zip(self.torque_limits, (11, 11, 11, 5, 5, 5))):
            raise ValueError("Experiment torques must not exceed published rated motor torques")


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
    compensate_brush: bool = True
    touchdown_speed: float = 0.04  # m/s, final 6 mm of the lowering reference
    force_feedback_gain: float = 0.0008  # m/N, teacher's normal-force correction
    material_variation: float = 0.2
    force_tolerance: float = 0.3  # N, reward scale
    robot: RobotConfig = field(default_factory=RobotConfig)
    brush: BrushConfig = field(default_factory=BrushConfig)
    ink: InkConfig = field(default_factory=InkConfig)

    def __post_init__(self):
        if (
            not math.isfinite(self.timestep)
            or self.timestep <= 0
            or type(self.substeps) is not int
            or self.substeps <= 0
            or self.substeps % 5
            or not math.isclose(self.dt, 0.02)
        ):
            raise ValueError("Use a 50 Hz controller and a positive multiple of 5 physics substeps")
        if any(not math.isfinite(v) or v <= 0 for v in (self.translation_step, self.rotation_step)):
            raise ValueError("Action scales must be positive")
        if not math.isfinite(self.touchdown_speed) or not 0 < self.touchdown_speed <= 0.04:
            raise ValueError("Touchdown speed must be in (0, 0.04] m/s")
        if not math.isfinite(self.force_feedback_gain) or self.force_feedback_gain < 0:
            raise ValueError("Force feedback gain must be finite and nonnegative")
        if not math.isfinite(self.material_variation) or not 0 <= self.material_variation < 1:
            raise ValueError("Material variation must be in [0, 1)")
        if not math.isfinite(self.force_tolerance) or self.force_tolerance <= 0:
            raise ValueError("Force tolerance must be finite and positive")
        if self.paper_x != 0.5 or self.paper_z != 0 or self.ink.extent != 0.21:
            raise ValueError("This scene uses a fixed 210 mm paper at (0.5, 0, 0)")
        if self.brush.backend == "cable" and self.timestep > 0.0002:
            raise ValueError("Cable bristles require timestep <= 0.0002 s")

    @property
    def dt(self):
        return self.timestep * self.substeps

    def to_dict(self):
        return asdict(self)


def load_config(path: Path | None):
    if path is None:
        return SimConfig()
    with path.open("rb") as source:
        values = tomllib.load(source)
    return config_from_dict(values)


def config_from_dict(values):
    values = dict(values)
    return SimConfig(
        robot=RobotConfig(**values.pop("robot", {})),
        brush=BrushConfig(**values.pop("brush", {})),
        ink=InkConfig(**values.pop("ink", {})),
        **values,
    )


def free_hair_bundle(
    *,
    bundles=7,
    segments=6,
    hairs=1000,
    hair_radius=75e-6,
    young=567e6,
    density=1300.0,
    packing=0.45,
):
    """Free-sliding wet-hair reference; mass and summed EI preserved across bundles.

    Wet horse-hair tensile modulus is a material analogue, not brush calibration.
    Shear modulus assumes isotropic Poisson ratio 0.5. Capsule volume is corrected.
    """
    if type(bundles) is not int or bundles not in (7, 19, 37):
        raise ValueError("Bundle count must be 7, 19 or 37")
    if type(segments) is not int or not 2 <= segments <= 64:
        raise ValueError("Rod segments must be an integer in [2, 64]")
    if any(not math.isfinite(v) or v <= 0 for v in (hairs, hair_radius, young, density, packing)):
        raise ValueError("Hair-reference parameters must be finite and positive")
    if packing > 1 or hairs < bundles:
        raise ValueError("Packing must not exceed one; bundles must not outnumber hairs")
    n = hairs / bundles
    radius = math.sqrt(n / packing) * hair_radius
    effective_young = young * packing**2 / n
    length = 0.03
    segment_length = length / segments
    capsule_correction = segment_length / (segment_length + 4 * radius / 3)
    lateral_stiffness = 3 * hairs * young * math.pi * hair_radius**4 / (4 * length**3)
    return BrushConfig(
        backend="cable",
        bundles=bundles,
        rod_segments=segments,
        rod_radius=radius,
        young_modulus=effective_young,
        shear_modulus=effective_young / 3,
        rod_density=density * packing * capsule_correction,
        normal_stiffness=20.0,
        tangential_stiffness=lateral_stiffness,
        transfer_load=0.03,
    )
