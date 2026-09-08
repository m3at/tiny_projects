"""Bounded joint targets and explicit URDF-to-motor calibration; no bus I/O.

A fault latches and produces no new command. A real transport must implement its
own verified hold/stop behavior and independent watchdog; silence is not a stop.
"""

from dataclasses import dataclass

import numpy as np


def vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite six-joint vector")
    return result.copy()


@dataclass(frozen=True)
class JointCalibration:
    """q_motor = sign*q_urdf + zero; velocity and torque use the same sign.

    Identity is an unverified template, never evidence of encoder alignment.
    """

    signs: tuple = (1, 1, 1, 1, 1, 1)
    zeros_rad: tuple = (0, 0, 0, 0, 0, 0)

    def __post_init__(self):
        signs = vector(self.signs, "Motor signs")
        zeros = vector(self.zeros_rad, "Motor zeros")
        if not np.isin(signs, (-1, 1)).all():
            raise ValueError("Motor signs must be +1 or -1")
        object.__setattr__(self, "signs", tuple(signs))
        object.__setattr__(self, "zeros_rad", tuple(zeros))

    def to_motor(self, positions, velocities, torques):
        return (
            vector(positions, "Positions") * self.signs + self.zeros_rad,
            vector(velocities, "Velocities") * self.signs,
            vector(torques, "Torques") * self.signs,
        )

    def from_motor(self, positions, velocities, torques):
        return (
            (vector(positions, "Positions") - self.zeros_rad) * self.signs,
            vector(velocities, "Velocities") * self.signs,
            vector(torques, "Torques") * self.signs,
        )


class JointGovernor:
    """Acceleration/speed-limited critically damped target filter, with a lease.

    All times use one monotonic clock. Reset initializes from measured position;
    it does not send a home command. Limits are experiments, not safety ratings.
    """

    def __init__(self, limits, config, *, lease_s=0.04):
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (6, 2) or not np.isfinite(limits).all():
            raise ValueError("Joint limits must have shape (6, 2)")
        self.lower, self.upper = limits[:, 0] + 0.01, limits[:, 1] - 0.01
        if (self.lower >= self.upper).any() or not np.isfinite(lease_s) or lease_s <= 0:
            raise ValueError("Invalid joint bounds or command lease")
        self.config, self.lease_s = config, lease_s
        self.fault = "not initialized"

    def _fail(self, reason):
        self.fault = reason
        raise RuntimeError(f"Joint command rejected: {reason}")

    def reset(self, measured, now):
        q = vector(measured, "Measured position")
        if not np.isfinite(now) or (q < self.lower).any() or (q > self.upper).any():
            self._fail("reset outside joint limits or invalid clock")
        self.position = self.target = q.copy()
        self.velocity = np.zeros(6)
        self.last_update = self.acquired = self.decided = float(now)
        self.fault = None

    def submit(self, target, acquired, now):
        if self.fault:
            self._fail(self.fault)
        try:
            q = vector(target, "Target")
        except ValueError:
            self._fail("nonfinite or malformed target")
        if not np.isfinite([acquired, now]).all() or not self.acquired <= acquired <= now:
            self._fail("nonmonotonic acquisition time")
        if now < max(self.last_update, self.decided) or now - acquired > self.lease_s:
            self._fail("stale command")
        if (q < self.lower - 1e-12).any() or (q > self.upper + 1e-12).any():
            self._fail("target outside joint limits")
        self.target, self.acquired, self.decided = q, float(acquired), float(now)

    def advance(self, now):
        if self.fault:
            self._fail(self.fault)
        if (
            not np.isfinite(now)
            or now <= self.last_update
            or now < self.decided
            or now - self.last_update > self.lease_s + 1e-12
            or now - self.acquired > self.lease_s + 1e-12
        ):
            self._fail("expired command or nonmonotonic clock")
        dt = now - self.last_update
        cfg = self.config
        acceleration = np.clip(
            900 * (self.target - self.position) - 60 * self.velocity,
            -cfg.joint_acceleration,
            cfg.joint_acceleration,
        )
        velocity = np.clip(self.velocity + acceleration * dt, -cfg.joint_speed, cfg.joint_speed)
        position = self.position + velocity * dt
        # Do not silently clip into a discontinuous velocity at a hard boundary.
        if (position < self.lower).any() or (position > self.upper).any():
            self._fail("filtered command reached joint boundary")
        self.position, self.velocity, self.last_update = position, velocity, float(now)
        return position.copy(), velocity.copy()
