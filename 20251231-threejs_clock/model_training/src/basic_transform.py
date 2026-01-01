import math


def to_floats(t: str):
    """Transform an hour like 04_01_29 to two floats in [0, 1)"""
    h, m, s = map(int, t.split("_"))
    return (
        (h * 3600 + m * 60 + s) / 43200.0,  # hour in [0,1) over 12h, incl m/s
        (m * 60 + s) / 3600.0,  # minute in [0,1) over 60m, incl seconds
    )


def from_floats(hour_float: float):
    """Inverse, for display (only hour float is required)."""

    tot = int(round((hour_float % 1.0) * 43200)) % 43200
    h, rem = divmod(tot, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}_{m:02d}_{s:02d}"


def to_sincos(x: float):
    a = (x % 1.0) * 2.0 * math.pi
    return math.sin(a), math.cos(a)


def from_sincos(s: float, c: float):
    return (math.atan2(s, c) / (2.0 * math.pi)) % 1.0
