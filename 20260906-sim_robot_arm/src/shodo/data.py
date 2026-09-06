"""Pinned upstream data and stroke-preserving trajectory preparation."""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock
from svgpathtools import parse_path

DATA = Path("data/kanjivg")
URL = "https://github.com/KanjiVG/kanjivg/releases/download/r20250816/kanjivg-20250816-main.zip"
SHA256 = "69a2944ec1183086fdee5ba9c1f48bc306b867480a95b2f337f3203bf50689a3"
TRAIN = "一二三十木大人"
TEST = "永水日山"


def fetch(root: Path = DATA):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".fetch.lock", timeout=120):
        return _fetch(root)


def _fetch(root):
    archive = root / "source.zip"
    digest = None
    if archive.exists():
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        temporary = root / "source.download"
        urllib.request.urlretrieve(URL, temporary)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                raise ValueError("KanjiVG archive checksum mismatch")
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        count = 0
        for member in source.infolist():
            name = Path(member.filename).name
            if name.endswith(".svg") or "COPYING" in name or "LICENSE" in name:
                temporary = root / (name + ".pending")
                temporary.write_bytes(source.read(member))
                temporary.replace(root / name)
                count += name.endswith(".svg")
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": SHA256,
                "svg_count": count,
                "author": "Ulrich Apel and KanjiVG contributors",
                "license": "CC-BY-SA-3.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                "processing": "Original SVGs; trajectories resampled at runtime; see README",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {count} SVGs in {root}")


@lru_cache(maxsize=256)
def strokes(char: str, root: Path = DATA) -> tuple[np.ndarray, ...]:
    """Read-only paper-local XY in meters, centered at zero; X right and Y up."""
    if len(char) != 1:
        raise ValueError("Expected exactly one character")
    file = root / f"{ord(char):05x}.svg"
    if not file.exists():
        raise FileNotFoundError(f"Missing {file}; run make data")
    tree = ET.parse(file)
    result = []
    for node in tree.iter("{http://www.w3.org/2000/svg}path"):
        if "-s" not in node.get("id", ""):
            continue
        path = parse_path(node.attrib["d"])
        length = path.length()
        count = max(2, int(np.ceil(length * 0.18 / 109 / 0.0008)) + 1)
        # Vectorized chord-length table avoids hundreds of scalar root solves.
        dense = np.concatenate(
            [
                segment.point(np.linspace(0, 1, max(16, int(segment.length() * 4) + 1)))
                for segment in path
            ]
        )
        distance = np.r_[0, np.cumsum(np.abs(np.diff(dense)))]
        samples = np.linspace(0, distance[-1], count)
        points = np.c_[
            np.interp(samples, distance, dense.real), np.interp(samples, distance, dense.imag)
        ]
        points = (points / 109 - 0.5) * np.array([0.18, -0.18])
        points.flags.writeable = False
        result.append(points)
    if not result:
        raise ValueError(f"No strokes found in {file}")
    return tuple(result)


@lru_cache(maxsize=256)
def trajectory(char: str, touchdown_speed: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Paper-local XYZ at 50 Hz; Z is height above the paper, stroke IDs -1 in air.

    Each stroke follows lift, travel, lower, draw and lift phases.
    """
    paths = strokes(char)
    points, ids = [], []
    previous = np.r_[paths[0][0], 0.025]

    def segment(end, stroke_id, spacing=0.0012):
        nonlocal previous
        n = max(1, int(np.ceil(np.linalg.norm(end - previous) / spacing)))
        for p in np.linspace(previous, end, n + 1)[1:]:
            points.append(p)
            ids.append(stroke_id)
        previous = np.array(end)

    points.append(previous.copy())
    ids.append(-1)
    for i, path in enumerate(paths):
        segment(np.r_[path[0], 0.025], -1)
        if touchdown_speed < 0.04:
            segment(np.r_[path[0], 0.005], -1, 0.0008)
        segment(np.r_[path[0], -0.001], -1, touchdown_speed * 0.02)
        for j, point in enumerate(path):
            depth = 0.001 + 0.003 * max(0.0, np.sin(np.pi * j / (len(path) - 1))) ** 0.7
            segment(np.r_[point, -depth], i)
        segment(np.r_[path[-1], 0.025], -1, 0.0008)
    points, ids = np.asarray(points), np.asarray(ids)
    points.flags.writeable = ids.flags.writeable = False
    return points, ids
