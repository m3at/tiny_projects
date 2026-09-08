"""Pinned B601-RS description, converted without a ROS or Pinocchio dependency."""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from filelock import FileLock
from scipy.spatial.transform import Rotation

REVISION = "ce074041cd1c26f67ce74c2ca6fca9af22f8aee5"
ROOT = Path("data/robot/rebot")
PREFIX = "Rebot_Arm_description/RS/"
URDF = PREFIX + "urdf/ReBot_Arm_RS.urdf"
MANIFEST = Path(__file__).parent / "assets/rebot_manifest.json"
JOINTS = tuple(f"joint{i}" for i in range(1, 7))
ROBOT_CONTRACT = {
    "version": 1,
    "model": "reBot Arm B601-RS",
    "revision": REVISION,
    "joints": list(JOINTS),
    "joint_slots": "six URDF joints followed by one reserved zero; no gripper",
    "tool": "rigid wrist brush; vertical world axis; rotation actions suppressed",
}


def fetch_robot():
    ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(ROOT / ".fetch.lock", timeout=120):
        manifest = json.loads(MANIFEST.read_text())
        if manifest["revision"] != REVISION:
            raise ValueError("B601 manifest revision mismatch")

        def download(item):
            name, expected = item
            path = ROOT / name
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                return
            with urllib.request.urlopen(f"{manifest['url']}/{name}", timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"B601 checksum mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".pending")
            temporary.write_bytes(content)
            temporary.replace(path)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(download, manifest["sha256"].items()))
        (ROOT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified B601-RS: {len(manifest['sha256'])} files at {REVISION[:12]}", flush=True)


def _pose(origin):
    xyz = origin.get("xyz", "0 0 0")
    quat = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    return {"pos": xyz, "quat": " ".join(map(str, quat.as_quat(scalar_first=True)))}


@lru_cache(maxsize=1)
def source():
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["sha256"].items():
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError("B601-RS model missing; run make data")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"B601 source checksum mismatch: {name}; run make data")
    return ET.parse(ROOT / URDF).getroot()


def arm_xml(timestep, config):
    """Preserve URDF origins, axes, COM and full inertia; omit gripper descendants.

    Collision meshes are MuJoCo convex hulls of upstream collision geometry.
    Adjacent links are excluded by MuJoCo; other self/scene contacts remain active.
    """
    urdf = source()
    root = ET.Element("mujoco", model="rebot_b601_rs")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="auto")
    ET.SubElement(root, "option", timestep=str(timestep), integrator="implicitfast")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.3 -0.2 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    base = ET.SubElement(world, "body", name="base_link", pos=" ".join(map(str, config.base_xyz)))
    bodies = {"base_link": base}
    colors = {m.get("name"): m.find("color").get("rgba") for m in urdf.findall("material")}
    meshes = {}
    actuator = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", body1="base_link", body2="link1")
    for i, name in enumerate(JOINTS):
        joint = urdf.find(f"joint[@name='{name}']")
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        body = ET.SubElement(bodies[parent], "body", name=child, **_pose(joint.find("origin")))
        bodies[child] = body
        limit = joint.find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if i == 0:  # Wiki specifies +/-150 degrees; intersect with the wider URDF.
            lower, upper = max(lower, -np.deg2rad(150)), min(upper, np.deg2rad(150))
        ET.SubElement(
            body,
            "joint",
            name=name,
            type="hinge",
            axis=joint.find("axis").get("xyz"),
            range=f"{lower} {upper}",
            armature=str(config.armature),
        )
        # Affine MIT impedance. MuJoCo integrates velocity feedback implicitly;
        # actuator_force is the bounded physical torque, ctrl is the affine input.
        torque = config.torque_limits[i]
        ET.SubElement(
            actuator,
            "general",
            name=name,
            joint=name,
            gainprm="1",
            biastype="affine",
            biasprm=f"0 {-config.kp[i]} {-config.kd[i]}",
            forcerange=f"{-torque} {torque}",
            ctrllimited="false",
        )
    for name, body in bodies.items():
        link = urdf.find(f"link[@name='{name}']")
        inertial = link.find("inertial")
        inertia = inertial.find("inertia")
        # URDF inertia is in the inertial frame. Rotate before writing fullinertia.
        pose = _pose(inertial.find("origin"))
        q = np.fromstring(pose.pop("quat"), sep=" ")
        rotation = Rotation.from_quat(q, scalar_first=True).as_matrix()
        a = {k: float(v) for k, v in inertia.attrib.items()}
        tensor = np.array(
            [
                [a["ixx"], a["ixy"], a["ixz"]],
                [a["ixy"], a["iyy"], a["iyz"]],
                [a["ixz"], a["iyz"], a["izz"]],
            ]
        )
        tensor = rotation @ tensor @ rotation.T
        values = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        ET.SubElement(
            body,
            "inertial",
            **pose,
            mass=inertial.find("mass").get("value"),
            fullinertia=" ".join(map(str, values)),
        )
        for kind in ("visual", "collision"):
            for index, geometry in enumerate(link.findall(kind)):
                mesh = geometry.find("geometry/mesh")
                filename = PREFIX + mesh.get("filename").removeprefix("../")
                if filename not in meshes:
                    meshname = f"mesh{len(meshes)}"
                    meshes[filename] = meshname
                    ET.SubElement(
                        assets, "mesh", name=meshname, file=str((ROOT / filename).resolve())
                    )
                material = geometry.find("material")
                rgba = (
                    colors.get(material.get("name"), "0.2 0.2 0.2 1")
                    if material is not None
                    else "0.2 0.2 0.2 1"
                )
                visual = kind == "visual"
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{name}_{kind}_{index}",
                    type="mesh",
                    mesh=meshes[filename],
                    **_pose(geometry.find("origin")),
                    rgba=rgba,
                    group="1" if visual else "3",
                    contype="0" if visual else "1",
                    conaffinity="0" if visual else "7",
                )
    return root, bodies["link6"]
