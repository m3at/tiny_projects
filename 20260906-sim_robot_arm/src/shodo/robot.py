"""Pinned Menagerie model and torque-controlled six-dimensional brush pose."""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
ROOT = Path("data/robot/panda")
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_emika_panda"
DOWN = np.diag([1.0, -1.0, -1.0])
HOME = np.array([0, -0.45, 0, -2.2, 0, 1.8, -2.3562])


def fetch_robot():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "provenance.json"
    previous = json.loads(lock.read_text()).get("sha256", {}) if lock.exists() else {}

    def download(name):
        path = ROOT / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == previous.get(name):
            return name, previous[name]
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return name, hashlib.sha256(content).hexdigest()

    hashes = dict(download(name) for name in ("panda_nohand.xml", "LICENSE", "README.md"))
    tree = ET.parse(ROOT / "panda_nohand.xml")
    names = {"assets/" + node.attrib["file"] for node in tree.iter("mesh") if "file" in node.attrib}
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes.update(pool.map(download, sorted(names)))
    lock.write_text(
        json.dumps(
            {"revision": REVISION, "url": BASE, "license": "Apache-2.0", "sha256": hashes}, indent=2
        )
        + "\n"
    )
    print(f"Verified Panda model and {len(names)} meshes at {REVISION[:12]}", flush=True)


@lru_cache(maxsize=4)
def model_xml(timestep):
    if not (ROOT / "panda_nohand.xml").exists():
        raise FileNotFoundError("Panda model missing; run make data")
    tree = ET.parse(ROOT / "panda_nohand.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str((ROOT / "assets").resolve()))
    root.find("option").set("timestep", str(timestep))
    root.remove(root.find("keyframe"))
    actuator = root.find("actuator")
    actuator.clear()
    for i, limit in enumerate([87, 87, 87, 87, 12, 12, 12], 1):
        ET.SubElement(actuator, "motor", joint=f"joint{i}", ctrlrange=f"{-limit} {limit}")
    attachment = root.find(".//body[@name='attachment']")
    tool = ET.SubElement(attachment, "body", name="brush")
    ET.SubElement(
        tool,
        "geom",
        name="handle",
        type="capsule",
        fromto="0 0 0 0 0 0.13",
        size="0.006",
        mass="0.04",
        rgba="0.52 0.28 0.09 1",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(tool, "site", name="tip", pos="0 0 0.16", size="0.001")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0.5 0 -0.035",
        size="0.3 0.28 0.03",
        rgba="0.25 0.19 0.14 1",
    )
    ET.SubElement(
        root.find("asset"),
        "texture",
        name="ink",
        type="2d",
        builtin="flat",
        rgb1="0.98 0.97 0.93",
        width="256",
        height="256",
    )
    ET.SubElement(
        root.find("asset"), "material", name="inkpaper", texture="ink", texuniform="false"
    )
    ET.SubElement(
        world,
        "geom",
        name="paper",
        type="box",
        pos="0.5 0 -0.0025",
        size="0.105 0.105 0.0025",
        material="inkpaper",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    return ET.tostring(root, encoding="unicode")


class Panda:
    def __init__(self, timestep=0.002):
        self.model = mujoco.MjModel.from_xml_string(model_xml(timestep))
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.site = self.model.site("tip").id
        self.body = self.model.body("brush").id
        self.jac = np.zeros((6, 7))
        self.mass = np.zeros((7, 7))
        self.q_target = HOME.copy()

    @property
    def tip(self):
        return self.data.site_xpos[self.site]

    @property
    def rotation(self):
        return self.data.site_xmat[self.site].reshape(3, 3)

    def inverse(self, position, rotvec, iterations=5):
        data = self.scratch
        data.qpos[:] = self.q_target
        rotation = Rotation.from_rotvec(rotvec).as_matrix() @ DOWN
        for _ in range(iterations):
            mujoco.mj_forward(self.model, data)
            error = np.r_[
                position - data.site_xpos[self.site],
                Rotation.from_matrix(
                    rotation @ data.site_xmat[self.site].reshape(3, 3).T
                ).as_rotvec(),
            ]
            if np.linalg.norm(error) < 1e-6:
                break
            mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], self.site)
            delta = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + 1e-5 * np.eye(6), error)
            data.qpos[:] = np.clip(
                data.qpos + np.clip(delta, -0.15, 0.15),
                self.model.jnt_range[:, 0] + 0.01,
                self.model.jnt_range[:, 1] - 0.01,
            )
        return data.qpos.copy()

    def reset(self, position, rotvec):
        mujoco.mj_resetData(self.model, self.data)
        self.q_target = HOME.copy()
        self.q_target = self.inverse(position, rotvec, iterations=150)
        self.data.qpos[:] = self.q_target
        mujoco.mj_forward(self.model, self.data)
        if np.linalg.norm(self.tip - position) > 0.001:
            raise ValueError(f"Panda reset IK failed: {self.tip - position}")

    def step(self, force, torque):
        # Inverse-dynamics joint impedance, with actual tool load and torque limits.
        self.data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(
            self.model, self.data, force, torque, self.tip, self.body, self.data.qfrc_applied
        )
        mujoco.mj_fullM(self.model, self.data, self.mass)
        acceleration = 900 * (self.q_target - self.data.qpos) - 60 * self.data.qvel
        self.data.ctrl[:] = np.clip(
            self.mass @ acceleration + self.data.qfrc_bias,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
