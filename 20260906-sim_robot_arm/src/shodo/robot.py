"""Torque-driven B601-RS with a rigid brush and bounded MIT joint commands."""

import copy
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from shodo.actuation import JointGovernor
from shodo.config import BrushConfig, RobotConfig
from shodo.rebot import REVISION, arm_xml, fetch_robot  # noqa: F401

DOWN = np.diag([1.0, -1.0, -1.0])
HOME = np.array([0, 1.5, 1.5, 0, 0, 0])


@lru_cache(maxsize=4)
def model_xml(timestep, brush_config, robot_config=None):
    robot_config = robot_config or RobotConfig()
    root, attachment = arm_xml(timestep, robot_config)
    tool = ET.SubElement(
        attachment,
        "body",
        name="brush",
        pos=" ".join(map(str, robot_config.mount_xyz)),
        quat=" ".join(
            map(str, Rotation.from_euler("xyz", robot_config.mount_rpy).as_quat(scalar_first=True))
        ),
    )
    ET.SubElement(
        tool,
        "geom",
        name="handle",
        type="capsule",
        fromto="0 0 0 0 0 0.13",
        size="0.006",
        mass=str(robot_config.handle_mass),
        rgba="0.52 0.28 0.09 1",
        # Handle=4, rods=2, scene/arm=1. The clamp defines rod attachment;
        # unresolved embedded root geometry must not add segment-dependent contacts.
        contype="4",
        conaffinity="1",
    )
    ET.SubElement(tool, "site", name="tip", pos="0 0 0.16", size="0.001")
    if brush_config.backend == "cable":
        from shodo.cable import add_cables

        add_cables(root, tool, brush_config)
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos="0.4 0 -0.035",
        size="0.45 0.28 0.03",
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


@lru_cache(maxsize=4)
def _compiled_model(timestep, brush_config, robot_config):
    # Mesh compilation is expensive; the cached template is never handed to an environment.
    return mujoco.MjModel.from_xml_string(model_xml(timestep, brush_config, robot_config))


class RebotArm:
    def __init__(self, timestep=0.002, brush_config=None, robot_config=None):
        self.config = robot_config or RobotConfig()
        self.model = copy.copy(
            _compiled_model(timestep, brush_config or BrushConfig(), self.config)
        )
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.site = self.model.site("tip").id
        self.body = self.model.body("brush").id
        self.jac = np.zeros((6, self.model.nv))
        self.q_target = HOME.copy()
        self.q_command = HOME.copy()
        self.v_command = np.zeros(6)
        self.protected_geoms = {
            i
            for i in range(self.model.ngeom)
            if "_collision_" in self.model.geom(i).name or self.model.geom(i).name == "handle"
        }
        self.governor = JointGovernor(self.model.jnt_range[:6], self.config)

    @property
    def tip(self):
        return self.data.site_xpos[self.site]

    @property
    def rotation(self):
        return self.data.site_xmat[self.site].reshape(3, 3)

    def inverse(self, position, rotvec, iterations=5):
        data = self.scratch
        data.qpos[:6] = self.q_target
        rotation = Rotation.from_rotvec(rotvec).as_matrix() @ DOWN
        for _ in range(iterations):
            # IK needs only site transforms and motion axes, not contacts or dynamics.
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            error = np.r_[
                position - data.site_xpos[self.site],
                Rotation.from_matrix(
                    rotation @ data.site_xmat[self.site].reshape(3, 3).T
                ).as_rotvec(),
            ]
            if np.linalg.norm(error) < 1e-6:
                break
            mujoco.mj_jacSite(self.model, data, self.jac[:3], self.jac[3:], self.site)
            jac = self.jac[:, :6]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
            data.qpos[:6] = np.clip(
                data.qpos[:6] + np.clip(delta, -0.15, 0.15),
                self.model.jnt_range[:6, 0] + 0.01,
                self.model.jnt_range[:6, 1] - 0.01,
            )
        return data.qpos[:6].copy()

    def reset(self, position, rotvec):
        mujoco.mj_resetData(self.model, self.data)
        self.q_target = HOME.copy()
        self.q_target = self.inverse(position, rotvec, iterations=150)
        self.q_command = self.q_target.copy()
        self.v_command[:] = 0
        self.governor.reset(self.q_target, 0.0)
        self.data.qpos[:6] = self.q_target
        mujoco.mj_forward(self.model, self.data)
        if (
            np.linalg.norm(self.tip - position) > 0.001
            or np.linalg.norm(
                Rotation.from_matrix(
                    self.rotation @ (Rotation.from_rotvec(rotvec).as_matrix() @ DOWN).T
                ).as_rotvec()
            )
            > 0.01
        ):
            raise ValueError(f"B601-RS reset IK failed: {self.tip - position}")

    def set_target(self, target):
        self.governor.submit(target, self.data.time, self.data.time)
        self.q_target = np.asarray(target).copy()

    def step(self, force, torque):
        # External brush reaction couples into arm dynamics at the actual tip.
        self.data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(
            self.model, self.data, force, torque, self.tip, self.body, self.data.qfrc_applied
        )
        cfg = self.config
        self.q_command, self.v_command = self.governor.advance(
            self.data.time + self.model.opt.timestep
        )
        # MIT law: kp(q_cmd-q) + kd(v_cmd-v) + gravity/Coriolis feedforward.
        # Affine actuator applies position/velocity feedback and total force limits.
        self.data.ctrl[:] = (
            np.asarray(cfg.kp) * self.q_command
            + np.asarray(cfg.kd) * self.v_command
            + self.data.qfrc_bias[:6]
        )
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_positions(self):
        """Fixed observation slots: six measured arm joints and a reserved zero."""
        return np.r_[self.data.qpos[:6], 0.0]

    @property
    def joint_velocities(self):
        return np.r_[self.data.qvel[:6], 0.0]

    def forbidden_contacts(self, data=None):
        data = self.data if data is None else data
        return sum(
            c.dist < 0 and (c.geom1 in self.protected_geoms or c.geom2 in self.protected_geoms)
            for c in data.contact
        )
