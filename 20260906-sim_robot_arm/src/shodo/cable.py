"""Distributed elastic rods using MuJoCo's first-party cable plugin.

The rods are representative bundles, not individual hairs. Material parameters are
uncalibrated. Native constraints supply friction and force coupling to the flange.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from shodo.brush import Brush


def add_cables(root, tool, config):
    extension = ET.SubElement(root, "extension")
    ET.SubElement(extension, "plugin", plugin="mujoco.elasticity.cable")
    offsets = Brush(config).offsets
    for i, xy in enumerate(offsets):
        direction = np.r_[-xy * 0.002, 0.03]
        direction /= np.linalg.norm(direction)
        quaternion = np.zeros(4)
        mujoco.mju_quatZ2Vec(quaternion, direction)
        x_to_z = np.array([np.sqrt(0.5), 0, -np.sqrt(0.5), 0])
        combined = np.zeros(4)
        mujoco.mju_mulQuat(combined, quaternion, x_to_z)
        body = ET.SubElement(
            tool,
            "body",
            name=f"bristle_root_{i}",
            pos=" ".join(map(str, [xy[0] * config.radius, xy[1] * config.radius, 0.13])),
            quat=" ".join(map(str, combined)),
        )
        geometry = {"curve": "s", "count": f"{config.rod_segments + 1} 1 1", "size": ".03"}
        if config.rod_tip_offset:
            s = np.linspace(0, 1, config.rod_segments + 1)
            vertices = np.c_[0.03 * s, config.rod_tip_offset * s * s, np.zeros_like(s)]
            vertices *= 0.03 / np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()
            geometry = {"vertex": " ".join(map(str, vertices.ravel()))}
        cable = ET.SubElement(
            body,
            "composite",
            prefix=f"bristle{i}_",
            type="cable",
            initial="none",
            **geometry,
        )
        plugin = ET.SubElement(cable, "plugin", plugin="mujoco.elasticity.cable")
        ET.SubElement(plugin, "config", key="bend", value=str(config.young_modulus))
        ET.SubElement(plugin, "config", key="twist", value=str(config.shear_modulus))
        ET.SubElement(cable, "joint", kind="main", damping=str(config.rod_damping))
        ET.SubElement(
            cable,
            "geom",
            type="capsule",
            size=str(config.rod_radius),
            density=str(config.rod_density),
            contype="2",
            conaffinity="1",
            condim="3",
            friction=f"{config.friction} .005 .0001",
            solref=".0004 1",
            solimp=".99 .999 .0001",
            priority="1",
            rgba=".10 .065 .025 1",
            group="0",
        )


class CableBrush(Brush):
    """Adapt native rod contact samples to the common brush/ink interface."""

    native = True

    def __init__(self, robot, config):
        super().__init__(config)
        self.robot = robot
        self.paper_geom = robot.model.geom("paper").id
        self.geom_bundle = {}
        for i in range(robot.model.ngeom):
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("bristle"):
                self.geom_bundle[i] = int(name.split("_")[0][7:])
        self.first_sites = [
            robot.model.site(f"bristle{i}_S_first").id for i in range(config.bundles)
        ]
        self.last_sites = [robot.model.site(f"bristle{i}_S_last").id for i in range(config.bundles)]
        self._ink_positions = np.empty((0, 3))
        self._ink_loads = np.empty(0)
        self.max_penetration = 0.0
        self._contact_force = np.zeros(6)
        self._zero = np.zeros(3)

    def update(self, tip, rotation, dt, paper_z=0.0):
        del rotation, dt, paper_z
        robot = self.robot
        self.roots[:] = robot.data.site_xpos[self.first_sites]
        self.contact[:] = robot.data.site_xpos[self.last_sites]
        self.force[:] = 0
        self.torque[:] = 0
        self.normal[:] = 0
        self.tangent[:] = 0
        points, loads = [], []
        local = self._contact_force
        for index in range(robot.data.ncon):
            contact = robot.data.contact[index]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.paper_geom and g2 in self.geom_bundle:
                bundle, sign = self.geom_bundle[g2], 1
            elif g2 == self.paper_geom and g1 in self.geom_bundle:
                bundle, sign = self.geom_bundle[g1], -1
            else:
                continue
            mujoco.mj_contactForce(robot.model, robot.data, index, local)
            force = sign * (contact.frame.reshape(3, 3).T @ local[:3])
            self.force += force
            offset = contact.pos - tip
            self.torque[0] += offset[1] * force[2] - offset[2] * force[1]
            self.torque[1] += offset[2] * force[0] - offset[0] * force[2]
            self.torque[2] += offset[0] * force[1] - offset[1] * force[0]
            self.normal[bundle] += max(0.0, force[2])
            self.tangent[bundle] += force[:2]
            self.max_penetration = max(self.max_penetration, -contact.dist)
            if force[2] > 1e-7:
                points.append(contact.pos.copy())
                loads.append(force[2])
        self.touching[:] = self.normal > 1e-7
        self._ink_positions = np.asarray(points).reshape(-1, 3)
        self._ink_loads = np.asarray(loads)
        return self._zero, self._zero

    @property
    def ink_positions(self):
        return self._ink_positions

    @property
    def ink_loads(self):
        return self._ink_loads
