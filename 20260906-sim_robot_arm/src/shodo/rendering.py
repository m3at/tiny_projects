"""Offscreen MuJoCo rendering with live paper texture and bending bristle bundles."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw


class Renderer:
    def __init__(self, model):
        self.model = model
        self.gl = mujoco.GLContext(640, 480)
        self.gl.make_current()
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        self.scene = mujoco.MjvScene(model, maxgeom=max(2000, model.ngeom + 64))
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.3, 0, 0.22]
        self.camera.distance = 1.35
        self.camera.azimuth = 135
        self.camera.elevation = -35
        self.rect = mujoco.MjrRect(0, 0, 640, 480)
        self.rgb = np.empty((480, 640, 3), dtype=np.uint8)

    def frame(self, env):
        self.gl.make_current()
        paper = env.paper.image()
        texture = np.asarray(paper.resize((256, 256)))[::-1]
        texid = self.model.texture("ink").id
        start = self.model.tex_adr[texid]
        self.model.tex_data[start : start + texture.size] = texture.ravel()
        mujoco.mjr_uploadTexture(self.model, self.context, texid)
        mujoco.mjv_updateScene(
            self.model,
            env.data,
            self.options,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        roots = [] if env.brush.native else env.brush.roots
        contacts = [] if env.brush.native else env.brush.contact
        for root, contact in zip(roots, contacts, strict=True):
            geom = self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.09, 0.06, 0.03, 1.0]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0004, root, contact)
            self.scene.ngeom += 1
        mujoco.mjr_render(self.rect, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb, None, self.rect, self.context)
        canvas = Image.new("RGB", (960, 480), (32, 37, 43))
        canvas.paste(Image.fromarray(self.rgb[::-1]), (0, 0))
        canvas.paste(paper.resize((300, 300)), (650, 95))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 18), "FRANKA PANDA / ELASTIC BRUSH", fill="white")
        draw.text((650, 65), "WATER + PIGMENT / FIBROUS PAPER", fill="white")
        draw.text(
            (650, 410),
            f"t = {env.data.time:.2f}s | force = {env.brush.force[2]:.3f} N",
            fill="white",
        )
        draw.text(
            (650, 432),
            f"Contact bundles: {env.brush.touching.sum()}/{env.brush.config.bundles}",
            fill="white",
        )
        return np.asarray(canvas)

    def close(self):
        self.gl.make_current()
        self.context.free()
        self.gl.free()
