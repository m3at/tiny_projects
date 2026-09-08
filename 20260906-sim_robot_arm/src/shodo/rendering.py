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

    def camera_frame(self, env):
        """Raw perspective RGB, with no diagnostics or privileged paper inset."""
        self.gl.make_current()
        paper = env.paper.image()
        # MuJoCo's box top maps texture rows toward -world Y, like Paper.image().
        # Only the framebuffer readback below needs an OpenGL bottom-up flip.
        texture = np.asarray(paper.resize((256, 256)))
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
        return self.rgb[::-1].copy()

    def frame(self, env):
        rgb = self.camera_frame(env)
        paper = env.paper.image()
        canvas = Image.new("RGB", (960, 480), (32, 37, 43))
        canvas.paste(Image.fromarray(rgb), (0, 0))
        canvas.paste(paper.resize((300, 300)), (650, 95))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 18), "reBot B601-RS / VERTICAL BRUSH", fill="white")
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

    def camera_metadata(self):
        """Pinhole calibration of the last raw monoscopic free-camera frame.

        Camera coordinates are X right, Y down, Z forward. Integer pixel centers
        use the same top-row-first convention as camera_frame().
        """
        camera = mujoco.mjv_averageCamera(*self.scene.camera)
        forward = np.asarray(camera.forward, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera.up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        transform = np.eye(4)
        transform[:3, :3] = np.stack([right, down, forward])
        transform[:3, 3] = -transform[:3, :3] @ camera.pos
        height, width = self.rgb.shape[:2]
        focal = height * camera.frustum_near / (camera.frustum_top - camera.frustum_bottom)
        return {
            "intrinsics": [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
            "world_to_camera": transform.tolist(),
            "pixel_convention": "integer pixel centers; top-left=(0,0); X right, Y down",
            "camera_axes": "X right, Y down, Z forward",
            "distortion": "none (synthetic perspective renderer)",
        }

    def close(self):
        self.gl.make_current()
        self.context.free()
        self.gl.free()
