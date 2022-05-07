import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (AmbientLight, DirectionalLight, Texture,
                          TextureStage, WindowProperties)

DEFAULT_TRAJECTORIES = np.array(
    [
        [[i / 100, 0, 0], [-i / 100, 0, 0], [i / 100, i / 100, i / 100]]
        for i in range(1000)
    ]
)


class Display3d(ShowBase):
    def __init__(
        self,
        trajectories=DEFAULT_TRAJECTORIES,
        camera_position=[0, 150, 0],
        object_scale=0.5,
        frame_rate=80,
    ):
        ShowBase.__init__(self)
        self.setBackgroundColor(0, 0, 0)
        # the Y position is the opposite as the one specified below, don't know why...
        self.trackball.node().set_pos(*camera_position)
        self.trajectories = trajectories
        self.star_list = []
        # texture = self.loader.loadTexture("static.jpg")
        # texture.setWrapU(Texture.WM_repeat)
        # texture.setWrapV(Texture.WM_repeat)
        p0 = trajectories[0]
        color_list = (p0 - p0.min()) / (p0.max() - p0.min())
        for a, b, c in color_list:
            star = self.loader.loadModel("ball")
            # star.setTexture(texture, 16)
            star.setColor(a, b, c, 1)
            star.setScale(object_scale)
            star.reparentTo(self.render)
            self.star_list.append(star)
        self.taskMgr.add(self.animate_star_list, "AnimateStarList")
        self.frame_rate = frame_rate
        dlight = DirectionalLight("dlight")
        alight = AmbientLight("alight")
        dlnp = self.render.attachNewNode(dlight)
        alnp = self.render.attachNewNode(alight)
        dlight.setColor((0.8, 0.8, 0.6, 1))
        alight.setColor((0.5, 0.5, 0.5, 1))
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        self.render.setLight(alnp)

    def set_position(self, step):
        for star, position in zip(self.star_list, self.trajectories[step]):
            star.setPos(*position)

    def animate_star_list(self, task):
        step = int(task.time * self.frame_rate) % self.trajectories.shape[0]
        self.set_position(step)
        x, y, z = self.camera.getPos()
        if hasattr(self, "title"):
            self.title.destroy()
        # v = [self.trackball.node().isInView(x.getPos()) for x in self.star_list]
        # n_visible = sum(v)
        self.title = OnscreenText(
            text=f"{step}|{x:.2f}:{y:.2f}:{z:.2f}",
            fg=(1, 1, 0, 1),
            pos=(0, -1),
            scale=0.1,
        )
        return Task.cont
