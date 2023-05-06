import inspect
import math
import os
import random
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 20.0,
}


def eulertoq(euler):
    phi, theta, psi = euler
    qx = np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qy = np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2) - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2)
    qz = np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2) + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
    qw = np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2) - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
    return np.array([qx, qy, qz, qw])


class BallEnv(MujocoEnv, utils.EzPickle):
    FILE = "ball.xml"
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 10,
    }

    def __init__(self,
                 ctrl_cost_coeff=1e-4,
                 vision=False,
                 width=64,
                 height=64,
                 render_mode=None,
                 *args, **kwargs):
        self.render_mode = render_mode
        self.height = height
        self.width = width
        self.vision = vision
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.args = args
        self.kwargs = kwargs
        
        self.frame_skip = 10
        obs_shape = 6
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)
        
        self.texture_path = os.path.dirname(__file__) + "/models/texture/random_gen.png"
        self.rebuild_model()
        
    def step(self, a):
        xposbefore = self.data.qpos.flat.copy()[0]
        
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos.flat.copy()[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - self.ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self.get_current_obs()
        return ob, reward, False, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def get_current_obs(self):
        gyro = self.data.sensor("gyro").data
        accel = self.data.sensor("accel").data
        return np.concatenate([gyro, accel], dtype=np.float32)
    
    def set_random_position(self):
        L = 5
        self.init_qpos[1] = self.np_random.uniform(-L, L)

        # random_angle = self.np_random.uniform(0, 2 * np.pi)
        # q = eulertoq(np.array([0, 0, random_angle]))
        # self.init_qpos[3:3 + 4] = q

    def reset_model(self):
        self.rebuild_model()
        self.set_random_position()
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self.get_current_obs()
    
    def rebuild_model(self):
        
        model_cls = self.__class__
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        MODEL_DIR = os.path.join(p.parent, "models", self.FILE)
        
        tree = ET.parse(MODEL_DIR)
        root = tree.getroot()
        
        # Random friction
        default_elem = tree.find(".//default").find(".//geom")
        default_elem.set("friction", f"{random.uniform(0.1, 1.5)} 0.5 0.5")

        # Random lighting conditions
        light_elem = root.find(".//light[@name='light']")
        light_elem.set('directional', "true")
        light_elem.set("exponent", f"{random.uniform(0, 10)}")
        light_elem.set("attenuation", f"{random.uniform(0, 10)} 0 0")
        light_elem.set("diffuse", f"{random.uniform(0, 2)} {random.uniform(0, 2)} {random.uniform(0, 2)}")
        light_elem.set("specular", f"{random.uniform(0.1, 1)} {random.uniform(0.1, 1)} {random.uniform(0.1, 1)}")   # diffuse parameter is critically important for illumination
        light_elem.set("pos", f"{random.uniform(-3, 3)} {random.uniform(-3, 3)} {random.uniform(0.3, 1)}")
        light_elem.set("dir", f"{0} {0} {random.uniform(-3.0, -0.1)}")

        # Random coloring the agent body
        agent_color = f"{random.random()} {random.random()} {random.random()} 1"
        sphere_elem = root.find(".//geom[@type='sphere']")
        sphere_elem.set('rgba', agent_color)  # Replace '1 0 0 1' with the desired RGBA value
        
        # Random sizing the agent body
        agent_size = random.uniform(0.3, 0.5)
        sphere_elem.set("size", f"{agent_size}")
        
        # Random perturbation of imu position
        imu_elem = root.find(".//site[@name='imu']")
        imu_elem.set("pos", f"{random.uniform(-0.1, 0.1)} {random.uniform(-0.1, 0.1)} {random.uniform(-0.1, 0.1)}")
        
        # Randomly attaching weights on the agent
        torso = tree.find(".//body[@name='agent_body']")
        
        new_body = ET.SubElement(
            torso, "body", dict(
                name="attached_weight",
            )
        )
        
        for i in range(random.randint(3, 5)):
            x, y, z = self.generate_random_point_in_sphere()
            ET.SubElement(
                new_body, "geom", dict(
                    name=f"weight{i}",
                    type="sphere",
                    size=f"{0.1 * random.random() + 0.02}",
                    pos=f"{agent_size * x} {agent_size * y} {agent_size * z}",
                    rgba=f"{random.random()} {random.random()} {random.random()} 1",
                )
            )

        # create new texture
        self.generate_new_texture()
        
        asset = tree.find(".//asset")
        ET.SubElement(
            asset, "texture", dict(
                name="floor_texture",
                type="2d",
                file=self.texture_path,
                width="10",
                height="10",
            )
        )
        ET.SubElement(
            asset, "material", dict(
                name="grass",
                texture="floor_texture",
                texrepeat="1 1"
            )
        )
        
        asset.find("material").set("texture", "floor_texture")
        
        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml") as tmpfile:
            file_path = tmpfile.name
            tree.write(file_path)
            
            utils.EzPickle.__init__(
                self,
                file_path,
                self.ctrl_cost_coeff,  # gym has 1 here!
                self.vision,
                self.width,
                self.height,
                **self.kwargs
            )
            
            if hasattr(self, "mujoco_renderer"):
                self.mujoco_renderer.close()
            
            MujocoEnv.__init__(
                self,
                file_path,
                self.frame_skip,
                self.observation_space,
                render_mode=self.render_mode,
                width=self.width if self.vision else 480,
                height=self.height if self.vision else 480,
                default_camera_config=DEFAULT_CAMERA_CONFIG
            )

    def generate_new_texture(self):
        """
        Generate a random texture
        :return:
        """
        height = 255
        width = 255
        num_rectangles = 100
        
        image = Image.new("RGB", (height, width))
        pixels = []
        for _ in range(height):
            row = []
            for _ in range(width):
                red = random.randint(0, 255)
                green = random.randint(0, 255)
                blue = random.randint(0, 255)
                row.append((red, green, blue))
            pixels.append(row)
        image.putdata([pixel for row in pixels for pixel in row])
        
        width, height = image.size
        for _ in range(num_rectangles):
            x = random.randint(-100,  width - 1)
            y = random.randint(-100, height - 1)
            rect_width = random.randint(0, 100)
            rect_height = random.randint(0, 100)
            angle = random.randint(0, 360)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            rect_image = Image.new("RGBA", (rect_width, rect_height))
            rect_draw = ImageDraw.Draw(rect_image)
            rect_draw.rectangle([(0, 0), (rect_width, rect_height)], fill=color, outline=color)
            rotated_image = rect_image.rotate(angle, expand=True)
            image.paste(rotated_image, (x, y), mask=rotated_image)
    
        image.save(self.texture_path)
    
    def generate_random_point_in_sphere(self):
        # Generate random spherical coordinates
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.random() - 1)
        
        # Convert spherical coordinates to Cartesian coordinates
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        
        return x, y, z
