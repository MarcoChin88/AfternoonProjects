import os
import time
from enum import Enum
from functools import partial
from random import choice

import numpy as np


os.environ['TERM'] = 'xterm'


def update_rotation(rotation_angles: np.ndarray, rotation_rates: np.ndarray, dt: float):
    return (rotation_angles + rotation_rates * dt) % 360


def rotate_pt(pos: np.ndarray, rot: np.ndarray):
    theta = np.radians(rot)

    c_x, c_y, c_z = np.cos(theta)
    s_x, s_y, s_z = np.sin(theta)

    R_x = np.array([
        [1, 0, 0],
        [0, c_x, -s_x],
        [0, s_x, c_x]
    ])

    R_y = np.array([
        [c_y, 0, s_y],
        [0, 1, 0],
        [-s_y, 0, c_y]
    ])

    R_z = np.array([
        [c_z, -s_z, 0],
        [s_z, c_z, 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x

    return pos @ R


class Torus():
    def __init__(
            self,
            r1: float,
            r2: float,
            position: np.ndarray = np.array([0, 0, 0]),
            rotation: np.ndarray = np.array([0, 0, 0]),
    ):
        self.r1: float = r1
        self.r2: float = r2

        self.position: np.ndarray = position
        self.rotation: np.ndarray = rotation
        # self.rotation_rates: np.ndarray = np.array([30, 45, 60])

    def rotate(
            self,
            rotation_rates: np.ndarray = np.array([30, 45, 60]),
            dt: float = 0.1
    ):
        while True:
            time.sleep(dt)
            self.rotation = update_rotation(self.rotation, rotation_rates, dt)

    def render_surface_pt(
            self,
            theta: float,
            phi: float,
            cam_distance: float = 1,
            fov: float = 45,
            screen_dims: np.ndarray = np.array([50, 50])
    ):
        c_theta, c_phi = np.cos(theta), np.cos(phi)
        s_theta, s_phi = np.sin(theta), np.sin(phi)

        t = (self.r1 + self.r2 * c_theta)
        x = t * c_phi
        y = t * s_phi
        z = self.r2 * s_theta

        pos3d = np.array([x, y, z])
        rotated_pos = rotate_pt(pos3d, self.rotation)

        k = np.max(screen_dims) / (2 * np.tan(np.radians(fov) / 2))

        x, y, z = rotated_pos
        persp_x, persp_y = k * x / (k + cam_distance), k * y / (k + cam_distance)

        width, height = screen_dims
        screen_coords = (
            int(width / 2 + persp_x),
            int(height / 2 - persp_y)
        )

        return screen_coords

    def render_surface(self, screen_dims=None):
        if screen_dims is None:
            screen_dims = [50, 50]

        screen = np.full(screen_dims, " ")
        for theta in range(360):
            for phi in range(360):
                coords = self.render_surface_pt(theta=theta, phi=phi)
                screen[coords] = "x"
                pass

        print(screen)
        p = "\n".join(["".join(_) for _ in screen])

        print("\033[H\033[J")
        print(p)

    def render_frame(self):
        dt = 0.1
        while True:
            self.render_surface()
            time.sleep(dt)


def main():
    z_cam = 10

    torus = Torus(0.7, 0.3)
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        torus.render_frame()


if __name__ == "__main__":
    main()
