from typing import *

import matplotlib
import numpy as np

class Cart:
    def __init__(self, x_center: float, width: float, height: float, wheel_radius: float):
        self.x_center = x_center
        self.width = width
        self.height = height
        self.wheel_radius = wheel_radius

    def plot(self, axis: matplotlib.axes.Axes):
        def circle(radius: float, centre: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            xs = np.linspace(centre[0] - radius, centre[0] + radius, 100)
            im = np.sqrt(np.clip(radius ** 2 - (xs - centre[0]) ** 2, 0, None))
            real = centre[1]
            upper_part, lower_part = real + im, real - im
            return upper_part, lower_part, xs

        # Bottom side
        axis.hlines(self.wheel_radius,
                    self.x_center - self.width / 2,
                    self.x_center + self.width / 2, colors='k')
        # Left side
        axis.vlines(self.x_center - self.width / 2,
                    self.wheel_radius,
                    self.wheel_radius + self.height, colors='k')
        # Top side
        axis.hlines(self.wheel_radius + self.height,
                    self.x_center - self.width / 2,
                    self.x_center + self.width / 2, colors='k')
        # Right side
        axis.vlines(self.x_center + self.width / 2,
                    self.wheel_radius,
                    self.wheel_radius + self.height, colors='k')

        # Left tire
        upper_left_part, lower_left_part, xs = circle(radius=self.wheel_radius,
                                                      centre=(self.x_center - self.width / 4, self.wheel_radius))
        axis.plot(xs, lower_left_part, color='k')

        # Right tire
        upper_right_part, lower_right_part, xs = circle(radius=self.wheel_radius,
                                                        centre=(self.x_center + self.width / 4, self.wheel_radius))
        axis.plot(xs, lower_right_part, color='k')
