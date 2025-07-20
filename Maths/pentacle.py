from pprint import pprint

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def polar_to_cartesian(theta: float, r: float) -> tuple[float, float]:
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


def plot_pentacle():
    fig, ax = plt.subplots()

    lim = 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    unit_circle = Circle(
        (0, 0),
        1,
        fill=False
    )

    ax.add_patch(unit_circle)

    num_pts = 5
    theta = 2 * math.pi / num_pts
    polar_pts = [(theta * i + math.pi / 2, 1) for i in range(num_pts)]
    cartesian_pts = [polar_to_cartesian(*_) for _ in polar_pts]
    X, Y = zip(*cartesian_pts)
    plt.scatter(X, Y)

    lines = [
        (cartesian_pts[0], cartesian_pts[2]),
        (cartesian_pts[2], cartesian_pts[4]),
        (cartesian_pts[4], cartesian_pts[1]),
        (cartesian_pts[1], cartesian_pts[3]),
        (cartesian_pts[3], cartesian_pts[0]),
    ]
    pprint(lines)

    for line in lines:
        ax.plot(*zip(*line))

    plt.show()


def main():
    plot_pentacle()


if __name__ == "__main__":
    main()
