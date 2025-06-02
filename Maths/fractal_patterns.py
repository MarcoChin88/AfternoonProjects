import random
from functools import cache

import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def polar_to_cartesian(theta: float, r: float) -> tuple[float, float]:
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


def cartesian_to_polar(x: float, y: float) -> tuple[float, float]:
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


@cache
def get_n_polygon_verts(
        n: int,
        r: float = 1,
        theta_offset: float = math.pi / 2,
) -> tuple[list[float], list[float]]:
    d_theta = math.pi * 2 / n

    thetas: list[float] = [
        d_theta * i + theta_offset
        for i in range(n)
    ]

    cartesian_verts = [
        polar_to_cartesian(theta=theta, r=r)
        for theta in thetas
    ]
    cartesian_verts.append(cartesian_verts[0])

    X, Y = zip(*cartesian_verts)
    return X, Y


def plot_polygon(
        fig,
        ax,
        n_verts: int = 7,
        n_pts: int = int(1e4),
        fractal_distance: float = 0.692
):
    ax.cla()

    # Polygon
    ax.set_title(f"{n_verts}-gon")
    X, Y = get_n_polygon_verts(n=n_verts)
    ax.plot(X, Y)

    # Dots
    verts = list(zip(X, Y))
    dots = [(0, 0)]
    for _ in range(n_pts):
        x_dot, y_dot = dots[-1]
        x_vert, y_vert = random.choice(verts)

        dx = x_vert - x_dot
        dy = y_vert - y_dot

        dx *= fractal_distance
        dy *= fractal_distance

        next_dot = (x_dot + dx, y_dot + dy)
        dots.append(next_dot)
    X_dot, Y_dot = zip(*dots[1:])
    ax.scatter(
        X_dot,
        Y_dot,
        s=2
    )

    # Axes settings
    padding = 0.2
    ax.set_xlim(min(X) - padding, max(X) + padding)
    ax.set_ylim(min(Y) - padding, max(Y) + padding)
    ax.axis('off')
    ax.autoscale(enable=False)
    ax.set_aspect('equal')

    fig.canvas.draw_idle()


def add_slider(
        fig,
        label: str,
        init_val: float,
        min_val: float,
        max_val: float,
        val_step: float,
        width: float,
        bottom: float,
        height: float = 0.03
):
    left = (1 - width) / 2

    slider_ax = fig.add_axes([left, bottom, width, height])
    return Slider(
        ax=slider_ax,
        label=label,
        valmin=min_val,
        valmax=max_val,
        valstep=val_step,
        valinit=init_val
    )


def plot_fractrals(
):
    fig, ax = plt.subplots(
        figsize=(14, 8),
    )
    plt.tight_layout()

    # Verts slider
    fig.subplots_adjust(bottom=0.25)
    vert_slider = add_slider(
        fig=fig,
        label='N Verts',
        init_val=7, min_val=3, max_val=30, val_step=1,
        width=0.75, bottom=0.1
    )
    distance_slider = add_slider(
        fig=fig,
        label='Fractal Distance',
        init_val=0.692, min_val=0.1, max_val=0.99, val_step=0.001,
        width=0.75, bottom=0.2
    )

    def update(val):
        plot_polygon(
            fig=fig,
            ax=ax,
            n_verts=vert_slider.val,
            fractal_distance=distance_slider.val
        )

    vert_slider.on_changed(update)
    distance_slider.on_changed(update)

    # Plot polygon
    plot_polygon(
        fig=fig,
        ax=ax,
        n_verts=vert_slider.val,
        fractal_distance=distance_slider.val
    )
    plt.show()


def main():
    # for n in range(3, 10):
    plot_fractrals()


if __name__ == "__main__":
    main()
