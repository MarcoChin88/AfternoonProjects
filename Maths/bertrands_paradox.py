"""
Inspired by https://www.youtube.com/watch?v=mZBwsm6B280
Find probability random chord in circle is greater than the inscribed equalateral triangle's side length.
Explore different ways of generating random chords.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias

import tabulate
from math import sqrt, pi, cos, sin, radians, atan2
from random import uniform

from matplotlib import pyplot as plt

RADIUS: float = 1.0
NUM_LINES: int = 1000

Pt: TypeAlias = tuple[float, float]


def get_random_theta_radians() -> float:
    return uniform(0, 2 * pi)


def get_quadratic_roots(a: float, b: float, c: float) -> Pt:
    """
    Returns roots of quadratic via quadratic formula
    """
    discriminant: float = sqrt(b ** 2 - 4 * a * c)

    _2a: float = 2 * a
    return (-b + discriminant) / _2a, (-b - discriminant) / _2a


def cartesian_to_polar(x: float, y: float) -> Pt:
    r: float = (x ** 2 + y ** 2) ** 0.5
    theta_radians: float = atan2(y, x)

    return r, theta_radians


def polar_to_cartesian(r: float, theta_radians: float) -> Pt:
    x: float = r * cos(theta_radians)
    y: float = r * sin(theta_radians)

    return x, y


def get_intersection(line1: Chord, line2: Chord) -> Pt:
    (x1_1, y1_1), (x1_2, y1_2) = line1
    (x2_1, y2_1), (x2_2, y2_2) = line2

    dx1, dy1 = x1_1 - x1_2, y1_1 - y1_2
    dx2, dy2 = x2_1 - x2_2, y2_1 - y2_2

    denom = dx1 * dy2 - dy1 * dx2
    if denom == 0:
        raise Exception(f"Lines: {line1=}, {line2=} are parallel or coincident (the same line).")

    det1: float = x1_1 * y1_2 - y1_1 * x1_2
    det2: float = x2_1 * y2_2 - y2_1 * x2_2

    x: float = (det1 * dx2 - det2 * dx1) / denom
    y: float = (det1 * dy2 - det2 * dy1) / denom

    return x, y


@dataclass
class Vec2:
    x: float
    y: float

    @classmethod
    def from_polar(cls, r: float, theta_radians: float) -> Vec2:
        x, y = polar_to_cartesian(r=r, theta_radians=theta_radians)
        return cls(x=x, y=y)

    def to_polar(self):
        return cartesian_to_polar(x=self.x, y=self.y)

    def get_perpendicular_tangent_line(self) -> Pt:
        m: float = float('inf') if self.y == 0 else -self.x / self.y

        # y = mx + b (y_intercept)
        y_intercept: float = self.y - m * self.x

        return m, y_intercept


def get_distance(pt1: Pt, pt2: Pt):
    x1, y1 = pt1
    x2, y2 = pt2
    return sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )


Chord: TypeAlias = tuple[Pt, Pt]


class Circle:
    def __init__(self, radius: float = 1.0):
        self.radius: float = radius

    def get_random_circumference_pt(self) -> Pt:
        return polar_to_cartesian(
            r=self.radius,
            theta_radians=get_random_theta_radians()
        )

    def get_random_pt_via_random_polar_coords(self) -> Vec2:
        return Vec2.from_polar(
            r=uniform(0, self.radius),
            theta_radians=get_random_theta_radians()
        )

    def get_random_pt_via_guess_n_check(self) -> Vec2:
        """
            Gets random (x,y) pt in a circle of radius: <radius>

            Essentially guess and check in the square surrounding the circle until it's in the circle.
        """
        get_random_pt = lambda: (
            uniform(-self.radius, self.radius),
            uniform(-self.radius, self.radius)
        )

        x, y = get_random_pt()
        while x ** 2 + y ** 2 >= 1:
            x, y = get_random_pt()

        return Vec2(x=x, y=y)

    def get_random_pt_via_chord_intersection(self) -> Pt:
        thetas: list[float] = sorted(get_random_theta_radians() for _ in range(4))
        pt1_1, pt2_1, pt1_2, pt2_2 = (
            polar_to_cartesian(
                r=self.radius,
                theta_radians=t
            ) for t in thetas
        )
        return get_intersection(
            line1=(pt1_1, pt1_2),
            line2=(pt2_1, pt2_2)
        )

    def get_line_intersections(self, slope: float, y_intercept: float) -> Chord:
        # Vertical line
        if slope == float('inf'):
            return (0.0, -self.radius), (0.0, self.radius)

        """
             Find intersections between line and circle

             1.  y = mx + b
             2.  x^2 + y^2 = r^2

             x^2 + (mx + b)^2 = r^2
             x^2 + m^2x^2 + 2mxb + b^2 - r^2 = 0
             =>

             (1+m^2)x^2  + (2m(y_intercept))x + (y_intercept^2-r^2) = 0
             (a_part)x^2 + (b_part)         x + (c_part)            = 0

             Solve with quadratic formula
        """
        a_part: float = (1 + slope ** 2)
        b_bart: float = 2 * slope * y_intercept
        c_part: float = y_intercept ** 2 - self.radius ** 2

        x1, x2 = get_quadratic_roots(a=a_part, b=b_bart, c=c_part)
        y1: float = slope * x1 + y_intercept
        y2: float = slope * x2 + y_intercept

        return (x1, y1), (x2, y2)

    def get_random_chord_from_circumference(self) -> Chord:
        return self.get_random_circumference_pt(), self.get_random_circumference_pt()

    def get_random_chord_from_guess_n_check_pt(self) -> Chord:
        """
        Gets a chord by
            1. Getting line from the origin to a random pt in the circle
            2. Get chord made by the line perpendicular to that line and on the random pt
        """
        pt: Vec2 = self.get_random_pt_via_guess_n_check()

        m, y_intercept = pt.get_perpendicular_tangent_line()
        return self.get_line_intersections(
            slope=m,
            y_intercept=y_intercept
        )

    def get_random_chord_from_random_polar_coords(self) -> Chord:
        pt: Vec2 = self.get_random_pt_via_random_polar_coords()

        m, y_intercept = pt.get_perpendicular_tangent_line()
        return self.get_line_intersections(
            slope=m,
            y_intercept=y_intercept
        )

    def get_eq_triangle_side_length(self) -> float:
        return sqrt(3) * self.radius

    def get_eq_triangle_vertices(self) -> tuple[Pt, Pt, Pt]:
        half_radius: float = self.radius / 2
        root3_over_2: float = sqrt(3) / 2

        top_vertex = (0, self.radius)
        bottom_left_vertex = (-root3_over_2, -half_radius)
        bottom_right_vertex = (root3_over_2, -half_radius)
        return top_vertex, bottom_left_vertex, bottom_right_vertex, top_vertex

    def plot_ax(
            self,
            ax,
            chord_fn,
            title: str,
            n: int = 10_000,
    ):
        # Calculate
        pts = [chord_fn() for _ in range(n)]
        triangle_side_len = self.get_eq_triangle_side_length()

        # Setup plot
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontsize=32)
        ax.set_ylim(-self.radius, 1.6 * self.radius)
        ax.set_xlim(-self.radius, self.radius)

        # Plot circle
        circle = plt.Circle((0, 0),
                            self.radius,
                            edgecolor='blue',
                            facecolor='none',
                            linewidth=2)
        ax.add_patch(circle)

        # Plot chords
        for p1, p2 in pts:
            x1, y1 = p1
            x2, y2 = p2

            # color = "red"
            # alpha = 0.2
            # is_greater_than: bool = get_distance(p1, p2) > self.get_eq_triangle_side_length()
            # if is_greater_than:
            #     color = "green"
            #     alpha = 0.5

            ax.plot(
                [x1, x2], [y1, y2],
                color="purple",
                alpha=0.5,
                linestyle='-',
                linewidth=0.1
            )

        # Add legend with % chance of edge length greater than triangle edge length
        lengths = [get_distance(p1, p2) for p1, p2 in pts]

        n_lengths = len(lengths)

        n_gt = len([l for l in lengths if l > triangle_side_len])
        n_lt = n_lengths - n_gt
        percent_gt, percent_lt = n_gt / n, n_lt / n

        tab = [
            [f"Length({triangle_side_len:,.2f})", "%"],
            [f">", f"{percent_gt:,.2%}"],
            [f"<=", f"{percent_lt:,.2%}"],
        ]
        label = tabulate.tabulate(
            tab,
            headers='firstrow',
            colalign=['left', 'right'],
            tablefmt="plain",
            intfmt=',',
            floatfmt=',.2f'
        )
        ax.text(
            0.5,
            0.88,
            label,
            fontsize=28,
            color="white",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="purple",
                linewidth=2,
                alpha=0.8,
            ),
            transform=ax.transAxes,
        )

        # Plot triangle
        triangle = self.get_eq_triangle_vertices()
        x_t, y_t = zip(*triangle)
        ax.plot(
            x_t, y_t,
            color='blue',
            linewidth=2
        )


def main():
    ax_width = 12
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(ax_width * 3, ax_width)
    )
    padding = 0.05
    margin = 0.01
    fig.subplots_adjust(
        left=padding,
        right=1.0 - padding,
        bottom=padding,
        top=1 - padding,
        wspace=margin,
        hspace=margin
    )
    circle = Circle(radius=RADIUS)

    circle.plot_ax(ax=axes[0],
                   chord_fn=circle.get_random_chord_from_guess_n_check_pt,
                   title=f"Random Chord Midpoint")

    circle.plot_ax(ax=axes[1],
                   chord_fn=circle.get_random_chord_from_random_polar_coords,
                   title=f"Random Radial Pt")

    circle.plot_ax(ax=axes[2],
                   chord_fn=circle.get_random_chord_from_circumference,
                   title=f"Random Chord Intersection")

    plt.show()


if __name__ == "__main__":
    main()
