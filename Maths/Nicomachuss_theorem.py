"""
Tests this theorem
https://en.wikipedia.org/wiki/Squared_triangular_number
that states
1^3 + 2^3 + 3^3 + ... + n^3 = (1 + 2 + 3 + ... + n) ^ 2
"""

from functools import cache

import numpy as np


@cache
def get_summed_cubes_rec(n: int) -> int:
    if n <= 1:
        return 1

    return n ** 3 + get_summed_cubes_rec(n - 1)


def get_summed_cubes(n: int) -> int:
    return sum(np.arange(1, n + 1) ** 3)


def get_squared_sum(n: int) -> int:
    return sum(range(1, n + 1)) ** 2


def test_theorum(n: int):
    for i in range(1, n + 1):
        summed_cubes = get_summed_cubes(i)
        squared_sum = get_squared_sum(i)

        if summed_cubes == squared_sum:
            print(f"{i=}: {summed_cubes} == {squared_sum}")
        else:
            raise Exception(f"Oopsy daisies")


def main():
    test_theorum(1000)


if __name__ == "__main__":
    main()
