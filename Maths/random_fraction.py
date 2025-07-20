import random

import math

EXPECTED_ROUNDED = (5 - math.pi) / 4


def main():
    n = 0
    num_rounded_evens = 0
    num_floored_evens = 0
    while True:
        n += 1
        q = random.random() / random.random()

        if round(q) % 2 == 0:
            num_rounded_evens += 1
        if int(q) % 2 == 0:
            num_floored_evens += 1

        if n % 1_000_000 == 0:
            rounded_ratio = num_rounded_evens / n
            rounded_err = (EXPECTED_ROUNDED - rounded_ratio) / EXPECTED_ROUNDED
            print(f"{rounded_ratio=:.3f}, {EXPECTED_ROUNDED=:.3f}, {rounded_err:.3f}")

            floored_ratio = num_floored_evens / n


if __name__ == "__main__":
    main()
