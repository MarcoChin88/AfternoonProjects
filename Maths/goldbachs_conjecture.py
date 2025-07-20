import math
import time
from functools import wraps

from matplotlib import pyplot as plt

from utils.common import get_func_call_str


def log_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        runtime = time.time() - start_time

        func_call_str: str = get_func_call_str(func, *args, **kwargs)
        msg: str = f"{func_call_str}: {runtime: 0.2f}s"
        print(msg)

        return result

    return wrapper


@log_runtime
def sieve_of_eratosthenes(n: int = 30):
    mask = [True] * (n + 1)
    mask[0: 2] = [False, False]

    for i in range(2, n + 1):
        for m in range(2, n + 1):
            if i * m > n:
                break
            mask[i * m] = False

    return [i for i, is_prime in enumerate(mask) if is_prime]


@log_runtime
def sieve_of_euler(n: int = 1000) -> list[int]:
    mask = [True] * (n + 1)
    mask[0: 2] = [False, False]
    primes = []

    for i in range(2, n + 1):
        if mask[i]:
            primes.append(i)

        for p in primes:
            if i * p > n:
                break
            mask[i * p] = False
            if i % p == 0:
                break

    return primes


def get_prime_sums(even: int, primes: list[int]):
    primes_set = set(primes)
    sums = set()
    for p in primes:
        if p > even:
            break
        r = even - p
        if r in primes_set and {p, r} not in sums:
            sums.add(tuple(sorted([p, r])))
    return sums


def get_even_number_prime_sums(n: int = 1000):
    primes = sieve_of_eratosthenes(n=n)

    evens = list(range(4, n + 1, 2))
    d = {}
    for even in evens:
        sums = get_prime_sums(even, primes)
        d[even] = len(sums)

    return d


def main():
    data = get_even_number_prime_sums(n=1000)
    X, Y = zip(*data.items())

    fn = lambda n: n / (math.log(n) ** 2)
    Y_exp = [fn(_2n / 2) for _2n in X]

    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, Y_exp, color='red')
    plt.show()


if __name__ == "__main__":
    main()
