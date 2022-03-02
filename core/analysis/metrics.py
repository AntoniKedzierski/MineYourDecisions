import numpy as np


def my_score(a, b, split, p, q):
    return sum(np.power(np.abs(a[0:split] - b[0:split]), p)) ** (1 / p) + sum(np.power(np.abs(a[split:] - b[split:]), q)) ** (1 / q)
