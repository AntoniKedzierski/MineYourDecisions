import numbers
import numpy as np

def triang_fuzzy(x, a, b, c):
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number) and isinstance(c, numbers.Number):
        return np.max(np.c_[
            np.zeros(x.shape),
            np.min(np.c_[
                (x - a * np.ones(x.shape)) / (b * np.ones(x.shape) - a * np.ones(x.shape)),
                (c * np.ones(x.shape) - x) / (c * np.ones(x.shape) - b * np.ones(x.shape))
            ], axis=1)
        ], axis=1)

if __name__ == '__main__':
    print(triang_fuzzy(np.asarray([1, 2, 3, 4]), 1.4, 2.6, 5))



