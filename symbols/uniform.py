import numpy as np
from scipy.stats import uniform


class UniformSymbol:
    """
    Parameters:
    a - minimum value
    b - maximum value
    n - number of points summarised by symbol
    """

    def __init__(self, a: float, b: float, n: int):
        if b < a:
            raise ValueError(f"Uniform symbol must have b < a but a={a} and b={b}")
        if not isinstance(n, int):
            raise TypeError(f"n must be integer but was {type(n)}")
        self.a = a
        self.b = b
        self.n = n

    # support tuple unpacking
    def __len__(self):
        return 3

    # support tuple unpacking
    def __getitem__(self, key):
        if type(key) == int:
            if key == 0:
                return self.a
            elif key == 1:
                return self.b
            elif key == 2:
                return self.n
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError(f"indices must be integers, not {type(key)}")

    def heuristic_min_max(self):
        return self.a, self.b

    def pdf(self, x: np.ndarray):
        return uniform.pdf(x, loc=self.a, scale=self.b - self.a)

    def cdf(self, x: np.ndarray):
        return uniform.cdf(x, loc=self.a, scale=self.b - self.a)

    def rvs(self):
        return uniform.rvs(loc=self.a, scale=self.b - self.a, size=self.n)



