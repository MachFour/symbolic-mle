import numpy as np
from scipy.stats import norm


class NormalSymbol:
    """
    Parameters:
    mu - mean
    sigma - standard deviation (nonzero; negative values will be rectified)
    n - number of points summarised by symbol
    """

    def __init__(self, mu: float, sigma: float, n: int):
        if sigma == 0:
            raise ValueError("sigma cannot be zero")
        elif sigma < 0:
            sigma = -sigma

        if not isinstance(n, int):
            raise TypeError(f"n must be integer but was {type(n)}")
        self.mu = mu
        self.sigma = sigma
        self.n = n

    # support tuple unpacking
    def __len__(self):
        return 3

    # support tuple unpacking
    def __getitem__(self, key):
        if type(key) == int:
            if key == 0:
                return self.mu
            elif key == 1:
                return self.sigma
            elif key == 2:
                return self.n
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError(f"indices must be integers, not {type(key)}")

    def heuristic_min_max(self):
        return self.mu - 3*self.sigma, self.mu + 3*self.sigma

    def pdf(self, x: np.ndarray):
        return norm.pdf(x, loc=self.mu, scale=self.sigma)

    def cdf(self, x: np.ndarray):
        return norm.cdf(x, loc=self.mu, scale=self.sigma)

    def rvs(self):
        return norm.rvs(loc=self.mu, scale=self.sigma, size=self.n)


