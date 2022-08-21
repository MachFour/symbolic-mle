import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from symbols.distribution import Distribution
from symbols.symbol import DistributionSymbol


class NormalParameters(Distribution):
    """
    Parameters:
    mu - mean
    sigma - standard deviation (nonzero; negative values will be rectified)
    n - number of points summarised by symbol
    """

    def __init__(self, mu: float, sigma: float):
        super().__init__([mu, sigma])
        if sigma == 0:
            raise ValueError("sigma cannot be zero")
        elif sigma < 0:
            sigma = -sigma

        self.mu = mu
        self.sigma = sigma

    def heuristic_min_max(self):
        return self.mu - 3*self.sigma, self.mu + 3*self.sigma

    def pdf(self, x: np.ndarray):
        return norm.pdf(x, loc=self.mu, scale=self.sigma)

    def cdf(self, x: np.ndarray):
        return norm.cdf(x, loc=self.mu, scale=self.sigma)

    def rvs(self, size: ArrayLike):
        return norm.rvs(loc=self.mu, scale=self.sigma, size=size)


class NormalSymbol(DistributionSymbol[NormalParameters]):
    def __init__(self, mu: float, sigma: float, n: int):
        super().__init__(NormalParameters(mu, sigma), n)


    @property
    def mu(self):
        return self.data.mu

    @property
    def sigma(self):
        return self.data.sigma