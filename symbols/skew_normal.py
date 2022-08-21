import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import skewnorm

from symbols.distribution import Distribution
from symbols.symboldata import SymbolData


class SkewNormalParameters(SymbolData, Distribution):
    """
    Parameters:
    xi - location
    omega - scale
    alpha - shape
    """

    def __init__(self, xi: float, omega: float, alpha: float):
        super().__init__([xi, omega, alpha])
        if omega == 0:
            raise ValueError("omega cannot be zero")
        elif omega < 0:
            omega = -omega

        self.xi = xi
        self.omega = omega
        self.alpha = alpha

    def heuristic_min_max(self):
        # TODO
        return self.xi - 3*self.omega, self.xi + 3*self.omega

    def pdf(self, x: np.ndarray):
        return skewnorm.pdf(x, self.alpha, loc=self.xi, scale=self.omega)

    def cdf(self, x: np.ndarray):
        return skewnorm.cdf(x, self.alpha, loc=self.xi, scale=self.omega)

    def rvs(self, size: ArrayLike):
        return skewnorm.rvs(self.alpha, loc=self.xi, scale=self.omega, size=size)
