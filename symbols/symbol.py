from typing import TypeVar, Generic

import numpy as np
from numpy.typing import ArrayLike

from symbols.distribution import Distribution
from symbols.symboldata import SymbolData

D = TypeVar('D', bound=SymbolData)
P = TypeVar('P', bound=Distribution)


class Symbol(Generic[D]):
    """
    Instance variables:
    data: Object containing summary information; instance of SymbolData class
    n - number of points summarised by symbol
    """

    def __init__(self, data: D, n: int):
        self.data = data
        self.n = n


class DistributionSymbol(Symbol[P], Distribution):
    """
    Instance variables:
    data: Object containing distribution information; instance of Distribution class
    n - number of points summarised by symbol
    """

    def __init__(self, data: P, n: int):
        super().__init__(data, n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.data.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.data.cdf(x)

    def rvs(self, size: ArrayLike) -> np.ndarray:
        return self.data.rvs(size=self.n)

    def heuristic_min_max(self) -> tuple[float, float]:
        return self.data.heuristic_min_max()
