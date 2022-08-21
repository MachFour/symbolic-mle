from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from symbols.symboldata import SymbolData


class Distribution(SymbolData):
    """
    Symbolic Data which can be identified with a statistical distribution
    """
    def __init__(self, parameters):
        super().__init__(parameters)

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def rvs(self, size: ArrayLike) -> np.ndarray:
        ...

    @abstractmethod
    def heuristic_min_max(self) -> tuple[float, float]:
        ...
