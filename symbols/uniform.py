import numpy as np
from scipy.stats import uniform

from symbols.distribution import Distribution
from symbols.order_statistic import OrderStatisticSymbol
from symbols.symbol import DistributionSymbol
from symbols.symboldata import SymbolData


class UniformParameters(SymbolData, Distribution):
    """
    Parameters:
    a - minimum value
    b - maximum value
    """

    def __init__(self, a: float, b: float):
        if b < a:
            raise ValueError(f"Uniform symbol must have b < a but a={a} and b={b}")
        if not isinstance(n, int):
            raise TypeError(f"n must be integer but was {type(n)}")
        self.a = a
        self.b = b

    def heuristic_min_max(self):
        return self.a, self.b

    def pdf(self, x: np.ndarray):
        return uniform.pdf(x, loc=self.a, scale=self.b - self.a)

    def cdf(self, x: np.ndarray):
        return uniform.cdf(x, loc=self.a, scale=self.b - self.a)

    def rvs(self):
        return uniform.rvs(loc=self.a, scale=self.b - self.a, size=self.n)


class UniformSymbol(DistributionSymbol[UniformParameters]):
    def __init__(self, a: float, b: float, n: int):
        super().__init__(UniformParameters(a, b), n)

    @property
    def a(self):
        return self.data.a

    @property
    def b(self):
        return self.data.b

    def to_order_statistic_symbol(self) -> OrderStatisticSymbol:
        return OrderStatisticSymbol(self.a, self.b, 1, self.n, self.n)

