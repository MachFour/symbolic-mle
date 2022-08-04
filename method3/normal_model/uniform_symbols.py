"""
Fitting a normal distribution by MLE parameter estimation to uniform symbols
"""
from typing import Sequence

from symbols.uniform import UniformSymbol


def uniform_symbols_mean_mle(symbols: Sequence[UniformSymbol]) -> float:
    N = sum(s.n for s in symbols)

    # mean of uniform(a, b) is (a + b)/2
    return sum(n / N * (a + b) / 2 for (a, b, n) in symbols)


def uniform_symbols_variance_mle(symbols: Sequence[UniformSymbol]) -> float:
    N = sum(s.n for s in symbols)
    m = uniform_symbols_mean_mle(symbols)

    # variance of uniform(a, b) is (b - a)^2 / 12
    return sum(n / N * ((1 - 1 / N) * (b - a) ** 2 / 12 + ((a + b) / 2 - m) ** 2) for (a, b, n) in symbols)


