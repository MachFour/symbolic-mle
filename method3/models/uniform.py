"""
Fitting a uniform distribution by MLE parameter estimation to symbols

Minimum

The estimator of the minimum parameter of a uniform distribution
in terms of a symbol distributions is given by the expected value
of random variable A with CDF:

F_A(a) = 1 - prod { (1 - F_k(a))^n_k },
where F_k is the CDF of the k'th symbol and and n_k is the
number of points summarised by the symbol

----------------------------------------------------------------

Maximum

The estimator of the maximum parameter of a uniform distribution in terms of symbol
distributions is given by the expected value of random variable B with CDF:

F_B(b) = prod { (F_k(b))^n_k }, where F_k is the CDF of the kth symbol
"""
import numpy as np
from scipy.stats import norm, uniform

from helper.utils import expectation_from_cdf
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def uniform_symbol_min_cdf(x: np.ndarray, symbols: tuple[UniformSymbol]) -> np.ndarray:
    # sf is survival function = 1 - cdf
    product_terms = tuple((uniform.sf(x, loc=s.a, scale=s.b - s.a)) ** s.n for s in symbols)
    return 1 - np.prod(np.asarray(product_terms), axis=0)


def uniform_symbol_max_cdf(x: np.ndarray, symbols: tuple[UniformSymbol]) -> np.ndarray:
    product_terms = tuple((uniform.cdf(x, loc=s.a, scale=s.b - s.a)) ** s.n for s in symbols)
    return np.prod(np.asarray(product_terms), axis=0)


def normal_symbol_min_cdf(x: np.ndarray, symbols: tuple[NormalSymbol]) -> np.ndarray:
    # sf is survival function = 1 - cdf
    product_terms = tuple((norm.sf(x, loc=s.mu, scale=s.sigma)**s.n for s in symbols))
    return 1 - np.prod(np.asarray(product_terms), axis=0)


def normal_symbol_max_cdf(x: np.ndarray, symbols: tuple[NormalSymbol]) -> np.ndarray:
    product_terms = tuple((norm.cdf(x, loc=s.mu, scale=s.sigma)**s.n for s in symbols))
    return np.prod(np.asarray(product_terms), axis=0)


def uniform_symbol_min_mle(symbols: tuple[UniformSymbol]) -> float:
    x_min, x_max = uniform_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    return expectation_from_cdf(lambda x: uniform_symbol_min_cdf(x, symbols), x_min, x_max)


def uniform_symbol_max_mle(symbols: tuple[UniformSymbol]) -> float:
    x_min, x_max = uniform_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    return expectation_from_cdf(lambda x: uniform_symbol_max_cdf(x, symbols), x_min, x_max)


def normal_symbol_min_mle(symbols: tuple[NormalSymbol]) -> float:
    x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    return expectation_from_cdf(lambda x: normal_symbol_min_cdf(x, symbols), x_min, x_max)


def normal_symbol_max_mle(symbols: tuple[NormalSymbol]) -> float:
    x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    return expectation_from_cdf(lambda x: normal_symbol_max_cdf(x, symbols), x_min, x_max)


def normal_symbols_heuristic_min_max(
        symbols: tuple[NormalSymbol],
        expand_factor: float | None = None
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20
    for (mu, ss, _) in symbols:
        x_min = min(x_min, mu - 10*np.sqrt(ss))
        x_max = max(x_max, mu + 10*np.sqrt(ss))

    if expand_factor:
        midpoint = (x_min + x_max) / 2
        new_range = (x_max - x_min) * expand_factor
        x_min = midpoint - new_range / 2
        x_max = midpoint + new_range / 2

    return x_min, x_max


def uniform_symbols_heuristic_min_max(
    symbols: tuple[UniformSymbol],
    expand_factor: float | None = None
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20
    for (a, b, _) in symbols:
        x_min = min(x_min, a)
        x_max = max(x_max, b)

    if expand_factor:
        midpoint = (x_min + x_max) / 2
        new_range = (x_max - x_min) * expand_factor
        x_min = midpoint - new_range / 2
        x_max = midpoint + new_range / 2

    return x_min, x_max
