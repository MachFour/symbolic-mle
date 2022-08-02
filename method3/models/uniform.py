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
from typing import Sequence

import numpy as np
from scipy.stats import norm, uniform

from helper.utils import expectation_from_cdf
from symbols.common import symbols_heuristic_min_max
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def uniform_symbol_min_cdf(x: np.ndarray, symbols: Sequence[UniformSymbol]) -> np.ndarray:
    # sf is survival function = 1 - cdf
    product_terms = tuple((uniform.sf(x, loc=s.a, scale=s.b - s.a)) ** s.n for s in symbols)
    return 1 - np.prod(np.asarray(product_terms), axis=0)


def uniform_symbol_max_cdf(x: np.ndarray, symbols: Sequence[UniformSymbol]) -> np.ndarray:
    product_terms = tuple((uniform.cdf(x, loc=s.a, scale=s.b - s.a)) ** s.n for s in symbols)
    return np.prod(np.asarray(product_terms), axis=0)


def normal_symbol_min_cdf(x: np.ndarray, symbols: Sequence[NormalSymbol]) -> np.ndarray:
    # sf is survival function = 1 - cdf
    product_terms = tuple((norm.sf(x, loc=s.mu, scale=s.sigma)**s.n for s in symbols))
    return 1 - np.prod(np.asarray(product_terms), axis=0)


def normal_symbol_max_cdf(x: np.ndarray, symbols: Sequence[NormalSymbol]) -> np.ndarray:
    product_terms = tuple((norm.cdf(x, loc=s.mu, scale=s.sigma)**s.n for s in symbols))
    return np.prod(np.asarray(product_terms), axis=0)


def uniform_symbol_min_mle(symbols: Sequence[UniformSymbol]) -> float:
    x_min, x_max = symbols_heuristic_min_max(symbols, 1.3)
    return expectation_from_cdf(lambda x: uniform_symbol_min_cdf(x, symbols), x_min, x_max)


def uniform_symbol_max_mle(symbols: Sequence[UniformSymbol]) -> float:
    x_min, x_max = symbols_heuristic_min_max(symbols, 1.3)
    return expectation_from_cdf(lambda x: uniform_symbol_max_cdf(x, symbols), x_min, x_max)


def normal_symbol_min_mle(symbols: Sequence[NormalSymbol]) -> float:
    x_min, x_max = symbols_heuristic_min_max(symbols, 1.3)
    return expectation_from_cdf(lambda x: normal_symbol_min_cdf(x, symbols), x_min, x_max)


def normal_symbol_max_mle(symbols: Sequence[NormalSymbol]) -> float:
    x_min, x_max = symbols_heuristic_min_max(symbols, 1.3)
    return expectation_from_cdf(lambda x: normal_symbol_max_cdf(x, symbols), x_min, x_max)


