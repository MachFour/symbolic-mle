from typing import Sequence

import numpy as np
import scipy.optimize
from scipy.integrate import quad

from symbols.normal import NormalSymbol
from symbols.skew_normal import SkewNormalParameters


def skew_normal_class_entropy(parameters: SkewNormalParameters, symbol: NormalSymbol) -> float:
    result = quad(lambda x: -np.log(parameters.pdf(x)) * symbol.pdf(x), np.Infinity, np.Infinity)
    return result[0]


def skew_normal_symbols_entropy(parameters: SkewNormalParameters, symbols: Sequence[NormalSymbol]) -> float:
    N = sum(s.n for s in symbols)
    return sum(s.n / N * skew_normal_class_entropy(parameters, s) for s in symbols)


def method2_skewnormal_fit(symbols: Sequence[NormalSymbol]) -> tuple[float, float, float]:
    # find parameters that minimise skew_normal_symbols_entropy
    # TODO how to choose initial parameters?

    def minimise_fn(x: np.ndarray) -> float:
        xi, omega, alpha = x[0], x[1], x[2]
        parameters = SkewNormalParameters(xi, omega, alpha, 1)
        return skew_normal_class_entropy(parameters, symbols)
    scipy.optimize.minimize(skew_normal_symbols_entropy)



