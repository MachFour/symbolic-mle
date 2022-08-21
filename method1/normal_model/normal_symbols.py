from typing import Sequence

from method3.normal_model.normal_symbols import normal_symbols_mean_mle, normal_symbols_variance_mle, VarianceBiasType
from symbols.normal import NormalSymbol


def method1_normal_fit(symbols: Sequence[NormalSymbol]) -> tuple[float, float]:
    mu = normal_symbols_mean_mle(symbols)
    sigma = normal_symbols_variance_mle(symbols, VarianceBiasType.M1_BIASED_SUMMARY)**0.5
    return mu, sigma
