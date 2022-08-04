from typing import Sequence

import numpy as np

from symbols.uniform import UniformSymbol


def fit_to_uniform_symbols(symbols: Sequence[UniformSymbol]) -> tuple[float, float]:
    """
    Implements minimisation of 5.3.9 from theis by direct minimisation.

    By calculations, we have

    μ̂ = sum { n_k / N * c_k }
    σ̂^2 = sum { n_k / N * (c_k - μ̂)^2 + (b_k - a_k)^2/12

    where c_k = (a_k + b_k) / 2

    :param symbols: Uniform symbols to minimise
    :return: Tuple of [μ̂, σ̂] parameters for fitted Normal distribution
    """

    N = sum(s.n for s in symbols)
    mu_hat = sum(s.n/N * (s.a + s.b)/2 for s in symbols)
    sigma_hat_sq = sum(s.n/N * ((mu_hat - (s.a + s.b)/2)**2 + (s.b - s.a)**2 / 12) for s in symbols)
    sigma_hat = np.sqrt(sigma_hat_sq)

    return mu_hat, sigma_hat

