"""
Fitting a normal distribution by MLE parameter estimation to symbols

Mean

The estimator of the mean parameter of a normal distribution in terms
of symbol distributions is given by the weighted sum of expected values of each symbol distribution:

mu_S = sum { n_k/N * m_k } where m_k is the expected value of distribution F_k,
and n_k is the number of points in the symbol

----------------------------------------------------------------

Variance

The estimator of the variance parameter of a normal distribution
in terms of symbol distributions is given by the following weighted sum:

s^2_s = sum { n_k/N * ((1 - 1/N) s^2_k + (m_k - m)^2) }
where n_k is the number of points in the symbol, m_k and s^2_k are respectively
the mean and variance of F_k, and m is the symbolic mean (see above)

NOTE: this estimator is a little bit biased, just like the usual MLE of variance
for a normal distribution.
"""
from enum import Enum, unique
from typing import Iterable

from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def uniform_symbols_mean_mle(symbols: Iterable[UniformSymbol]) -> float:
    N = sum(s.n for s in symbols)

    # mean of uniform(a, b) is (a + b)/2
    return sum(n / N * (a + b) / 2 for (a, b, n) in symbols)


def uniform_symbols_variance_mle(symbols: Iterable[UniformSymbol]) -> float:
    N = sum(s.n for s in symbols)
    m = uniform_symbols_mean_mle(symbols)

    # variance of uniform(a, b) is (b - a)^2 / 12
    return sum(n / N * ((1 - 1 / N) * (b - a) ** 2 / 12 + ((a + b) / 2 - m) ** 2) for (a, b, n) in symbols)


def normal_symbols_mean_mle(symbols: Iterable[NormalSymbol]) -> float:
    """
    
    µ̂ = sum { n_k / N * m_k },
    where m_k is class mean and n_k is class size, N is sum of class sizes

    :param symbols: Symbolic data
    :return: Estimator of mean parameter of univariate normal distribution model
    """
    N = sum(s.n for s in symbols)

    return sum(n / N * mu for (mu, _, n) in symbols)


@unique
class VarianceBiasType(Enum):
    M1_BIASED_SUMMARY = "method1_biased_summary"
    M1_UNBIASED_SUMMARY = "method1_unbiased_summary"
    M3_BIASED_ESTIMATOR = "method3_biased_estimator"
    M3_UNBIASED_ESTIMATOR = "method3_unbiased_estimator"


def normal_symbols_variance_mle(symbols: Iterable[NormalSymbol], bias_type: VarianceBiasType) -> float:
    """
    σ̂^2 = sum { s_k * b1 + (m_k - µ̂)^2 * b2 },
    where m_k and s_k are class mean and variance respectively, N is sum of class sizes,
    µ̂ is defined in normal_symbols_mean_mle, and b1 and b2 are bias factors.

    M1_BIASED_SUMMARY: b1 = n_k/N, b2 = n_k/N
    Equivalent to σ̂^2 = sum { n_k / N * (s_k + (m_k - µ̂)^2) },
    Derived from the original formulation of Method 1 which used the biased sample variance
    as the summary function: s^2 = 1/n * sum { (x_i - x̄)^2 }
    (Note: this was also the result obtained from Method 2, but this fact isn't enough to prove
    that variant overall has zero/least bias)

    M1_UNBIASED_SUMMARY: b1 = (n_k - 1)/N, b2 = n_k/N
    Equivalent to σ̂^2 = sum { n_k / N * (s_k*(1 - 1/n_k) + (m_k - µ̂)^2) },
    Derived by applying Method 1 but replacing the biased sample variance summary function
    with unbiased: s^2 = 1/(n-1) * sum { (x_i - x̄)^2 }

    M3_BIASED_ESTIMATOR: b1 = n_k/N * (1 - 1/N), b2 = n_k/N
    Equivalent to σ̂^2 = sum { n_k / N * (s_k * (1 - 1/N) + (m_k - µ̂)^2) },
    Derived by applying Method 3 with the vanilla (biased) ML estimator for variance.

    M3_UNBIASED_ESTIMATOR: b1 = n_k/N, b2 = n_k/(N-1)
    Equivalent to σ̂^2 = sum { n_k / N * (s_k * (1 - 1/N) + (m_k - µ̂)^2) },
    Derived by applying Method 3 with the bias-corrected ML estimator for variance.

    :param symbols: Symbolic data
    :param bias_type: See above
    :return: Estimator of variance parameter of normal distribution model
    """
    N = sum(s.n for s in symbols)
    m = normal_symbols_mean_mle(symbols)

    # define bias factors
    def b1(n_k: float) -> float:
        match bias_type:
            case VarianceBiasType.M1_BIASED_SUMMARY:
                return n_k / N
            case VarianceBiasType.M1_UNBIASED_SUMMARY:
                return (n_k - 1) / N
            case VarianceBiasType.M3_BIASED_ESTIMATOR:
                return (n_k / N) * (1 - 1/N)
            case VarianceBiasType.M3_UNBIASED_ESTIMATOR:
                return n_k / N
            case _:
                raise ValueError(f"Unrecognised bias type: {bias_type}")

    def b2(n_k: float) -> float:
        match bias_type:
            case VarianceBiasType.M1_BIASED_SUMMARY:
                return n_k / N
            case VarianceBiasType.M1_UNBIASED_SUMMARY:
                return n_k / N
            case VarianceBiasType.M3_BIASED_ESTIMATOR:
                return n_k / N
            case VarianceBiasType.M3_UNBIASED_ESTIMATOR:
                return n_k / (N - 1)
            case _:
                raise ValueError(f"Unrecognised bias type: {bias_type}")

    return sum(b1(n)*sigma**2 + b2(n)*(mu - m)**2 for (mu, sigma, n) in symbols)
