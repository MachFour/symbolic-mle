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

from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def uniform_symbols_mean_mle(symbols: tuple[UniformSymbol]) -> float:
    N = sum(s.n for s in symbols)

    # mean of uniform(a, b) is (a + b)/2
    return sum(n / N * (a + b) / 2 for (a, b, n) in symbols)


def uniform_symbols_variance_mle(symbols: tuple[UniformSymbol]) -> float:
    N = sum(s.n for s in symbols)
    m = uniform_symbols_mean_mle(symbols)

    # variance of uniform(a, b) is (b - a)^2 / 12
    return sum(n / N * ((1 - 1 / N) * (b - a) ** 2 / 12 + ((a + b) / 2 - m) ** 2) for (a, b, n) in symbols)


def normal_symbols_mean_mle(symbols: tuple[NormalSymbol]) -> float:
    N = sum(s.n for s in symbols)

    return sum(n / N * mu for (mu, _, n) in symbols)


def normal_symbols_variance_mle(symbols: tuple[NormalSymbol]) -> float:
    N = sum(s.n for s in symbols)
    m = normal_symbols_mean_mle(symbols)

    return sum(n / N * ((1 - 1 / N) * sigma ** 2 + (mu - m) ** 2) for (mu, sigma, n) in symbols)
