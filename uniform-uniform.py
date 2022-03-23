import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from scipy.integrate import quad
from numbers import Number
from typing import Callable

# symbol parameters grouped together
a = [-15, -6, 10, 30]
b = [40, 50,  60, 45]
n = [4, 43, 2, 12]

uniform_symbols = tuple(zip(a, b, n))

UniformSymbol = tuple[float, float, int]


# plot density and CDF of uniform symbols
# plot density of minimum
# plot density of maximum

def check_uniform_distribution_parameters(a, b):
    if b < a:
        msg = f"Invalid uniform distribution: b = {b} < a = {a}"
        raise ValueError(msg)


def symbolic_min_cdf(x: np.ndarray, symbols: tuple[UniformSymbol]) -> np.ndarray:
    """
    The estimator of the minimum parameter of a uniform distribution fitted to
    uniform symbols is given by the expected value of random variable A with CDF:

    F_A(a) = 1 - prod { (1 - F_k(a))^n_k }, where F_k is the CDF of the k'th symbol
    """
    # sf is survival function = 1 - cdf
    product_terms = list((scipy.stats.uniform.sf(x, loc=a, scale=b-a))**n for (a, b, n) in symbols)
    sf = np.prod(np.asarray(product_terms), axis=0)
    return 1 - sf


def symbolic_max_cdf(x: np.ndarray, symbols: tuple[UniformSymbol]) -> np.ndarray:
    """
    The estimator of the maximum parameter of a uniform distribution fitted to
    uniform symbols is given by the expected value of random variable B with CDF:

    F_B(b) = prod { (F_k(b))^n_k }, where F_k is the CDF of the kth symbol
    """
    product_terms = list((scipy.stats.uniform.cdf(x, loc=a, scale=b-a))**n for (a, b, n) in symbols)
    return np.prod(np.asarray(product_terms), axis=0)

def calculate_min_max(
    symbols: tuple[UniformSymbol],
    expand_factor: float|None = None
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20
    for (a, b, _) in symbols:
        check_uniform_distribution_parameters(a, b)
        x_min = min(x_min, a)
        x_max = max(x_max, b)

    if expand_factor:
        midpoint = (x_min + x_max) / 2
        new_range = (x_max - x_min) * expand_factor
        x_min = midpoint - new_range / 2
        x_max = midpoint + new_range / 2

    return (x_min, x_max)


def make_axis_values(plot_min: float, plot_max: float, density: float = 10) -> np.ndarray:
    return np.linspace(plot_min, plot_max, round(plot_max - plot_min) * 10)


def plot_uniform_unifom_fitting(symbols: tuple[UniformSymbol]):
    """
    Plot distributions of each uniform symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = calculate_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    cdf_A = symbolic_min_cdf(x, symbols)
    mean_A = expectation_from_cdf(lambda x: symbolic_min_cdf(x, symbols), x_min, x_max)

    cdf_B = symbolic_max_cdf(x, symbols)
    mean_B = expectation_from_cdf(lambda x: symbolic_max_cdf(x, symbols), x_min, x_max)

    fig: plt.Figure = plt.figure()
    fig.suptitle("Fitting a Uniform distribution to Uniform symbols")
    ax0: plt.Axes = fig.add_subplot(4, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")

    ax1: plt.Axes = fig.add_subplot(4, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F(x)")

    for (a, b, _) in symbols:
        ax0.plot(x, scipy.stats.uniform.pdf(x, loc=a, scale=b-a))
        ax1.plot(x, scipy.stats.uniform.cdf(x, loc=a, scale=b-a))
    ax0.stem(mean_A, ax0.get_ylim()[1], linefmt='C5')
    ax0.stem(mean_B, ax0.get_ylim()[1], linefmt='C5')
    ax1.stem(mean_A, ax1.get_ylim()[1], linefmt='C5')
    ax1.stem(mean_B, ax1.get_ylim()[1], linefmt='C5')


    ax2: plt.Axes = fig.add_subplot(4, 1, 3)
    ax2.set_title("CDF of A - fitted minimum is E[A]")
    ax2.set_xlabel("a")
    ax2.set_ylabel("F_A(a)")


    ax3: plt.Axes = fig.add_subplot(4, 1, 4)
    ax3.set_title("CDF of B - fitted maximum is E[B]")
    ax3.set_xlabel("b")
    ax3.set_ylabel("F_B(b)")

    ax2.plot(x, cdf_A)
    ax3.plot(x, cdf_B)
    ax2.stem(mean_A, 1, linefmt='C5')
    ax3.stem(mean_B, 1, linefmt='C5')

    #fig.tight_layout()

def expectation_from_cdf(
    cdf: Callable[np.ndarray, np.ndarray],
    lower_limit: float = -np.Infinity,
    upper_limit: float = np.Infinity
) -> float:
    """
    Let X be a continouous, real-valued random variable with CDF F(x).
    This function computes the expected value of X via numerical integration.
    There are at least two methods / equations for this.

    #1 from comment in https://stats.stackexchange.com/a/222497/262661
        E[X] = ∫_{[0, ∞)} (1−F(x)) dx − ∫_{(−∞, 0]} F(x) dx.
        whenever the expectation is finite. This apparently can be derived from
        E[g(X)] = ∫ g(x) dF(x) via integration by parts, letting dF = -d(1 - F)

    #2 from https://stats.stackexchange.com/a/18439/262661
       E[X] = ∫_[0, 1] F^{−1}(p) dp
       when F is invertible or a suitable pseudo-inverse can be defined.

    This function uses Method #1 above
    """
    if upper_limit < lower_limit:
        return -expectation_from_cdf(cdf, upper_limit, lower_limit)

    if lower_limit >= 0:
        positive_part, _ = quad(lambda x: 1 - cdf(x), lower_limit, upper_limit)
    elif lower_limit < 0 and upper_limit > 0:
        positive_part, _ = quad(lambda x: 1 - cdf(x), 0, upper_limit)
    else: #upper_limit <= 0:
        positive_part = 0

    if upper_limit <= 0:
        negative_part, _ = quad(cdf, lower_limit, upper_limit)
    elif lower_limit < 0 and upper_limit > 0:
        negative_part, _ = quad(cdf, lower_limit, 0)
    else: #lower_limit >= 0:
        negative_part = 0

    return positive_part - negative_part


def main():
    plot_uniform_unifom_fitting(uniform_symbols)
    plt.show()

if __name__ == "__main__":
    main()



