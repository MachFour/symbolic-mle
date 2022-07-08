import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from helper.utils import make_axis_values
from method3.models.normal import normal_symbols_variance_mle, normal_symbols_mean_mle
from method3.models.uniform import normal_symbols_heuristic_min_max
from symbols.normal import NormalSymbol


def plot_normal_normal_fitting(symbols: tuple[NormalSymbol], method: int):
    """
    Plot distributions of each normal symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    fig: plt.Figure = plt.figure()
    fig.suptitle("Fitting a Normal distribution to Normal symbols")
    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Normal model cdf")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F(x)")

    mu = normal_symbols_mean_mle(symbols)
    sigma_2 = normal_symbols_variance_mle(symbols)

    for s in symbols:
        ax0.plot(x, norm.pdf(x, loc=s.mu, scale=s.sigma))
        ax0.plot(x, norm.pdf(x, loc=mu, scale=np.sqrt(sigma_2)), linewidth=3)

        ax1.plot(x, norm.cdf(x, loc=s.mu, scale=s.sigma))
        ax1.plot(x, norm.cdf(x, loc=mu, scale=np.sqrt(sigma_2)), linewidth=3)


def main():
    # Each column forms one symbol
    m = [-15, -6, 10, 30]
    s = [  6,  7, 8, 6.5]
    n = [  4, 43, 2,  12]

    normal_symbols = tuple(NormalSymbol(*params) for params in zip(m, s, n))

    plot_normal_normal_fitting(normal_symbols)
    plt.show()


if __name__ == "__main__":
    main()
