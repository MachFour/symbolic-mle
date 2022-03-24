import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform

from models.normal import uniform_symbols_mean_mle, uniform_symbols_variance_mle
from models.uniform import uniform_symbols_heuristic_min_max
from symbols.uniform import UniformSymbol
from utils import make_axis_values


def plot_normal_uniform_fitting(symbols: tuple[UniformSymbol]):
    """
    Plot distributions of each uniform symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = uniform_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    mu = uniform_symbols_mean_mle(symbols)
    sigma_2 = uniform_symbols_variance_mle(symbols)

    fig: plt.Figure = plt.figure()
    fig.suptitle("Fitting a Normal distribution to Uniform symbols")
    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Normal model pdf")
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Normal model cdf")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F(x)")

    for (a, b, _) in symbols:
        ax0.plot(x, uniform.pdf(x, loc=a, scale=b - a))
        ax0.plot(x, norm.pdf(x, loc=mu, scale=np.sqrt(sigma_2)), linewidth=3)
        ax1.plot(x, uniform.cdf(x, loc=a, scale=b - a))
        ax1.plot(x, norm.cdf(x, loc=mu, scale=np.sqrt(sigma_2)), linewidth=3)


def main():
    # each column makes one symbol
    a = [-15, -6, 10, 30, 200]
    b = [40, 50, 60, 45, 210]
    n = [4, 43, 63, 15, 10]

    uniform_symbols = tuple(UniformSymbol(*params) for params in zip(a, b, n))

    plot_normal_uniform_fitting(uniform_symbols)
    plt.show()


if __name__ == "__main__":
    main()
