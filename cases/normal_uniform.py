from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from helper.utils import make_axis_values
from method3.models.normal import uniform_symbols_mean_mle, uniform_symbols_variance_mle
from symbols.normal import plot_normal_distribution, NormalSymbol
from symbols.uniform import UniformSymbol, uniform_symbols_heuristic_min_max, plot_uniform_symbols


def normal_uniform_method1(symbols: Iterable[UniformSymbol]) -> tuple[float, float]:
    return mu, sigma


def normal_uniform_method3(symbols: Iterable[UniformSymbol]) -> tuple[float, float]:
    mu = uniform_symbols_mean_mle(symbols)
    sigma = np.sqrt(uniform_symbols_variance_mle(symbols))
    return mu, sigma


def plot_normal_uniform_method(symbols: Iterable[UniformSymbol], method: int):
    match method:
        case 1:
            plot_normal_uniform_method_comparison(symbols, with_m1=True, with_m3=False)
        case 3:
            plot_normal_uniform_method_comparison(symbols, with_m1=False, with_m3=True)
        case _:
            raise ValueError(f"Unsupported method: {method}")


def plot_normal_uniform_method_comparison(symbols: Iterable[UniformSymbol], with_m1: bool = True, with_m3: bool = True):
    """
    Plot distributions of each uniform symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """
    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fitting a Normal distribution to Uniform symbols", fontweight = 'bold')

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Normal model pdf")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Normal model cdf")

    x_min, x_max = uniform_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    plot_uniform_symbols(symbols, ax0, ax1, x)

    if with_m1:
        mu1, sigma1 = normal_uniform_method1(symbols)
        plot_normal_distribution(NormalSymbol(mu1, sigma1, 1), x, ax0, ax1, linewidth=3, color='C5', label='Method 1')
        print(f"mu1 = {mu1}, sigma1 = {sigma1}")
    if with_m3:
        mu3, sigma3 = normal_uniform_method3(symbols)
        plot_normal_distribution(NormalSymbol(mu3, sigma3, 1), x, ax0, ax1, linewidth=3, color='C6', label='Method 3')
        print(f"mu3 = {mu3}, sigma3 = {sigma3}")

    ax0.legend()
    ax1.legend()


def main():
    uniform_symbols = (
        UniformSymbol(-15, 40, 4),
        UniformSymbol(-6, 50, 43),
        UniformSymbol(30, 45, 15),
        UniformSymbol(200, 210, 58)
    )

    plot_normal_uniform_method(uniform_symbols, 3)
    plt.show()


if __name__ == "__main__":
    main()
