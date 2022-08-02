from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from helper.utils import make_axis_values
from method3.models.normal import normal_symbols_variance_mle, normal_symbols_mean_mle, VarianceBiasType
from symbols.common import plot_normal_symbols, symbols_heuristic_min_max
from symbols.normal import NormalSymbol, plot_normal_distribution


def normal_normal_method1(symbols: Sequence[NormalSymbol]) -> tuple[float, float]:
    mu = normal_symbols_mean_mle(symbols)
    sigma = np.sqrt(normal_symbols_variance_mle(symbols, VarianceBiasType.M1_BIASED_SUMMARY))
    return mu, sigma


def normal_normal_method3(symbols: Sequence[NormalSymbol]) -> tuple[float, float]:
    mu = normal_symbols_mean_mle(symbols)
    sigma = np.sqrt(normal_symbols_variance_mle(symbols, VarianceBiasType.M3_BIASED_ESTIMATOR))
    return mu, sigma


def plot_normal_normal_method(symbols: Sequence[NormalSymbol], method: int):
    match method:
        case 1:
            plot_normal_normal_method_comparison(symbols, with_m1=True, with_m3=False)
        case 3:
            plot_normal_normal_method_comparison(symbols, with_m1=False, with_m3=True)
        case _:
            raise ValueError(f"Unsupported method: {method}")


def plot_normal_normal_method_comparison(symbols: Sequence[NormalSymbol], with_m1: bool = True, with_m3: bool = True):
    """
    Plot distributions of each normal symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = symbols_heuristic_min_max(symbols, 1.2)
    x = make_axis_values(x_min, x_max)

    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.suptitle("Fitting a Normal distribution to Normal symbols", fontweight="bold")
    fig.subplots_adjust(hspace=0.5)
    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax1: plt.Axes = fig.add_subplot(2, 1, 2)

    plot_normal_symbols(symbols, ax0, ax1, x)

    ax0.set_title("PDF of symbols, and fitted Normal model pdf")
    ax1.set_title("CDF of symbols, and fitted Normal model cdf")

    if with_m1:
        mu1, sigma1 = normal_normal_method1(symbols)
        plot_normal_distribution(NormalSymbol(mu1, sigma1, 1), x, ax0, ax1, linewidth=3, color='C5', label='Method 1')
        print(f"mu1 = {mu1}, sigma1 = {sigma1}")
    if with_m3:
        mu3, sigma3 = normal_normal_method3(symbols)
        plot_normal_distribution(NormalSymbol(mu3, sigma3, 1), x, ax0, ax1, linewidth=3, color='C6', label='Method 3')
        print(f"mu3 = {mu3}, sigma3 = {sigma3}")

    ax0.legend()
    ax1.legend()


def main():
    # Each column forms one symbol
    normal_symbols = (
        NormalSymbol(-15, 6, 4),
        NormalSymbol(-6, 7, 43),
        NormalSymbol(10, 8, 23),
        NormalSymbol(30, 6.5, 12)
    )

    plot_normal_normal_method_comparison(normal_symbols)
    plt.show()


if __name__ == "__main__":
    main()
