from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from helper.utils import linspace_dense, maximise_in_grid
from method1.uniform_model.mean_variance_symbols import mean_variance_symbolic_likelihood
from method3.models.uniform import normal_symbol_max_mle, normal_symbol_min_mle, \
    normal_symbol_max_cdf, normal_symbol_min_cdf
from symbols.common import plot_symbols, symbols_heuristic_min_max
from symbols.normal import NormalSymbol


def uniform_normal_method1(
    symbols: Sequence[NormalSymbol],
    precision: int = 3,
    plot_intermediate: bool = False
) -> tuple[float, float]:
    if any(s.n == 1 for s in symbols):
        raise ValueError("Invalid mean-variance symbol with n == 1 for Method 1")
    # search for maximising values of a and b for Uniform(a, b) model
    max_mu = max(s.mu for s in symbols)
    min_mu = min(s.mu for s in symbols)
    min_sigma = min(s.sigma for s in symbols)

    a_bounds = min(s.mu - s.sigma * max(2, np.sqrt(s.n)) for s in symbols), min_mu - 0.5 * min_sigma
    b_bounds = max_mu + 0.5 * min_sigma, max(s.mu + s.sigma * max(2, np.sqrt(s.n)) for s in symbols)
    print(f"Uniform-normal [M1]: search for maximising values for a in {a_bounds}, b in {b_bounds}")

    grid_size = 20
    mc_samples_per_class = round(10 ** precision)

    # log likelihood function to maximise
    def ll(a2, b2):
        value = mean_variance_symbolic_likelihood(symbols, a2, b2, log=True, mc_samples_per_class=mc_samples_per_class)
        return value if value > -0.5e16 else -np.inf

    maximum_log_likelihood, max_a, max_b = maximise_in_grid(ll, a_bounds, b_bounds, grid_size, plot=plot_intermediate)

    print(f"Uniform-normal [M1]: fitted model is [{max_a}, {max_b}]")
    return max_a, max_b


def uniform_normal_method3(symbols: Sequence[NormalSymbol], plot_intermediate: bool = False) -> tuple[float, float]:
    mean_A = normal_symbol_min_mle(symbols)
    mean_B = normal_symbol_max_mle(symbols)

    if plot_intermediate:
        x_min, x_max = symbols_heuristic_min_max(symbols, 1.2)
        x = linspace_dense(x_min, x_max)

        fig: plt.Figure = plt.figure(figsize=(10, 10))
        fig.suptitle(
            "Fitting a Uniform distribution to Normal symbols: Method 3\n"
            "CDFs of random variables A and B whose expectations are the model parameters", fontweight='bold'
        )

        ax2: plt.Axes = fig.add_subplot(2, 1, 1)
        ax2.set_title("CDF of A; E[A] is fitted minimum")
        ax2.set_xlabel("a")
        ax2.set_ylabel("F_A(a)")

        ax3: plt.Axes = fig.add_subplot(2, 1, 2)
        ax3.set_title("CDF of B; E[B] is fitted maximum")
        ax3.set_xlabel("b")
        ax3.set_ylabel("F_B(b)")

        ax2.plot(x, normal_symbol_min_cdf(x, symbols))
        ax3.plot(x, normal_symbol_max_cdf(x, symbols))

        ax2.plot([mean_A] * 2, (0, 1), color='C5', marker='o')
        ax3.plot([mean_B] * 2, (0, 1), color='C5', marker='o')

    print(f"Uniform-normal [M3]: fitted model is [{mean_A}, {mean_B}]")
    return mean_A, mean_B


def plot_uniform_normal_method(symbols: Sequence[NormalSymbol], method: int):
    """
    Plot distributions of each normal symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    if method not in (1, 3):
        raise ValueError(f"Unrecognised method: {method}")

    if method == 1:
        title = "Fitting a Uniform distribution to Normal symbols: Method 1\n" \
                "(Direct likelihood analogue)"
        a_mle, b_mle = uniform_normal_method1(symbols, precision=3, plot_intermediate=False)
    else:  # method == 3:
        title = "Fitting a Uniform distribution to Normal symbols: Method 3\n" \
                "(Simulated sample expectation)"
        a_mle, b_mle = uniform_normal_method3(symbols, plot_intermediate=False)

    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.7)
    fig.suptitle(title, fontweight='bold')

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")

    plot_symbols(symbols, ax0, ax1)

    for a in (ax0, ax1):
        y_coords = (0, (a.get_ylim()[1]))
        a.plot([a_mle] * 2, y_coords, color='C5', marker='o')
        a.plot([b_mle] * 2, y_coords, color='C5', marker='o')


def plot_uniform_normal_method_comparison(
    symbols: Sequence[NormalSymbol],
    method1_precision: int
):
    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle(
        "Fitting a Uniform distribution to Normal symbols: Comparison of Method 1 and Method 3",
        fontweight='bold'
    )

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")

    plot_symbols(symbols, ax0, ax1)

    a_mle1, b_mle1 = uniform_normal_method1(symbols, precision=method1_precision, plot_intermediate=False)
    a_mle3, b_mle3 = uniform_normal_method3(symbols, plot_intermediate=False)

    for a in (ax0, ax1):
        y_coords = (0, (a.get_ylim()[1]))
        a.plot([a_mle1] * 2, y_coords, color='C5', marker='o', label='Method1')
        a.plot([b_mle1] * 2, y_coords, color='C5', marker='o')
        a.plot([a_mle3] * 2, y_coords, color='C6', marker='o', label='Method3')
        a.plot([b_mle3] * 2, y_coords, color='C6', marker='o')

        a.legend()


def main():
    # Each column forms one symbol
    # m = [-15, -6, 10, 30]
    # s = [  6,  7, 8, 6.5]
    # n = [  4, 43, 2,  12]
    normal_symbols = (
        NormalSymbol(mu=0, sigma=1, n=20),
        NormalSymbol(mu=50, sigma=100, n=50),
        NormalSymbol(mu=100, sigma=5, n=5)
    )

    plot_uniform_normal_method_comparison(normal_symbols, method1_precision=4)
    # plot_uniform_normal_method1(normal_symbols)
    plt.show()


if __name__ == "__main__":
    main()
