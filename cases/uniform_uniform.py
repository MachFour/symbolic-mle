from typing import Sequence

import matplotlib.pyplot as plt

from helper.utils import linspace_dense
from method3.models.uniform import uniform_symbol_min_cdf, uniform_symbol_max_cdf, \
    uniform_symbol_min_mle, uniform_symbol_max_mle
from symbols.common import symbols_heuristic_min_max, plot_symbols
from symbols.uniform import UniformSymbol


def uniform_uniform_method1(
    symbols: Sequence[UniformSymbol],
) -> tuple[float, float]:
    # maximiser of symbolic likelihood is just maximum allowed value of a
    # and minimum allowed value of b

    a_mle = min(s.a for s in symbols)
    b_mle = max(s.b for s in symbols)
    print(f"Uniform-uniform [M1] - a_mle = {a_mle}, b_mle = {b_mle}")

    return a_mle, b_mle


def uniform_uniform_method3(
    symbols: Sequence[UniformSymbol],
    plot_intermediate: bool = False
) -> tuple[float, float]:
    mean_A = uniform_symbol_min_mle(symbols)
    mean_B = uniform_symbol_max_mle(symbols)

    if plot_intermediate:
        fig: plt.Figure = plt.figure()

        ax2: plt.Axes = fig.add_subplot(2, 1, 1)
        ax2.set_title("CDF of A - fitted minimum is E[A]")
        ax2.set_xlabel("a")
        ax2.set_ylabel("F_A(a)")
        ax3: plt.Axes = fig.add_subplot(2, 1, 2)
        ax3.set_title("CDF of B - fitted maximum is E[B]")
        ax3.set_xlabel("b")
        ax3.set_ylabel("F_B(b)")

        x_min, x_max = symbols_heuristic_min_max(symbols, 1.2)
        x = linspace_dense(x_min, x_max)

        ax2.plot(x, uniform_symbol_min_cdf(x, symbols))
        ax2.plot([mean_A] * 2, (0, 1), color='C5')

        ax3.plot(x, uniform_symbol_max_cdf(x, symbols))
        ax3.plot([mean_B] * 2, (0, 1), color='C5')

    return mean_A, mean_B


def plot_uniform_uniform_method(symbols: Sequence[UniformSymbol], method: int):
    match method:
        case 1:
            plot_uniform_uniform_method_comparison(symbols, with_m1=True, with_m3=False)
        case 3:
            plot_uniform_uniform_method_comparison(symbols, with_m1=False, with_m3=True)
        case _:
            raise ValueError(f"Unsupported method: {method}")


def plot_uniform_uniform_method_comparison(
    symbols: Sequence[UniformSymbol],
    with_m1: bool = True,
    with_m3: bool = True
):
    """
    Plot distributions of each uniform symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = symbols_heuristic_min_max(symbols, 1.2)
    x = linspace_dense(x_min, x_max)

    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fitting a Uniform distribution to Uniform symbols - Methods 1 and 3 comparison")
    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")

    plot_symbols(symbols, ax0, ax1, x)

    a_mle1, b_mle1 = uniform_uniform_method1(symbols)
    a_mle3, b_mle3 = uniform_uniform_method3(symbols)

    for ax in (ax0, ax1):
        y_coords = (0, ax.get_ylim()[1])
        if with_m1:
            ax.plot([a_mle1] * 2, y_coords, color='C5', marker='o', label='Method1')
            ax.plot([b_mle1] * 2, y_coords, color='C5', marker='o')
        if with_m3:
            ax.plot([a_mle3] * 2, y_coords, color='C6', marker='o', label='Method3')
            ax.plot([b_mle3] * 2, y_coords, color='C6', marker='o')
        ax.legend()


def main():
    # each column defines one symbol
    uniform_symbols = (
        UniformSymbol(-15, 40, 4),
        UniformSymbol(-6, 50, 43),
        UniformSymbol(10, 60, 2),
        UniformSymbol(30, 45, 12)
    )

    plot_uniform_uniform_method_comparison(uniform_symbols)
    plt.show()


if __name__ == "__main__":
    main()
