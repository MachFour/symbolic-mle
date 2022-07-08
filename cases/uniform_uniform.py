import matplotlib.pyplot as plt
from scipy.stats import uniform

from helper.utils import make_axis_values
from method3.models.uniform import uniform_symbol_min_cdf, uniform_symbol_max_cdf, \
    uniform_symbol_min_mle, uniform_symbol_max_mle, uniform_symbols_heuristic_min_max
from symbols.uniform import UniformSymbol


def plot_uniform_uniform_fitting(symbols: tuple[UniformSymbol], method: int):
    """
    Plot distributions of each uniform symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = uniform_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

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

    ax2: plt.Axes = fig.add_subplot(4, 1, 3)
    ax2.set_title("CDF of A - fitted minimum is E[A]")
    ax2.set_xlabel("a")
    ax2.set_ylabel("F_A(a)")

    ax3: plt.Axes = fig.add_subplot(4, 1, 4)
    ax3.set_title("CDF of B - fitted maximum is E[B]")
    ax3.set_xlabel("b")
    ax3.set_ylabel("F_B(b)")

    for (a, b, _) in symbols:
        ax0.plot(x, uniform.pdf(x, loc=a, scale=b - a))
        ax1.plot(x, uniform.cdf(x, loc=a, scale=b - a))

    mean_A = uniform_symbol_min_mle(symbols)
    mean_B = uniform_symbol_max_mle(symbols)

    ax0.stem(mean_A, ax0.get_ylim()[1], linefmt='C5')
    ax0.stem(mean_B, ax0.get_ylim()[1], linefmt='C5')

    ax1.stem(mean_A, ax1.get_ylim()[1], linefmt='C5')
    ax1.stem(mean_B, ax1.get_ylim()[1], linefmt='C5')

    ax2.plot(x, uniform_symbol_min_cdf(x, symbols))
    ax2.stem(mean_A, 1, linefmt='C5')

    ax3.plot(x, uniform_symbol_max_cdf(x, symbols))
    ax3.stem(mean_B, 1, linefmt='C5')


def main():
    # each column defines one symbol
    a = [-15, -6, 10, 30]
    b = [40, 50, 60, 45]
    n = [4, 43, 2, 12]

    uniform_symbols = tuple(UniformSymbol(*params) for params in zip(a, b, n))

    plot_uniform_uniform_fitting(uniform_symbols)
    plt.show()


if __name__ == "__main__":
    main()
