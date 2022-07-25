import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from helper.utils import make_axis_values
from method3.models.uniform import normal_symbol_max_mle, normal_symbol_min_mle, \
    normal_symbol_max_cdf, normal_symbol_min_cdf, normal_symbols_heuristic_min_max
from symbols.normal import NormalSymbol


def plot_uniform_normal_fitting(symbols: tuple[NormalSymbol], method: int):
    match method:
        case 3:
            plot_uniform_normal_method3(symbols)
        case 1:
            plot_uniform_normal_method1(symbols)
        case _:
            print(f"plot_uniform_normal_fitting(): unsupported method ({method})")


def plot_uniform_normal_method1(symbols: tuple[NormalSymbol]):
    x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    fig: plt.Figure = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fitting a Uniform distribution to Normal symbols: Method 1\n"
                 "(Direct likelihood analogue)", fontweight='bold')


def plot_uniform_normal_method3(symbols: tuple[NormalSymbol]):
    """
    Plot distributions of each normal symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    fig: plt.Figure = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fitting a Uniform distribution to Normal symbols: Method 3\n"
                 "(Simulated sample expectation)", fontweight='bold')

    ax0: plt.Axes = fig.add_subplot(4, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")

    ax1: plt.Axes = fig.add_subplot(4, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F(x)")

    ax2: plt.Axes = fig.add_subplot(4, 1, 3)
    ax2.set_title("CDF of A; E[A] is fitted minimum")
    ax2.set_xlabel("a")
    ax2.set_ylabel("F_A(a)")

    ax3: plt.Axes = fig.add_subplot(4, 1, 4)
    ax3.set_title("CDF of B; E[B] is fitted maximum")
    ax3.set_xlabel("b")
    ax3.set_ylabel("F_B(b)")

    mean_A = normal_symbol_min_mle(symbols)
    mean_B = normal_symbol_max_mle(symbols)

    for (mu, ss, _) in symbols:
        ax0.plot(x, norm.pdf(x, loc=mu, scale=np.sqrt(ss)))
        ax1.plot(x, norm.cdf(x, loc=mu, scale=np.sqrt(ss)))

    for a in (ax0, ax1):
        a.stem(mean_A, a.get_ylim()[1], linefmt='C5')
        a.stem(mean_B, a.get_ylim()[1], linefmt='C5')

    ax2.plot(x, normal_symbol_min_cdf(x, symbols))
    ax3.plot(x, normal_symbol_max_cdf(x, symbols))

    ax2.stem(mean_A, 1, linefmt='C5')
    ax3.stem(mean_B, 1, linefmt='C5')



def main():
    # Each column forms one symbol
    m = [-15, -6, 10, 30]
    s = [  6,  7, 8, 6.5]
    n = [  4, 43, 2,  12]

    normal_symbols = tuple(NormalSymbol(*params) for params in zip(m, s, n))

    plot_uniform_normal_fitting(normal_symbols)
    plt.show()


if __name__ == "__main__":
    main()



