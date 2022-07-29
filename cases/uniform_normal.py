import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from scipy.stats import norm

from helper.utils import make_axis_values, maximise_in_grid
from method1.uniform_model.mean_variance_symbols import mean_variance_symbolic_likelihood
from method3.models.uniform import normal_symbol_max_mle, normal_symbol_min_mle, \
    normal_symbol_max_cdf, normal_symbol_min_cdf, normal_symbols_heuristic_min_max
from symbols.normal import NormalSymbol


def plot_normal_symbols(
    symbols: tuple[NormalSymbol],
    pdf_axes: Axes,
    cdf_axes: Axes,
    x_values: np.ndarray | None = None
):
    pdf_axes.set_xlabel("x")
    pdf_axes.set_ylabel("f(x)")

    cdf_axes.set_xlabel("x")
    cdf_axes.set_ylabel("F(x)")

    if not x_values:
        x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
        x = make_axis_values(x_min, x_max)
    else:
        x = x_values

    for (mu, sigma, _) in symbols:
        pdf_axes.plot(x, norm.pdf(x, loc=mu, scale=sigma))
        cdf_axes.plot(x, norm.cdf(x, loc=mu, scale=sigma))


def uniform_normal_method1(
    symbols: tuple[NormalSymbol],
    precision: int = 4,
    plot_intermediate: bool = False
) -> tuple[float, float]:
    # search for maximising values of a and b for Uniform(a, b) model
    max_mu = max(s.mu for s in symbols)
    min_mu = min(s.mu for s in symbols)
    min_sigma = min(s.sigma for s in symbols)

    a_bounds = min(s.mu - 0.5*s.sigma*np.sqrt(s.n) for s in symbols), min_mu - 0.5*min_sigma
    b_bounds = max_mu + 0.5*min_sigma, max(s.mu + 0.5*s.sigma*np.sqrt(s.n) for s in symbols)

    grid_size = 20
    mc_samples_per_class = round(10**precision)

    # log likelihood function to maximise
    def ll(a2, b2):
        value = mean_variance_symbolic_likelihood(symbols, a2, b2, log=True, mc_samples_per_class=mc_samples_per_class)
        return value if value > -0.5e16 else -np.inf

    maximum_log_likelihood, max_a, max_b = maximise_in_grid(ll, a_bounds, b_bounds, grid_size, plot=plot_intermediate)

    return max_a, max_b


def uniform_normal_method3(symbols: tuple[NormalSymbol], plot_intermediate: bool = False) -> tuple[float, float]:
    mean_A = normal_symbol_min_mle(symbols)
    mean_B = normal_symbol_max_mle(symbols)

    if plot_intermediate:
        x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.2)
        x = make_axis_values(x_min, x_max)

        fig: plt.Figure = plt.figure(figsize=(10, 10))
        fig.suptitle("Fitting a Uniform distribution to Normal symbols: Method 3\n"
                     "CDFs of random variables A and B whose expectations are the model parameters", fontweight='bold')

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

        ax2.stem(mean_A, 1, linefmt='C5')
        ax3.stem(mean_B, 1, linefmt='C5')

    return mean_A, mean_B


def plot_uniform_normal_method1(symbols: tuple[NormalSymbol]):

    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fitting a Uniform distribution to Normal symbols: Method 1\n"
                 "(Direct likelihood analogue)", fontweight='bold')

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")
    a_mle, b_mle = uniform_normal_method1(symbols, plot_intermediate=False)

    plot_normal_symbols(symbols, ax0, ax1)

    for a in (ax0, ax1):
        a.stem(a_mle, a.get_ylim()[1], linefmt='C5')
        a.stem(b_mle, a.get_ylim()[1], linefmt='C5')


def plot_uniform_normal_method3(symbols: tuple[NormalSymbol]):
    """
    Plot distributions of each normal symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle("Fitting a Uniform distribution to Normal symbols: Method 3\n"
                 "(Simulated sample expectation)", fontweight='bold')

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")

    plot_normal_symbols(symbols, ax0, ax1)

    a_mle, b_mle = uniform_normal_method3(symbols, plot_intermediate=False)

    for a in (ax0, ax1):
        a.stem(a_mle, a.get_ylim()[1], linefmt='C5')
        a.stem(b_mle, a.get_ylim()[1], linefmt='C5')


def plot_uniform_normal_method1_method3_comparison(symbols: tuple[NormalSymbol]):
    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.8)
    fig.suptitle("Fitting a Uniform distribution to Normal symbols: Comparison of Method 1 and Method 3", fontweight='bold')

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")

    plot_normal_symbols(symbols, ax0, ax1)

    a_mle1, b_mle1 = uniform_normal_method1(symbols, precision=4, plot_intermediate=False)
    a_mle3, b_mle3 = uniform_normal_method3(symbols, plot_intermediate=False)

    for a in (ax0, ax1):
        stem_height = a.get_ylim()[1]
        y_coords = (0, stem_height)
        a.plot([a_mle1] * 2, y_coords, color='C5', marker='o', label='Method1')
        a.plot([b_mle1] * 2, y_coords, color='C5', marker='o', label='Method1')
        a.plot([a_mle3] * 2, y_coords, color='C6', marker='o', label='Method3')
        a.plot([b_mle3] * 2, y_coords, color='C6', marker='o', label='Method3')

        a.legend()


def main():
    # Each column forms one symbol
    m = [-15, -6, 10, 30]
    s = [  6,  7, 8, 6.5]
    n = [  4, 43, 2,  12]

    normal_symbols = tuple(NormalSymbol(*params) for params in zip(m, s, n))

    plot_uniform_normal_method1_method3_comparison(normal_symbols)
    plt.show()


if __name__ == "__main__":
    main()



