from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from helper.utils import linspace_dense
from method1.normal_model.order_statistic_symbols import order_statistic_symbolic_likelihood
from method2.normal_model.uniform_symbols import fit_to_uniform_symbols
from method3.normal_model.uniform_symbols import uniform_symbols_mean_mle, uniform_symbols_variance_mle
from symbols.common import symbols_heuristic_min_max, plot_symbols, plot_as_distribution
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def normal_uniform_method1(symbols: Sequence[UniformSymbol]) -> tuple[float, float]:
    order_stat_symbols = [s.to_order_statistic_symbol() for s in symbols]
    # search over mu and sigma
    # initial arguments
    N = sum(s.n for s in symbols)
    initial_mu = sum([(s.a + s.b)/2*s.n/N for s in symbols])
    initial_sigma = (max(s.b for s in symbols) - min(s.a for s in symbols))/3

    def fn_to_minimise(_x: np.ndarray) -> float:
        _mu, _sigma = _x[0], _x[1]
        return -order_statistic_symbolic_likelihood(order_stat_symbols, _mu, _sigma, log=True)

    x0 = np.array((initial_mu, initial_sigma))

    print(f"Normal-uniform [M1]: call minimisation with initial point x0 = {x0}")

    minimise_result = minimize(
        fn_to_minimise,
        x0=x0,
        method='nelder-mead',
        bounds=((-np.inf, np.inf), (1e-1, np.inf))  # constrain σ̂^2 >= 0.1
    )

    if minimise_result.success:
        x = minimise_result['x']
        print(f"Normal-uniform [M1]: minimisation completed successfully, x = {x}")
        return x[0], x[1]
    else:
        print(f"Normal-uniform [M1]: minimisation failed ({minimise_result['message']})")
        return np.nan, np.nan


def normal_uniform_method2(symbols: Sequence[UniformSymbol]) -> tuple[float, float]:
    return fit_to_uniform_symbols(symbols)


def normal_uniform_method3(symbols: Sequence[UniformSymbol]) -> tuple[float, float]:
    mu = uniform_symbols_mean_mle(symbols)
    sigma = np.sqrt(uniform_symbols_variance_mle(symbols))
    return mu, sigma


def plot_normal_uniform_method(symbols: Sequence[UniformSymbol], method: int):
    match method:
        case 1:
            plot_normal_uniform_method_comparison(symbols, with_m1=True, with_m2=False, with_m3=False)
        case 1:
            plot_normal_uniform_method_comparison(symbols, with_m1=False, with_m2=True, with_m3=False)
        case 3:
            plot_normal_uniform_method_comparison(symbols, with_m1=False, with_m2=False, with_m3=True)
        case _:
            raise ValueError(f"Unsupported method: {method}")


def plot_normal_uniform_method_comparison(
    symbols: Sequence[UniformSymbol],
    with_m1: bool = True,
    with_m2: bool = True,
    with_m3: bool = True
):
    """
    Plot distributions of each uniform symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """
    fig: plt.Figure = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Fitting a Normal distribution to Uniform symbols", fontweight='bold')

    ax0: plt.Axes = fig.add_subplot(2, 1, 1)
    ax0.set_title("Weighted PDFs of symbols and fitted Normal model PDF")

    ax1: plt.Axes = fig.add_subplot(2, 1, 2)
    ax1.set_title("CDFs of symbols (unweighted) and fitted Normal model CDF")

    x_min, x_max = symbols_heuristic_min_max(symbols, 1.5)
    x = linspace_dense(x_min, x_max)

    plot_symbols(symbols, ax0, ax1, x, class_size_weighted_pdf=True)

    if with_m1:
        mu1, sigma1 = normal_uniform_method1(symbols)
        plot_as_distribution(NormalSymbol(mu1, sigma1, 1), x, ax0, ax1, linewidth=3, color='C5', label='Method 1')
        print(f"mu1 = {mu1}, sigma1 = {sigma1}")
    if with_m2:
        mu2, sigma2 = normal_uniform_method2(symbols)
        plot_as_distribution(NormalSymbol(mu2, sigma2, 1), x, ax0, ax1, linewidth=3, color='C7', label='Method 2')
        print(f"mu2 = {mu2}, sigma2 = {sigma2}")
    if with_m3:
        mu3, sigma3 = normal_uniform_method3(symbols)
        plot_as_distribution(NormalSymbol(mu3, sigma3, 1), x, ax0, ax1, linewidth=3, color='C6', label='Method 3')
        print(f"mu3 = {mu3}, sigma3 = {sigma3}")

    ax0.legend()
    ax1.legend()


def main():
    uniform_symbols = (
        UniformSymbol(-1, 1, 2000),
    )
    uniform_symbols_2 = (
        #UniformSymbol(-15, 40, 4000),
        UniformSymbol(-6, 50, 4300),
        UniformSymbol(30, 45, 500),
        UniformSymbol(40, 50, 800),
        #UniformSymbol(50, 1000, 4),
    )

    uniform_symbols_3 = (
        UniformSymbol(-15, 40, 4000),
        UniformSymbol(-6, 50, 4300),
        UniformSymbol(30, 45, 500),
        UniformSymbol(40, 50, 800),
        UniformSymbol(50, 1000, 4000),
    )

    #plot_normal_uniform_method(uniform_symbols, 1)
    plot_normal_uniform_method_comparison(uniform_symbols)
    plt.show()


if __name__ == "__main__":
    main()
