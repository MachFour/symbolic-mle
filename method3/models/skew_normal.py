"""
Fitting a skew-normal distribution by MLE parameter estimation to symbols

Since the MLE for a skew-normal distribution is not known in closed form,
a simulation-based method is used.

"""

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, skewnorm

from helper.centred_skew_normal import skewnorm_centered
from helper.utils import make_axis_values
from symbols.common import symbols_heuristic_min_max, plot_symbols
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol

SimFunc = Callable[[], np.ndarray]


def fit_to_data(generate_data: SimFunc, reps: int, use_centered_sn: bool = True) -> np.ndarray:
    """
    Simulates data using the given function and returns the average of ML parameter estimates
    for a univariate skew-normal distribution across all simulations.
    https://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    simulation_mles = np.zeros((reps, 3))
    for i in range(reps):
        data = generate_data()

        if use_centered_sn:
            sim_mle = skewnorm_centered.fit(data)

            print(
                f"γ_1^: {sim_mle[0]:7.3f},",
                f"μ^ = {sim_mle[1]:7.3f},",
                f"σ^ = {sim_mle[2]:7.3f}",
            )
            simulation_mles[i, :] = sim_mle
        else:
            # Use method of moments to derive initial estimate for shape (alpha) parameter to skew normal distribution
            # this requires inverting the population skewness equation, which depends only on alpha.

            skewness = skew(data, bias=False)
            # abs(delta) = sqrt(pi/2 * abs(skewness)^(2/3) / (abs(skewness)^(2/3) + (2 - pi/2)^(2/3))
            # sign(delta) = sign(skewness)
            sk23 = abs(skewness) ** (2 / 3)
            pi_2 = np.pi / 2
            delta_magnitude = np.sqrt(pi_2 * sk23 / (sk23 + (2 - pi_2) ** (2 / 3)))
            delta_estimate = np.clip(np.sign(skewness) * delta_magnitude, -0.99, 0.99)
            alpha_estimate = delta_estimate / np.sqrt(1 - delta_estimate ** 2)
            sim_mle = skewnorm.fit(data)  # no initial guess
            alpha_mle = sim_mle[0]

            dont_use = abs(alpha_estimate) < 1e3 and abs(alpha_mle / alpha_estimate) > 1e4
            print(
                f"sample skew: {skewness:8.3f},",
                f"α: {alpha_estimate:6.3f} -> {alpha_mle:15.3f},",
                f"ξ^ = {sim_mle[1]:7.3f},",
                f"ω^ = {sim_mle[2]:7.3f}",
                "[skip]" if dont_use else ""
            )

            # TODO exclude sample if alpha blows up
            #  -> if alpha_estimate < 1e4 and abs(alpha_mle / alpha_estimate) > 1e4
            simulation_mles[i, :] = np.nan * np.ones((1, 3)) if dont_use else sim_mle

    return np.nanmean(simulation_mles, 0)


def find_mle(
    symbols: Sequence[UniformSymbol | NormalSymbol],
    sim_reps: int = 1000,
    use_centered: bool = True,
    plot_intermediate: bool = False,
) -> tuple[float, float, float]:
    def sim_func() -> np.ndarray:
        # simulate points from the symbol distributions
        return np.hstack(list(s.rvs() for s in symbols))

    x_min, x_max = symbols_heuristic_min_max(symbols, 1.5)
    x = make_axis_values(x_min, x_max)
    shape, loc, scale = fit_to_data(sim_func, sim_reps, use_centered)

    if plot_intermediate:
        fig: plt.Figure = plt.figure(figsize=(10, 10))
        fig.suptitle(
            "Fitting Skew-Normal distribution to symbols via Method 3\n"
            "(Simulated sample expectation)", fontweight="bold")
        fig.subplots_adjust(hspace=0.5)
        ax0: plt.Axes = fig.add_subplot(2, 1, 1)
        ax1: plt.Axes = fig.add_subplot(2, 1, 2)
        ax0.set_title("Weighted PDFs of symbol distributions, and fitted skew-normal distribution")
        ax1.set_title("Unweighted CDFs of symbol distributions, and fitted skew-normal distribution")
        plot_symbols(symbols, ax0, ax1, x, class_size_weighted_pdf=True)
        dist = skewnorm_centered if use_centered else skewnorm
        ax0.plot(x, dist.pdf(x, shape, loc=loc, scale=scale), label="Fitted skew-normal distribution")
        ax1.plot(x, dist.cdf(x, shape, loc=loc, scale=scale), label="Fitted skew-normal distribution")
        ax0.legend()
        ax1.legend()

    return shape, loc, scale


def main():
    uniform_symbols_1 = (
        UniformSymbol(1, 2, 20),
        UniformSymbol(4, 6, 10),
        UniformSymbol(14, 16, 60),
    )
    # this one causes a flip of the skewness sign
    uniform_symbols_2 = (
        UniformSymbol(1, 2, 20),
        UniformSymbol(4, 6, 10),
        UniformSymbol(14, 16, 600),
    )

    normal_symbols_1 = (
        NormalSymbol(30, 10, 10),
        NormalSymbol(100, 10, 70),
        NormalSymbol(-100, 2, 10),
    )
    shape, loc, scale = find_mle(uniform_symbols_2, 10, plot_intermediate=True)
    print("Estimated parameters of skew-normal:")
    print(f"ξ = {loc}, ω = {scale}, α = {shape}")

    # normal_symbols_mle(normal_symbols_1, 10, plot_intermediate=True)
    plt.show()


if __name__ == "__main__":
    main()
