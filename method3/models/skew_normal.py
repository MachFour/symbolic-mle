"""
Fitting a skew-normal distribution by MLE parameter estimation to symbols

Since the MLE for a skew-normal distribution is not known in closed form,
a simulation-based method is used.

"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, skewnorm, uniform, norm

from helper.centred_skew_normal import skewnorm_centered
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol

SimFunc = Callable[[], np.ndarray]


def get_x_axis(symbols_min: float, symbols_max: float, expand_factor: float = 1.5) -> np.ndarray:
    plot_mid = (symbols_min + symbols_max) / 2
    plot_half_length = (symbols_max - symbols_min) / 2 * expand_factor

    x = np.linspace(plot_mid - plot_half_length, plot_mid + plot_half_length, round(plot_half_length*10))


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

            print(f"γ_1^: {sim_mle[0]:7.3f},",
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
            sk23 = abs(skewness)**(2/3)
            pi_2 = np.pi / 2
            delta_magnitude = np.sqrt(pi_2 * sk23 / (sk23 + (2 - pi_2)**(2/3)))
            delta_estimate = np.clip(np.sign(skewness) * delta_magnitude, -0.99, 0.99)
            alpha_estimate = delta_estimate / np.sqrt(1 - delta_estimate**2)
            sim_mle = skewnorm.fit(data)  # no initial guess
            alpha_mle = sim_mle[0]

            dont_use = abs(alpha_estimate) < 1e3 and abs(alpha_mle / alpha_estimate) > 1e4
            print(f"sample skew: {skewness:8.3f},",
                  f"α: {alpha_estimate:6.3f} -> {alpha_mle:15.3f},",
                  f"ξ^ = {sim_mle[1]:7.3f},",
                  f"ω^ = {sim_mle[2]:7.3f}",
                  "[skip]" if dont_use else ""
                  )

            # TODO exclude sample if alpha blows up
            #  -> if alpha_estimate < 1e4 and abs(alpha_mle / alpha_estimate) > 1e4
            simulation_mles[i, :] = np.nan * np.ones((1, 3)) if dont_use else sim_mle

    return np.nanmean(simulation_mles, 0)


def uniform_symbols_mle(symbols: tuple[UniformSymbol, ...], sim_reps: int = 1000, use_centered: bool = True) -> float:
    def sim_func() -> np.ndarray:
        # simulate points from the symbol distributions
        return np.hstack(list(uniform.rvs(loc=s.a, scale=s.b - s.a, size=s.n) for s in symbols))

    # plot the distribution and points
    symbols_min = min(s.a for s in symbols)
    symbols_max = max(s.b for s in symbols)
    x = get_x_axis(symbols_min, symbols_max)

    shape, loc, scale = fit_to_data(sim_func, sim_reps, use_centered)
    pdf_func = skewnorm_centered.pdf if use_centered else skewnorm.pdf
    y = pdf_func(x, shape, loc=loc, scale=scale)

    # visualise uniform symbols as stems between symbol min and max,
    # with number of stems equal to number of points in the symbol,
    # and height inversely proportional to the distance between min and max
    # multiplied by the relative symbol size (n) among all symbols
    N = sum(s.n for s in symbols)
    for (a, b, n) in symbols:
        stem_data_x = np.linspace(a, b, n)
        stem_data_y = np.ones(stem_data_x.shape)/(b-a) * n / N
        plt.stem(stem_data_x, stem_data_y, linefmt='r-', markerfmt='ro')
    plt.plot(x, y)
    plt.show()


def normal_symbols_mle(symbols: tuple[NormalSymbol, ...], sim_reps: int = 1000, use_centered: bool = True) -> float:
    def sim_func() -> np.ndarray:
        # simulate points from the symbol distributions
        return np.hstack(list(norm.rvs(loc=s.mu, scale=s.sigma, size=s.n) for s in symbols))

    # plot the distribution and points
    symbols_min = min(s.mu - 3*s.sigma for s in symbols)
    symbols_max = max(s.mu + 3*s.sigma for s in symbols)
    x = get_x_axis(symbols_min, symbols_max)

    shape, loc, scale = fit_to_data(sim_func, sim_reps, use_centered)

    pdf_func = skewnorm_centered.pdf if use_centered else skewnorm.pdf
    y = pdf_func(x, shape, loc=loc, scale=scale)

    N = sum(s.n for s in symbols)
    for (mu, sigma, n) in symbols:
        stem_data_y = norm.pdf(x, loc=mu, scale=sigma) * n / N
        plt.plot(x, stem_data_y, 'r-')
    plt.plot(x, y)
    plt.show()


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
    #uniform_symbols_mle(uniform_symbols_1, 10)

    normal_symbols_1 = (
        NormalSymbol(30, 10, 10),
        NormalSymbol(100, 10, 70),
        NormalSymbol(-100, 2, 10),
    )
    normal_symbols_mle(normal_symbols_1, 10)


if __name__ == "__main__":
    main()
