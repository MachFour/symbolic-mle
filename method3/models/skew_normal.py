"""
Fitting a skew-normal distribution by MLE parameter estimation to symbols

Mean

The estimator of the mean parameter of a normal distribution in terms
of symbol distributions is given by the weighted sum of expected values of each symbol distribution:

mu_S = sum { n_k/N * m_k } where m_k is the expected value of distribution F_k,
and n_k is the number of points in the symbol

----------------------------------------------------------------

Variance

The estimator of the variance parameter of a normal distribution
in terms of symbol distributions is given by the following weighted sum:

s^2_s = sum { n_k/N * ((1 - 1/N) s^2_k + (m_k - m)^2) }
where n_k is the number of points in the symbol, m_k and s^2_k are respectively
the mean and variance of F_k, and m is the symbolic mean (see above)

NOTE: this estimator is a little bit biased, just like the usual MLE of variance
for a normal distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, skewnorm, uniform

from centred_skew_normal import skewnorm_centered
from symbols.normal import NormalSymbol
from symbols.uniform import UniformSymbol


def uniform_symbols_mle(symbols: tuple[UniformSymbol, ...], sim_reps: int = 1000, use_centered: bool = True) -> float:
    """Returns a triple representing the MLE of skew-normal parameters
    for the given Uniform symbols
    The parametrisation used is location (ξ), scale (ω), shape (α)
    https://en.wikipedia.org/wiki/Skew_normal_distribution

    Since there is no closed-form MLE for the skew normal distribution,
    a simulation-based method is used.
    """

    simulation_mles = np.zeros((sim_reps, 3))
    for i in range(sim_reps):
        # simulate points from the symbol distributions
        sim_data = np.hstack(list(uniform.rvs(loc=s.a, scale=s.b - s.a, size=s.n) for s in symbols))

        if use_centered:
            sim_mle = skewnorm_centered.fit(sim_data)

            print(f"γ_1^: {sim_mle[0]:7.3f},",
                  f"μ^ = {sim_mle[1]:7.3f},",
                  f"σ^ = {sim_mle[2]:7.3f}",
                  )
            simulation_mles[i, :] = sim_mle
        else:
            # Use method of moments to derive initial estimate for shape (alpha) parameter to skew normal distribution
            # this requires inverting the population skewness equation, which depends only on alpha.

            skewness = skew(sim_data, bias=False)
            # abs(delta) = sqrt(pi/2 * abs(skewness)^(2/3) / (abs(skewness)^(2/3) + (2 - pi/2)^(2/3))
            # sign(delta) = sign(skewness)
            sk23 = abs(skewness)**(2/3)
            pi_2 = np.pi / 2
            delta_magnitude = np.sqrt(pi_2 * sk23 / (sk23 + (2 - pi_2)**(2/3)))
            delta_estimate = np.clip(np.sign(skewness) * delta_magnitude, -0.99, 0.99)
            alpha_estimate = delta_estimate / np.sqrt(1 - delta_estimate**2)
            sim_mle = skewnorm.fit(sim_data)  # no initial guess
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

    print(simulation_mles.shape)
    overall_mle = np.nanmean(simulation_mles, 0)
    # plot the distribution and points
    x = np.linspace(0, 1.5*max(s.b for s in symbols), 100)
    shape, loc, scale = overall_mle[:]
    if use_centered:
        y = skewnorm_centered.pdf(x, shape, loc=loc, scale=scale)
    else:
        y = skewnorm.pdf(x, shape, loc=loc, scale=scale)
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

    print(overall_mle)
    plt.show()


def normal_symbols_mle(symbols: tuple[NormalSymbol]) -> float:
    pass


def main():
    uniform_symbols = (
        UniformSymbol(1, 20, 20),
        UniformSymbol(49, 60, 10),
    )
    uniform_symbols_mle(uniform_symbols, 10)


if __name__ == "__main__":
    main()
