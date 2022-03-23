import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from uniform_uniform import make_axis_values, expectation_from_cdf

# symbol parameters grouped together
mu = [-15, -6, 10, 30]
sigma_2 = [40, 50,  60, 45]
n = [4, 43, 2, 12]

normal_symbols = tuple(zip(mu, sigma_2, n))

NormalSymbol = tuple[float, float, int]


def symbolic_min_cdf(x: np.ndarray, symbols: tuple[NormalSymbol]) -> np.ndarray:
    """
    The estimator of the minimum parameter of a uniform distribution fitted to
    uniform symbols is given by the expected value of random variable A with CDF:

    F_A(a) = 1 - prod { (1 - F_k(a))^n_k }, where F_k is the CDF of the k'th symbol
    """
    # sf is survival function = 1 - cdf
    product_terms = list(
        (scipy.stats.norm.sf(x, loc=mu, scale=np.sqrt(ss))**n
        for (mu, ss, n) in symbols)
    )
    return 1 - np.prod(np.asarray(product_terms), axis=0)


def symbolic_max_cdf(x: np.ndarray, symbols: tuple[NormalSymbol]) -> np.ndarray:
    """
    The estimator of the maximum parameter of a uniform distribution fitted to
    uniform symbols is given by the expected value of random variable B with CDF:

    F_B(b) = prod { (F_k(b))^n_k }, where F_k is the CDF of the kth symbol
    """
    product_terms = list(
        (scipy.stats.norm.cdf(x, loc=mu, scale=np.sqrt(ss))**n
        for (mu, ss, n) in symbols)
    )
    return np.prod(np.asarray(product_terms), axis=0)

def calculate_min_max(
    symbols: tuple[NormalSymbol],
    expand_factor: float|None = None
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20
    for (mu, ss, _) in symbols:
        x_min = min(x_min, mu - 10*np.sqrt(ss))
        x_max = max(x_max, mu + 10*np.sqrt(ss))

    if expand_factor:
        midpoint = (x_min + x_max) / 2
        new_range = (x_max - x_min) * expand_factor
        x_min = midpoint - new_range / 2
        x_max = midpoint + new_range / 2

    return (x_min, x_max)



def plot_uniform_normal_fitting(symbols: tuple[NormalSymbol]):
    """
    Plot distributions of each normal symbol, then the CDFs of the estimated
    minimum and maximum of a uniform model fitted to the symbols
    """

    x_min, x_max = calculate_min_max(symbols, expand_factor=1.2)
    x = make_axis_values(x_min, x_max)

    cdf_A = symbolic_min_cdf(x, symbols)
    mean_A = expectation_from_cdf(lambda x: symbolic_min_cdf(x, symbols), x_min, x_max)

    cdf_B = symbolic_max_cdf(x, symbols)
    mean_B = expectation_from_cdf(lambda x: symbolic_max_cdf(x, symbols), x_min, x_max)

    fig: plt.Figure = plt.figure()
    fig.suptitle("Fitting a Uniform distribution to Normal symbols")
    ax0: plt.Axes = fig.add_subplot(4, 1, 1)
    ax0.set_title("Densities of symbols, and fitted Uniform model min/max")
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")

    ax1: plt.Axes = fig.add_subplot(4, 1, 2)
    ax1.set_title("CDF of symbols, and fitted Uniform model min/max")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F(x)")

    for (mu, ss, _) in symbols:
        ax0.plot(x, scipy.stats.norm.pdf(x, loc=mu, scale=np.sqrt(ss)))
        ax1.plot(x, scipy.stats.norm.pdf(x, loc=mu, scale=np.sqrt(ss)))
    ax0.stem(mean_A, ax0.get_ylim()[1], linefmt='C5')
    ax0.stem(mean_B, ax0.get_ylim()[1], linefmt='C5')
    ax1.stem(mean_A, ax1.get_ylim()[1], linefmt='C5')
    ax1.stem(mean_B, ax1.get_ylim()[1], linefmt='C5')


    ax2: plt.Axes = fig.add_subplot(4, 1, 3)
    ax2.set_title("CDF of A - fitted minimum is E[A]")
    ax2.set_xlabel("a")
    ax2.set_ylabel("F_A(a)")


    ax3: plt.Axes = fig.add_subplot(4, 1, 4)
    ax3.set_title("CDF of B - fitted maximum is E[B]")
    ax3.set_xlabel("b")
    ax3.set_ylabel("F_B(b)")

    ax2.plot(x, cdf_A)
    ax3.plot(x, cdf_B)
    ax2.stem(mean_A, 1, linefmt='C5')
    ax3.stem(mean_B, 1, linefmt='C5')

    #fig.tight_layout()


def main():
    plot_uniform_normal_fitting(normal_symbols)
    plt.show()

if __name__ == "__main__":
    main()



