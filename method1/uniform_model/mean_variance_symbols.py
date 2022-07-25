from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, norm


# generates points on the intersection of the n-hypersphere of radius r,
# intersected with the hyperplane of x_1 + ... + x_n = 0
from symbols.normal import NormalSymbol


def generate_points(samples: int, n: int, r: float, plot: bool = False) -> np.ndarray:
    # Generate samples from standard n-variate Normal distribution
    # Each sample forms 1 column
    x = norm.rvs(size=(n, samples))
    # normalise columns to have L2 norm of 2
    y = x / np.linalg.norm(x, axis=0, keepdims=True)
    # project onto plane through origin perpendicular to (1, 1, ..., 1) (normalised)
    normal_vector = np.ones((n, 1)) / np.sqrt(n)
    # take dot product with normal vector (1/sqrt(n) * vector of ones)
    dot_product_with_normal_vector = np.sum(y / np.sqrt(n), axis=0)
    # perpendicular component is dot product times normal vector
    # y is already normalised, otherwise we would need to normalise it
    normal_component_y = dot_product_with_normal_vector * normal_vector
    z = y - normal_component_y
    # now normalise back out to scaled sphere, now already in the plane
    v = z / np.linalg.norm(z, axis=0, keepdims=True) * r

    if plot:
        fig: plt.Figure = plt.figure(dpi=200)
        if n == 3:
            ax: plt.Axes = fig.add_subplot(projection="3d")
            ax.scatter(v[0, :], v[1, :], v[2, :])
        elif n == 2:
            ax: plt.Axes = fig.add_subplot()
            ax.scatter(v[0, :], v[1, :])
        else:
            f"Plotting unsupported in {n} dimensions"

    return v


def mean_variance_class_likelihood(
    class_mu: float,
    class_sigma: float,
    class_size: int,
    a: float,
    b: float,
    log: bool = False,
    num_samples: int = 100,
) -> float:
    """
    Adapted from https://gitlab.com/machfour/thesis-scripts/-/blob/master/uniform-normal-comparison.R

    :param class_mu:
    :param class_sigma:
    :param class_size:
    :param a:
    :param b:
    :param log: whether to return the log-likelihood or just the normal one
    :param num_samples: Number of Monte-Carlo simulation samples to use
    :return:
    """
    # assumes a < b
    m = class_mu
    s = np.abs(class_sigma)
    a_dash = (a - m) / s
    b_dash = (b - m) / s

    n_k = class_size

    n_2 = n_k // 2
    # the following integral approximation is correct up to a constant depending on n_k
    if a_dash >= -1 / np.sqrt(n_k - 1) or b_dash <= 1 / np.sqrt(n_k - 1):
        # this check strengthens the more trivial requirement of !(a_dash >= 0 || b_dash <= 0)
        # for the integral to be nonzero
        # these conditions can be shown using geometric arguments from 3D case
        integral_approx = 0
    elif (b_dash - a_dash) < n_k / np.sqrt(n_2 * (n_k - n_2)):
        # Ithink this is right? n_2 might substitute for 1 in the case above too,
        # since they coincide for n=3
        integral_approx = 0
    elif a_dash <= -np.sqrt(n_k) and b_dash >= np.sqrt(n_k):
        # all points will lie in the sphere
        integral_approx = 1
    else:
        points = generate_points(num_samples, n_k, n_k)
        # determine which points/samples lie in the [a_dash, b_dash] hypercube
        # equivalent to checking if all coordinates in each column lie in [a_dash, b_dash]
        points_in_hypercube = np.logical_and(a_dash <= points, points <= b_dash).all(axis=0)
        integral_approx = sum(points_in_hypercube) / num_samples

    if log:
        return np.log(integral_approx) - n_k * np.log(b - a)
    else:
        return integral_approx / (b - a) ** n_k


def mean_variance_symbolic_likelihood(
    symbols: Iterable[NormalSymbol],
    a: float,
    b: float,
    log: bool = False,
    num_samples: int = 100,
) -> float:
    



def main():
    generate_points(2000, 3, 9, plot=True)
    plt.show()


if __name__ == "__main__":
    main()
