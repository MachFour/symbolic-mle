# noinspection DuplicatedCode
from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import norm

from helper.utils import make_axis_values, expand_interval


class NormalSymbol:
    """
    Parameters:
    mu - mean
    sigma - standard deviation (nonzero; negative values will be rectified)
    n - number of points summarised by symbol
    """

    def __init__(self, mu: float, sigma: float, n: int):
        if sigma == 0:
            raise ValueError("sigma cannot be zero")
        elif sigma < 0:
            sigma = -sigma

        if not isinstance(n, int):
            raise TypeError(f"n must be integer but was {type(n)}")
        self.mu = mu
        self.sigma = sigma
        self.n = n

    # support tuple unpacking
    def __len__(self):
        return 3

    # support tuple unpacking
    def __getitem__(self, key):
        if type(key) == int:
            if key == 0:
                return self.mu
            elif key == 1:
                return self.sigma
            elif key == 2:
                return self.n
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError(f"indices must be integers, not {type(key)}")


def plot_normal_distribution(
    parameters: NormalSymbol,
    x_values: np.ndarray,
    pdf_axes: Axes,
    cdf_axes: Axes,
    pdf_weight: float = 1.0,
    **kwargs,
):
    x = x_values
    pdf_axes.plot(x, pdf_weight * norm.pdf(x, loc=parameters.mu, scale=parameters.sigma), **kwargs)
    cdf_axes.plot(x, norm.cdf(x, loc=parameters.mu, scale=parameters.sigma), **kwargs)


def plot_normal_symbols(
    symbols: Iterable[NormalSymbol],
    pdf_axes: Axes,
    cdf_axes: Axes,
    x_values: np.ndarray | None = None,
    label_axes: bool = True,
    class_size_weighted_pdf: bool = False,
):
    if label_axes:
        pdf_axes.set_xlabel("x")
        pdf_axes.set_ylabel("f(x)")

        cdf_axes.set_xlabel("x")
        cdf_axes.set_ylabel("F(x)")

    if x_values is None:
        x_min, x_max = normal_symbols_heuristic_min_max(symbols, expand_factor=1.5)
        x = make_axis_values(x_min, x_max)
    else:
        x = x_values

    N = sum(s.n for s in symbols)
    for s in symbols:
        pdf_weight = s.n / N if class_size_weighted_pdf else 1.0
        plot_normal_distribution(s, x, pdf_axes, cdf_axes, pdf_weight=pdf_weight)


def normal_symbols_heuristic_min_max(
    symbols: Iterable[NormalSymbol],
    expand_factor: float | None = None
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20
    for (mu, sigma, _) in symbols:
        x_min = min(x_min, mu - 3*sigma)
        x_max = max(x_max, mu + 3*sigma)

    if expand_factor:
        x_min, x_max = expand_interval(x_min, x_max, expand_factor)

    return x_min, x_max
