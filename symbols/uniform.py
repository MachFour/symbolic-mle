# noinspection DuplicatedCode
from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import uniform

from helper.utils import make_axis_values, expand_interval


class UniformSymbol:
    """
    Parameters:
    a - minimum value
    b - maximum value
    n - number of points summarised by symbol
    """

    def __init__(self, a: float, b: float, n: int):
        if b < a:
            raise ValueError(f"Uniform symbol must have b < a but a={a} and b={b}")
        if not isinstance(n, int):
            raise TypeError(f"n must be integer but was {type(n)}")
        self.a = a
        self.b = b
        self.n = n

    # support tuple unpacking
    def __len__(self):
        return 3

    # support tuple unpacking
    def __getitem__(self, key):
        if type(key) == int:
            if key == 0:
                return self.a
            elif key == 1:
                return self.b
            elif key == 2:
                return self.n
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError(f"indices must be integers, not {type(key)}")


def plot_uniform_distribution(
    params: UniformSymbol,
    x_values: np.ndarray,
    pdf_axes: Axes,
    cdf_axes: Axes,
    pdf_weight: float = 1.0,
    **kwargs,
):
    x = x_values
    pdf_axes.plot(x, pdf_weight * uniform.pdf(x, loc=params.a, scale=params.b - params.a), **kwargs)
    cdf_axes.plot(x, uniform.cdf(x, loc=params.a, scale=params.b - params.a), **kwargs)


def plot_uniform_symbols(
    symbols: Iterable[UniformSymbol],
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
        x_min, x_max = uniform_symbols_heuristic_min_max(symbols, expand_factor=1.5)
        x = make_axis_values(x_min, x_max)
    else:
        x = x_values

    N = sum(s.n for s in symbols)
    for s in symbols:
        pdf_weight = s.n / N if class_size_weighted_pdf else 1.0
        plot_uniform_distribution(s, x, pdf_axes, cdf_axes, pdf_weight=pdf_weight)


def uniform_symbols_heuristic_min_max(
    symbols: Iterable[UniformSymbol],
    expand_factor: float | None = None
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20
    for (a, b, _) in symbols:
        x_min = min(x_min, a)
        x_max = max(x_max, b)

    if expand_factor:
        x_min, x_max = expand_interval(x_min, x_max, expand_factor)

    return x_min, x_max
