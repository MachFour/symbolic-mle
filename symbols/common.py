from typing import Sequence

import numpy as np
from matplotlib.axes import Axes

from helper.utils import linspace_dense, expand_interval
from symbols.normal import NormalSymbol
from symbols.order_statistic import OrderStatisticSymbol
from symbols.uniform import UniformSymbol


def get_symbol_type(symbols: Sequence) -> type | None:
    if len(symbols) == 0:
        return None
    elif all(isinstance(s, UniformSymbol) for s in symbols):
        return UniformSymbol
    elif all(isinstance(s, NormalSymbol) for s in symbols):
        return NormalSymbol
    elif all(isinstance(s, OrderStatisticSymbol) for s in symbols):
        return OrderStatisticSymbol
    else:
        return None


def plot_symbols(
    symbols: Sequence[UniformSymbol] | Sequence[NormalSymbol],
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
        x_min, x_max = symbols_heuristic_min_max(symbols, expand_factor=1.5)
        x = linspace_dense(x_min, x_max)
    else:
        x = x_values

    N = sum(s.n for s in symbols)
    for s in symbols:
        pdf_weight = s.n / N if class_size_weighted_pdf else 1.0
        plot_as_distribution(s, x, pdf_axes, cdf_axes, pdf_weight=pdf_weight)


def symbols_heuristic_min_max(
    symbols: Sequence[UniformSymbol] | Sequence[NormalSymbol] | Sequence[OrderStatisticSymbol],
    expand_factor: float = 1.0
) -> tuple[float, float]:
    x_min = 1e20
    x_max = -1e20

    for s in symbols:
        symbol_min, symbol_max = s.heuristic_min_max()
        x_min = min(x_min, symbol_min)
        x_max = max(x_max, symbol_max)

    return expand_interval(x_min, x_max, expand_factor)


def plot_as_distribution(
    symbol: UniformSymbol | NormalSymbol,
    x_values: np.ndarray,
    pdf_axes: Axes,
    cdf_axes: Axes,
    pdf_weight: float = 1.0,
    **kwargs,
):
    x = x_values
    pdf_axes.plot(x, pdf_weight * symbol.pdf(x), **kwargs)
    cdf_axes.plot(x, symbol.cdf(x), **kwargs)
