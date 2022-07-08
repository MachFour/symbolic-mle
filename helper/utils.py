from typing import Callable

import numpy as np
from scipy.integrate import quad


def make_axis_values(plot_min: float, plot_max: float, density: float = 10) -> np.ndarray:
    return np.linspace(plot_min, plot_max, round((plot_max - plot_min) * density))


def expectation_from_cdf(
        cdf: Callable[[np.ndarray], np.ndarray],
        lower_limit: float = -np.Infinity,
        upper_limit: float = np.Infinity
) -> float:
    """
    Let X be a continuous, real-valued random variable with CDF F(x).
    This function computes the expected value of X via numerical integration.
    There are at least two methods / equations for this.

    #1 from comment in https://stats.stackexchange.com/a/222497/262661
        E[X] = ∫_{[0, ∞)} (1−F(x)) dx − ∫_{(−∞, 0]} F(x) dx.
        whenever the expectation is finite. This apparently can be derived from
        E[g(X)] = ∫ g(x) dF(x) via integration by parts, letting dF = -d(1 - F)

    #2 from https://stats.stackexchange.com/a/18439/262661
       E[X] = ∫_[0, 1] F^{−1}(p) dp
       when F is invertible or a suitable pseudo-inverse can be defined.

    This function uses Method #1 above
    """
    if upper_limit < lower_limit:
        return -expectation_from_cdf(cdf, upper_limit, lower_limit)

    if lower_limit >= 0:
        # noinspection PyTupleAssignmentBalance
        positive_part, _ = quad(lambda x: 1 - cdf(x), lower_limit, upper_limit)
    elif lower_limit < 0 < upper_limit:
        # noinspection PyTupleAssignmentBalance
        positive_part, _ = quad(lambda x: 1 - cdf(x), 0, upper_limit)
    else:  # upper_limit <= 0:
        positive_part = 0

    if upper_limit <= 0:
        # noinspection PyTupleAssignmentBalance,PyTypeChecker
        negative_part, _ = quad(cdf, lower_limit, upper_limit)
    elif lower_limit < 0 < upper_limit:
        # noinspection PyTupleAssignmentBalance,PyTypeChecker
        negative_part, _ = quad(cdf, lower_limit, 0)
    else:  # lower_limit >= 0:
        negative_part = 0

    return positive_part - negative_part
