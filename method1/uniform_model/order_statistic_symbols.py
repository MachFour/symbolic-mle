from typing import Sequence

import numpy as np
from scipy.special import comb

from helper.utils import linspace_dense
from symbols.common import symbols_heuristic_min_max
from symbols.order_statistic import OrderStatisticSymbol


def order_statistic_class_likelihood(
    symbol: OrderStatisticSymbol,
    a: float,
    b: float,
    log: bool = False
) -> float:
    """Single-class symbolic likelihood evaluation function for a Uniform(a, b) model and single
    order-statistic symbol; implements 4.3.14 from thesis.

    To evaluate the likelihood for multiple symbols/classes, the likelihood values obtained from each class
    need to be multiplied, or added together in the case of log-likelihood

    :param symbol: order statistic data and class size
    :param a: minimum parameter for Uniform model; likelihood is evaluated at this point
    :param b: maximum parameter for Uniform model; likelihood is evaluated at this point
    :param log: Whether to make a log-likelihood function rather than ordinary one. This
        should be combined with other classes' log-likelihoods via summation rather than
        multiplication
    :return: (Log)-likelihood of the given symbol data for given μ, σ parameters.
    """
    # For 1 <= l < u <= n, a <= s_l <= s_u <= b, the symbolic class likelihood is given by
    # L_k(a, b, s_l, s_u, l, u, n) =
    #   C * (s_l - a)^(l-1) * (s_u - s_l)^(u - l - 1) * (b - s_u)^(n-u) / (b-a)^n
    # where
    # a, b are the parameters of the Uniform(a, b) model
    # l and u are respectively the lower and upper ordinals,
    # s_l and s_u are the observed order statistics of order l and u respectively.

    # The constant of proportionality is given by
    # C = n! / [ (l - 1)! * (u - l - 1)! (n - u)! = nCu * uCl * l * (u-l)

    n, l, u = symbol.n, symbol.lower_order, symbol.upper_order
    s_l, s_u = symbol.lower_stat, symbol.upper_stat

    if a > s_l or b < s_u:
        return -np.inf if log else 0

    C = comb(n, u, exact=True) * comb(u, l, exact=True) * l * (u - l)

    if log:
        l_term = np.log(s_l - a) * (l-1)
        difference_term = np.log(s_u - s_l) * (u - l - 1)
        u_term = np.log(b - s_u) * (n - u)
        denominator = np.log(b - a) * n
        return l_term + difference_term + u_term + np.log(C) - denominator
    else:
        l_term = (s_l - a)**(l - 1) if l - 1 > 0 else 1
        difference_term = (s_u - s_l)**(u - l - 1) if u - l - 1 > 0 else 1
        u_term = (b - s_u)**(n - u) if n - u > 0 else 1
        denominator = (b - a)**n
        return l_term * difference_term * u_term * C / denominator


def order_statistic_symbolic_likelihood(
    symbols: Sequence[OrderStatisticSymbol],
    a: float,
    b: float,
    log: bool = False
) -> float:
    if b < a:
        return -np.inf if log else 0
    if len(symbols) == 0:
        return 0 if log else 1

    terms = [order_statistic_class_likelihood(s, a, b, log=log) for s in symbols]
    return sum(terms) if log else np.product(terms)


# TODO test this method with nontrivial order statistic symbols
#  (i.e. not from uniform distribution)
def fit_uniform_model(
    symbols: Sequence[OrderStatisticSymbol],
    debug_print: bool = False
) -> tuple[float, float]:
    # maximise symbolic likelihood

    if all(s.lower_order == 1 for s in symbols):
        a_mle = min(s.lower_stat for s in symbols)
        if debug_print:
            print(f"Uniform model [M1]: lower statistic is min; setting a = {a_mle}")
    else:
        a_mle = np.nan
    if all(s.upper_order == s.n for s in symbols):
        b_mle = max(s.upper_stat for s in symbols)
        if debug_print:
            print(f"Uniform model [M1]: upper statistic is max; setting b = {b_mle}")
    else:
        b_mle = np.nan

    # we need a < s_l for all symbols and b > s_u
    max_a = min(s.lower_stat for s in symbols)
    min_b = max(s.upper_stat for s in symbols)

    # try to find a reasonable upper limit for search space
    min_a, max_b = symbols_heuristic_min_max(symbols, 2)

    N = sum(s.n for s in symbols)

    max_ll = -1e20

    def test_ll_candidate(_a: float, _b: float):
        nonlocal max_ll, a_mle, b_mle
        ll = order_statistic_symbolic_likelihood(symbols, _a, _b, log=True)
        if ll > max_ll:
            max_ll, a_mle, b_mle = ll, _a, _b
            if debug_print:
                print(f"Uniform model [M1]: found new max_ll = {max_ll} for a = {_a}, b = {_b}")

    def find_a(_b: float) -> float:
        # If l_k > 1 for some k, then given a maximising value â, b is given by
        # b = â + N / sum { (l_k - 1) / (s^(l)_k - â ) }
        def summation_term(s: OrderStatisticSymbol) -> float:
            return 0 if s.upper_stat == _b else (s.n - s.upper_order) / (_b - s.upper_stat)

        return _b - N / sum(summation_term(s) for s in symbols)

    def find_b(_a: float) -> float:
        # If u_k < n_k for some k, then a maximising value b̂, a is given by
        # a = b̂ - N / sum { (n_k - u_k) / (b̂ - s^(u)_k) }
        def summation_term(s: OrderStatisticSymbol) -> float:
            return 0 if s.lower_stat == _a else (s.lower_order - 1) / (s.lower_stat - _a)

        return _a + N / sum(summation_term(s) for s in symbols)

    if np.isnan(a_mle):
        # need to search for a
        if debug_print:
            print(f"Uniform model [M1]: Search values of a from {min_a} to {max_a}")
        for a in linspace_dense(min_a, max_a):
            test_ll_candidate(a, find_b(a))

    if np.isnan(b_mle):
        if debug_print:
            print(f"Uniform model [M1]: Search values of b from {min_b} to {max_b}")
        for b in linspace_dense(min_b, max_b):
            test_ll_candidate(find_a(b), b)

    if debug_print:
        print(f"Uniform model [M1] - a_mle = {a_mle}, b_mle = {b_mle}")

    return a_mle, b_mle
