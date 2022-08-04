from typing import Sequence

import numpy as np
from scipy.special import comb
from scipy.stats import norm

from symbols.order_statistic import OrderStatisticSymbol


def order_statistic_class_likelihood(
    symbol: OrderStatisticSymbol,
    mu: float,
    sigma: float,
    log: bool = False
) -> float:
    """Single-class symbolic likelihood evaluation function for a Gaussian(μ, σ^2) model and single
    order-statistic symbol. Implements 4.3.12 from thesis, specialised to the univariate Normal distribution.

    To evaluate the likelihood for multiple symbols/classes, the likelihood values obtained from each class
    need to be multiplied, or added together in the case of log-likelihood

    :param symbol: order statistic data and class size
    :param mu: mean parameter for Gaussian model; likelihood is evaluated at this point
    :param sigma: square root of variance parameter for Gaussian model; likelihood is evaluated at this point
    :param log: Whether to make a log-likelihood function rather than ordinary one. This
        should be combined with other classes' log-likelihoods via summation rather than
        multiplication
    :return: (Log)-likelihood of the given symbol data for given μ, σ parameters.
    """
    s_l, s_u = symbol.lower_stat, symbol.upper_stat
    l, u, n = symbol.lower_order, symbol.upper_order, symbol.n

    # For 1 <= l < u <= n, s_l <= s_u, the symbolic class likelihood is given by

    # L_k(θ, s_l, s_u, l, u, n) =
    # C * G(s_l; θ)^(l-1) * [G(s_u; θ) - G(s_l; θ)]^(u - l - 1) * (1 - G(s_u; θ))^(n-u) * g(s_u; θ) * g(s_l; θ)

    # where
    # θ is the parameter (vector) to be chosen,
    # G(.; θ) is the model CDF,
    # g(.; θ) is the model PDF,
    # l and u are respectively the lower and upper ordinals,
    # s_l and s_u are the observed order statistics of order l and u respectively.

    # The constant of proportionality is given by
    # C = n! / [ (l - 1)! * (u - l - 1)! (n - u)! = nCu * uCl * l * (u-l)

    C = comb(n, u, exact=True) * comb(u, l, exact=True) * l * (u - l)

    model = norm(loc=mu, scale=sigma)

    if log:
        # l_k(θ, s, n) =
        #   log G(s_l; θ) * (l-1)
        l_term = model.logcdf(s_l) * (l - 1)
        # + log (1 - G(s_u; θ)) * (n-u)
        u_term = model.logsf(s_u) * (n - u)
        # + log [G(s_u; θ) - G(s_l; θ)] * (u - l - 1)
        difference_term = np.log(max(model.cdf(s_u) - model.cdf(s_l), 1e-16)) * (u - l - 1)
        # + log g(s_l; θ) + log g(s_u; θ)
        pdf_term = model.logpdf(s_l) + model.logpdf(s_u)

        return l_term + difference_term + u_term + pdf_term + np.log(C)
    else:
        # L_k(θ, s, n) =
        #   G(s_l; θ)^(l-1)
        l_term = model.cdf(s_l)**(l - 1) if l - 1 > 0 else 1
        # * (1 - G(s_u; θ))^(n-u)
        u_term = model.sf(s_u)**(n - u) if n - u > 0 else 1
        # * [G(s_u; θ) - G(s_l; θ)]^(u - l - 1)
        difference_term = (model.cdf(s_u) - model.cdf(s_l))**(u - l - 1) if u - l - 1 > 0 else 1
        # * g(s_l; θ) * g(s_u; θ)
        pdf_term = model.pdf(s_l) * model.pdf(s_u)

        return l_term * difference_term * u_term * pdf_term * C


def order_statistic_symbolic_likelihood(
    symbols: Sequence[OrderStatisticSymbol],
    mu: float,
    sigma: float,
    log: bool = False
) -> float:
    if len(symbols) == 0:
        return 0 if log else 1
    terms = [order_statistic_class_likelihood(s, mu, sigma, log=log) for s in symbols]
    return sum(terms) if log else np.product(terms)
