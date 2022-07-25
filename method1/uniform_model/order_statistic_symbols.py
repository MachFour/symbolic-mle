from typing import Callable

import numpy as np
from scipy.special import comb
from scipy.stats import uniform


def order_statistic_class_likelihood(
    lower_statistic_value: float,
    lower_statistic_ordinal: int,
    upper_statistic_value: float,
    upper_statistic_ordinal: int,
    class_size: int,
    a: float,
    b: float,
    log: bool = False
) -> float:
    """Single-class symbolic likelihood evaluation function for a Uniform(a, b) model and single
    order-statistic symbol; implements 4.3.14 from thesis.

    To evaluate the likelihood for multiple symbols/classes, the likelihood values obtained from each class
    need to be multiplied, or added together in the case of log-likelihood

    :param lower_statistic_value: Observed summary value for lower order statistic
    :param lower_statistic_ordinal: Position of lower order statistic, between 1 and n
            where n is the size of the class. Must be less than upper statistic ordinal.
    :param upper_statistic_value: Observed summary value for upper order statistic
    :param upper_statistic_ordinal: Position of upper order statistic, between 1 and n
            where n is the size of the class. Must be greater than lower statistic ordinal.
    :param class_size: Number of points summarised by the symbol
    :param a: minimum parameter for Uniform model; likelihood is evaluated at this point
    :param b: maximum parameter for Uniform model; likelihood is evaluated at this point
    :param log: Whether to make a log-likelihood function rather than ordinary one. This
        should be combined with other classes' log-likelihoods via summation rather than
        multiplication
    :return: (Log)-likelihood of the given symbol data for given μ, σ parameters.
    """
    # L_k(θ, s, n) = G(s_l; θ)^(l-1) * [G(s_u; θ) - G(s_l; θ)]^(u - l - 1) * (1 - G(s_u; θ))^(n-u) g(s_u); θ) g(s_l); θ)
    # where
    # θ is the parameter (vector) to be chosen,
    # G(.; θ) is the model CDF,
    # g(.; θ) is the model PDF,
    # l and u are respectively the lower and upper ordinals,
    # s_l and s_u are the observed order statistics of order l and u respectively.

    # The constant of proportionality is given by
    # C = n! / [ (l - 1)! * (u - l - 1)! (n - u)! = nCu * uCl * l * (u-l)

    n = class_size
    l, u = lower_statistic_ordinal, upper_statistic_ordinal
    s_l, s_u = lower_statistic_value, upper_statistic_value

    if not (1 <= l < u <= n):
        raise ValueError("Must have 1 <= lower statistic order < upper statistic order <= class size")
    elif not s_l <= s_u:
        raise ValueError("Must have observed lower statistic <= observed upper statistic")
    elif not a <= b:
        raise ValueError("Must have a <= b")

    C = comb(n, u, exact=True) * comb(u, l, exact=True) * l * (u - l)

    if not log:
        # L_k(θ, s, n) =
        lower_pdf, upper_pdf = uniform.pdf([s_l, s_u], loc=a, scale=b-a).flat
        lower_cdf, upper_cdf = uniform.cdf([s_l, s_u], loc=a, scale=b-a).flat
        # G(s_l; θ)^(l-1)
        l_term = lower_cdf**(l-1) if l - 1 > 0 else 1
        # * (1 - G(s_u; θ))^(n-u)
        u_term = uniform.sf(s_u, loc=a, scale=b-a)**(n-u) if n - u > 0 else 1
        # * [G(s_u; θ) - G(s_l; θ)]^(u - l - 1)
        difference_term = (upper_cdf - lower_cdf) ** (u - l - 1) if (u - l - 1) > 0 else 1

        # G(s_l; θ)^(l-1) * [G(s_u; θ) - G(s_l; θ)]^(u - l - 1) * (1 - G(s_u; θ))^(n-u) g(s_u); θ) g(s_l); θ)
        return l_term * difference_term * u_term * lower_pdf * upper_pdf * C
    else:
        # l_k(θ, s, n) =
        lower_log_pdf, upper_log_pdf = uniform.logpdf([s_l, s_u], loc=a, scale=b-a).flat
        lower_log_cdf, upper_log_cdf = uniform.logcdf([s_l, s_u], loc=a, scale=b-a).flat
        lower_cdf, upper_cdf = uniform.cdf([s_l, s_u], loc=a, scale=b-a).flat
        # (l-1) * log G(s_l; θ)
        l_term = lower_log_cdf * (l - 1) if l - 1 > 0 else 0
        # + (n-u) * log (1 - G(s_u; θ))
        u_term = uniform.logsf(s_u, loc=a, scale=b-a) * (n - u) if n - u > 0 else 0
        # + (u - l - 1) * log [G(s_u; θ) - G(s_l; θ)]
        difference_term = np.log(upper_cdf - lower_cdf) * (u - l - 1) if (u - l - 1) > 0 else 1

        return l_term + difference_term + u_term + lower_log_pdf + upper_log_pdf + np.log(C)


def make_order_statistic_class_likelihood(
        lower_statistic_value: float,
        lower_statistic_ordinal: int,
        upper_statistic_value: float,
        upper_statistic_ordinal: int,
        class_size: int,
        log: bool = False
) -> Callable[[float, float], float]:
    """
    Implements 4.3.12 from honours thesis: single-class symbolic likelihood for 1D univariate normal
    when class summaries are order statistics (min and max are special case)

    This function can be used to create a 1D symbolic likelihood function per class,
    which can then be used to evaluate the symbolic likelihood for desired model parameters.

    :param lower_statistic_value: Observed summary value for lower order statistic
    :param lower_statistic_ordinal: Position of lower order statistic, between 1 and n
            where n is the size of the class. Must be less than upper statistic ordinal.
    :param upper_statistic_value: Observed summary value for upper order statistic
    :param upper_statistic_ordinal: Position of upper order statistic, between 1 and n
            where n is the size of the class. Must be greater than lower statistic ordinal.
    :param class_size: Number of points summarised by the symbol
    :param log: Whether to make a log-likelihood function rather than ordinary one. This
        should be combined with other classes' log-likelihoods via summation rather than
        multiplication
    :return: Function returning symbolic likelihood function for a Gaussian(μ, σ) distribution.
        This can be maximised or evaluated at model parameters of interest. Note σ is not squared.
    """

    def likelihood(a: float, b: float) -> float:
        return order_statistic_class_likelihood(
            lower_statistic_value,
            lower_statistic_ordinal,
            upper_statistic_value,
            upper_statistic_ordinal,
            class_size,
            a,
            b,
            log
        )

    return likelihood


