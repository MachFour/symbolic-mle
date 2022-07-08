import numpy as np
from scipy import stats, integrate

from scipy.stats._continuous_distns import _norm_pdf, _norm_cdf, _norm_logpdf, _norm_logcdf

"""
Skew-Normal distribution with centred parametrisation, as defined in 
A. Azzalini, A.Capitano, "The Skew-Normal and Related Families (2014).

and

Reinaldo B.Arellano-Valle, Adelchi Azzalini,
"The centred parametrization for the multivariate skew-normal distribution"
https://www.sciencedirect.com/science/article/pii/S0047259X08000341 [open access]
https://doi.org/10.1016/j.jmva.2008.01.020

In terms of the standard/direct parametrisation (ξ, ω, α), the centred
parameters are
μ = ξ + ω*μ_z
σ^2 = ω^2 (1 − μ_z^2)
γ_1 = (4−π)/2 * μ_z^3 / sqrt((1 - μ_z^2)^3)

where
δ = α/sqrt(1 + α^2)
b = sqrt(2/π)
μ_z = bδ


The inverse mapping is
α = δ / (1 - δ^2)
ω^2 = σ^2 / (1 - μ_z^2)
ξ = μ - ω μ_z

where
c = (2 γ_1/(4-π))^(1/3)
μ_z = c/sqrt(1 + c^2)
b = sqrt(2/π)
δ = μ_z/b


"""


def centred_to_direct_parameters(mu, sigma, gamma_1) -> tuple[float, float, float]:
    """
    :param mu: mean parameter for Skew-normal distribution [centred parametrisation]
    :param sigma: std deviation parameter for Skew-normal distribution [centred parametrisation]
    :param gamma_1: skewness parameter for Skew-normal distribution [centred parametrisation]
    :return: triple representing parameters of same skew-normal distribution under direct parametrisation.
    """
    if abs(gamma_1) > 1.0:
        return np.nan, np.nan, np.nan

    b = np.sqrt(2/np.pi)
    c = np.cbrt(2*gamma_1 / (4 - np.pi))
    mu_z = c / np.sqrt(1 + c**2)
    delta = np.clip(mu_z/b, -0.999, 0.999)
    omega = abs(sigma) / np.sqrt(1 - mu_z**2)
    xi = mu - omega*mu_z
    alpha = delta / np.sqrt(1 - delta**2)

    return xi, omega, alpha


def direct_to_centred_parameters(xi, omega, alpha):
    """
    :param xi: location parameter for Skew-normal distribution [direct parametrisation]
    :param omega: scale parameter for Skew-normal distribution [direct parametrisation]
    :param alpha: shape parameter for Skew-normal distribution [direct parametrisation]
    :return: triple representing parameters of same skew-normal distribution under centered parametrisation.
    """
    delta = alpha / np.sqrt(1 + alpha**2)
    b = np.sqrt(2/np.pi)
    mu_z = b*delta
    mu = xi + omega*mu_z
    sigma = omega * np.sqrt(1 - mu_z**2)
    gamma_1 = (4 - np.pi)/2 * (mu_z**3 / (1 - mu_z**2)**(3/2))

    return mu, sigma, gamma_1


class skew_norm_centered_gen(stats.rv_continuous):
    """
    A skew-normal random variable with centred parametrisation.
    Implementation adapted from Scipy skew normal distribution

    """

    # gamma_1 is skewness parameter,
    # Max value of delta is 1, so if b = sqrt(2/pi), then
    # c = b / sqrt(1 - b^2)
    # g_max = c**3 * (4 - pi) / 2 = 0.9952717464311568

    def _argcheck(self, g_1):
        return abs(g_1) <= 0.9952717464311568

    @staticmethod
    def _make_pdf_args(x, g_1):
        if np.isscalar(g_1):
            mu, sigma = 0, 1
            centre_params = centred_to_direct_parameters
        else:
            mu = np.zeros(g_1.shape)
            sigma = np.ones(g_1.shape)
            centre_params = np.vectorize(centred_to_direct_parameters)

        xi, omega, a = centre_params(mu, sigma, g_1)
        z = (x - xi) / omega
        return z, omega, a

    def _pdf(self, x, g_1):
        z, omega, a = self._make_pdf_args(x, g_1)
        return 2/omega * _norm_pdf(z)*_norm_cdf(a*z)

    def _logpdf(self, x, g_1):
        z, omega, a = self._make_pdf_args(x, g_1)
        return np.log(2/omega) + _norm_logpdf(z) + _norm_logcdf(a*z)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        if x <= 0:
            cdf = integrate.quad(self._pdf, _a, x, args=args)[0]
        else:
            t1 = integrate.quad(self._pdf, _a, 0, args=args)[0]
            t2 = integrate.quad(self._pdf, 0, x, args=args)[0]
            cdf = t1 + t2
        if cdf > 1:
            # Presumably numerical noise, e.g. 1.0000000000000002
            cdf = 1.0
        return cdf

    def _fitstart(self, data):
        """Starting point for fit (shape arguments + loc + scale)."""
        skewness = np.clip(stats.skew(data, bias=False), 0.99, 0.99)
        loc, scale = self._fit_loc_scale_support(data, skewness)
        return skewness, loc, scale

    def _rvs(self, *args, size=None, random_state=None):
        g_1 = args[0] if len(args) > 1 else 0
        xi, omega, a = centred_to_direct_parameters(0, 1, g_1)
        return stats.skewnorm.rvs(a, loc=xi, scale=omega, size=size, random_state=random_state)

    def _stats(self, a, moments='mvsk'):
        output = [None, None, None, None]
        const = np.sqrt(2/np.pi) * a/np.sqrt(1 + a**2)

        if 'm' in moments:
            output[0] = const
        if 'v' in moments:
            output[1] = 1 - const**2
        if 's' in moments:
            output[2] = ((4 - np.pi)/2) * (const/np.sqrt(1 - const**2))**3
        if 'k' in moments:
            output[3] = (2*(np.pi - 3)) * (const**4/(1 - const**2)**2)

        return output


skewnorm_centered = skew_norm_centered_gen(name='skewnorm_centered')


def main():
    xi_0 = 0
    omega_0 = 1
    alpha_0 = 1

    mu_0, sigma_0, gamma_1_0 = direct_to_centred_parameters(xi_0, omega_0, alpha_0)
    test_dp = stats.skewnorm(alpha_0, loc=xi_0, scale=omega_0)

    samples = test_dp.rvs(10000)

    #fig, ax = plt.subplots(1, 1)
    #ax.hist(samples)
    #fig.show()

    test_dp_fit = stats.skewnorm.fit(samples)
    test_cp_fit = skewnorm_centered.fit(samples)
    fit_alpha, fit_xi, fit_omega = test_dp_fit[:]
    fit_gamma1, fit_mu, fit_sigma = test_cp_fit[:]

    exp_mu, exp_sigma, exp_gamma_1 = direct_to_centred_parameters(fit_xi, fit_omega, fit_alpha)
    back_xi, back_omega, back_alpha = centred_to_direct_parameters(fit_mu, fit_sigma, fit_gamma1)

    print(f"True DP parameters: ξ={xi_0}, ω={omega_0}, α={alpha_0}")

    print(f"True CP parameters:",
          f"μ={mu_0:.3f},",
          f"σ={sigma_0:.3f},",
          f"γ_1={gamma_1_0}")

    print(f"Fitted DP parameters:",
          f"ξ={fit_xi:.3f},"
          f"ω={fit_omega:.3f},"
          f"α={fit_alpha:.3f}")

    print(f"Expected CP parameters from DP fit:",
          f"μ={exp_mu:.3f},"
          f"σ={exp_sigma:.3f},"
          f"γ_1={exp_gamma_1:.3f}")

    print(f"Fitted CP parameters:",
          f"μ={fit_mu:.3f},"
          f"σ={fit_sigma:.3f},"
          f"γ_1={fit_gamma1:.3f}")

    print(f"Backconverted fitted CP->DP parameters:",
          f"ξ={back_xi:.3f},"
          f"ω={back_omega:.3f},"
          f"α={back_alpha:.3f}")


if __name__ == "__main__":
    main()


