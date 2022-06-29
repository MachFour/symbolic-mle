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
σ^2 = ω^2 * sqrt(1 − μ_z^2)
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
δ = sqrt(π/2) * μ_z


"""


def centred_to_direct_parameters(mu, sigma_sq, gamma_1):
    """
    :param mu: mean parameter for Skew-normal distribution [centred parametrisation]
    :param sigma_sq: variance parameter for Skew-normal distribution [centred parametrisation]
    :param gamma_1: skewness parameter for Skew-normal distribution [centred parametrisation]
    :return: triple representing parameters of same skew-normal distribution under direct parametrisation.
    """
    c = (2*gamma_1 / (4 - np.pi))**(1/3)
    mu_z = c / np.sqrt(1 + c**2)
    omega_sq = sigma_sq / (1 - mu_z**2)
    xi = mu - np.sqrt(omega_sq)*mu_z
    delta = np.sqrt(np.pi/2) * mu_z
    delta = np.clip(delta, 0.999, 0.999)
    alpha = delta / np.sqrt(1 - delta**2)

    return xi, omega_sq, alpha


def direct_to_centred_parameters(xi, omega_sq, alpha):
    """
    :param xi: location parameter for Skew-normal distribution [direct parametrisation]
    :param omega_sq: scale parameter for Skew-normal distribution [direct parametrisation]
    :param alpha: shape parameter for Skew-normal distribution [direct parametrisation]
    :return: triple representing parameters of same skew-normal distribution under centered parametrisation.
    """
    delta = alpha / np.sqrt(1 + alpha**2)
    b = np.sqrt(2/np.pi)
    mu_z = b*delta
    mu = xi + np.sqrt(omega_sq)*mu_z
    sigma_sq = omega_sq * (1 - mu_z**2)
    gamma_1 = (4 - np.pi)/2 * (mu_z**3 / (1 - mu_z**2)**(3/2))

    return mu, sigma_sq, gamma_1


class skew_norm_centered_gen(stats.rv_continuous):
    """
    A skew-normal random variable with centred parametrisation.
    Implementation adapted from Scipy skew normal distribution

    """

    # gamma_1 is skewness parameter
    def _argcheck(self, g_1):
        return np.isfinite(g_1)

    def _pdf(self, x, g_1):
        xi, omega_sq, a = centred_to_direct_parameters(0, 1, g_1)
        omega = np.sqrt(omega_sq)
        z = (x - xi) / omega
        return 2/omega * _norm_pdf(z)*_norm_cdf(a*z)

    def _logpdf(self, x, g_1):
        xi, omega_sq, a = centred_to_direct_parameters(0, 1, g_1)
        omega = np.sqrt(omega_sq)
        z = (x - xi) / omega
        return np.log(2) - np.log(omega) + _norm_logpdf(z) + _norm_logcdf(a*z)

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

    def _sf(self, x, g_1):
        xi, omega_sq, a = centred_to_direct_parameters(0, 1, g_1)
        return stats.skewnorm._cdf(-x, -a)

    def _rvs(self, g_1, size=None, random_state=None):
        u0 = random_state.normal(size=size)
        v = random_state.normal(size=size)

        xi, omega_sq, a = centred_to_direct_parameters(0, 1, g_1)
        d = a/np.sqrt(1 + a**2)
        u1 = d*u0 + v*np.sqrt(1 - d**2)
        return xi + np.sqrt(omega_sq)*np.where(u0 >= 0, u1, -u1)

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

    mu_0, sigma_sq_0, gamma_1_0 = direct_to_centred_parameters(xi_0, omega_0**2, alpha_0)
    test_dp = stats.skewnorm(alpha_0, loc=xi_0, scale=omega_0)

    samples = test_dp.rvs(10000)

    #fig, ax = plt.subplots(1, 1)
    #ax.hist(samples)
    #fig.show()

    test_dp_fit = stats.skewnorm.fit(samples)
    test_cp_fit = skewnorm_centered.fit(samples)
    fit_alpha, fit_xi, fit_omega = test_dp_fit[:]
    fit_gamma1, fit_mu, fit_sigma = test_cp_fit[:]

    exp_mu, exp_sigma_sq, exp_gamma_1 = direct_to_centred_parameters(fit_xi, fit_omega**2, fit_alpha)
    back_xi, back_omega_sq, back_alpha = centred_to_direct_parameters(fit_mu, fit_sigma**2, fit_gamma1)

    print(f"True DP parameters: xi={xi_0}, omega={omega_0}, alpha={alpha_0}")

    print(f"True CP parameters:",
          f"mu={mu_0:.3f},",
          f"sigma={np.sqrt(sigma_sq_0):.3f},",
          f"gamma_1={gamma_1_0}")

    print(f"Fitted DP parameters:",
          f"xi={fit_xi:.3f},"
          f"omega={fit_omega:.3f},"
          f"alpha={fit_alpha:.3f}")

    print(f"Expected CP parameters from DP fit:",
          f"mu={exp_mu:.3f},"
          f"sigma={np.sqrt(exp_sigma_sq):.3f},"
          f"gamma_1={exp_gamma_1:.3f}")

    print(f"Fitted CP parameters:",
          f"mu={fit_mu:.3f},"
          f"sigma={fit_sigma:.3f},"
          f"gamma_1={fit_gamma1:.3f}")

    print(f"Backconverted fitted CP->DP parameters:",
          f"xi={back_xi:.3f},"
          f"omega={np.sqrt(back_omega_sq):.3f},"
          f"alpha={back_alpha:.3f}")


if __name__ == "__main__":
    main()


