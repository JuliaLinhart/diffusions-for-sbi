import torch
import math


def get_vpdiff_uniform_score(a, b, nse):
    # score of diffused prior: grad_t log prior_t (theta_t)
    #
    # prior_t = int p_{t|0}(theta_t|theta) p(theta)dtheta
    #         = uniform_cst * int_[a,b] p_{t|0}(theta_t|theta) dtheta
    # where p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
    #
    # ---> prior_t: uniform_cst * f_1(theta_t_1) * f_2(theta_t_2)
    # ---> grad log prior_t: (f_1_prime / f_1, f_2_prime / f_2)
    norm = torch.distributions.Normal(
        loc=torch.zeros((1,), device=a.device),
        scale=torch.ones((1,), device=a.device),
        validate_args=False,
    )

    def vpdiff_uniform_score(theta, t):
        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
        # with _t = theta_0 * scaling_t
        scaling_t = nse.alpha(t) ** 0.5
        sigma_t = nse.sigma(t)

        # N(theta_t|mu_t, sigma^2_t) = N(mu_t|theta_t, sigma^2_t)
        # int N(theta_t|mu_t, sigma^2_t) dtheta = int N(mu_t|theta_t, sigma^2_t) dmu_t / scaling_t
        # theta in [a, b] -> mu_t in [a, b] * scaling_t
        f = (
            norm.cdf((b * scaling_t - theta) / sigma_t)
            - norm.cdf((a * scaling_t - theta) / sigma_t)
        ) / scaling_t
        f_prime = (
            -1
            / sigma_t
            * (
                torch.exp(norm.log_prob((b * scaling_t - theta) / sigma_t))
                - torch.exp(norm.log_prob((a * scaling_t - theta) / sigma_t))
            )
            / scaling_t
        )

        # score of diffused prior: grad_t log prior_t (theta_t)
        prior_score_t = f_prime / (f + 1e-6)

        return prior_score_t

    return vpdiff_uniform_score


def get_vpdiff_gaussian_score(mean, cov, nse):
    # score of diffused prior: grad_t log prior_t (theta_t)
    # for Gaussian prior p(theta) = N(theta | mean, cov)

    def vpdiff_gaussian_score(theta, t):
        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t I)
        # with mu_t = theta * scaling_t
        scaling_t = nse.alpha(t) ** 0.5
        sigma_t = nse.sigma(t)

        # from Bishop 2006 (2.115)
        # p_t(theta_t) = int p_{t|0}(theta_t|theta) p(theta)dtheta
        # = N(theta_t | scaling_t * mean, sigma^2_t I + scaling_t^2 * cov)
        loc = scaling_t * mean
        covariance_matrix = (
            sigma_t**2 * torch.eye(theta.shape[-1], device=mean.device)
            + scaling_t**2 * cov
        )

        # grad_theta_t log N(theta_t | loc, cov) = - cov^{-1} * (theta_t - loc)
        prior_score_t = -(theta - loc) @ torch.linalg.inv(covariance_matrix)
        return prior_score_t

    return vpdiff_gaussian_score


def get_vpdiff_gamma_score(alpha, beta, nse):
    # score of diffused prior: grad_t log prior_t (theta_t)
    # for Gamma prior p(theta) = Gamma(theta | alpha, beta)
    # with theta 1D and alpha an integer

    def vp_diff_gamma_score(theta, t):
        # p_t(theta_t) = int p(theta)p_{t|0}(theta_t|theta)dtheta
        # = int Gamma(theta | alpha, beta) N(theta_t | theta * scaling_t, sigma^2_t) dtheta
        # = int theta^(alpha-1) * N(theta | mu_t, S2_t) dtheta * exp log_f_t(theta_t) * C
        # = M(alpha-1) * f_t(theta_t) * C where M(m) = E[theta^m] is the m-th moment of N(mu_t, S2_t) but only integral over R+

        # grad_t log p_t(theta_t) = grad_t log M(alpha-1)  + grad_t log_f_t(theta_t)
        alpha_t = nse.alpha(t)
        scaling_t = alpha_t**0.5
        sigma_t = nse.sigma(t)
        upsilon_t = sigma_t**2

        S2_t = upsilon_t / alpha_t

        # print("S2_t: ", S2_t, ", sqrt(alpha_t): ", scaling_t)

        mu_t = (
            theta / scaling_t - beta * S2_t
        )  # (scaling_t * theta / upsilon_t - beta) * S2_t
        # print("mu_t: ", mu_t)
        grad_mu_t = 1 / scaling_t  # scaling_t / upsilon_t * S2_t
        # print("grad_mu_t: ", grad_mu_t)

        m = alpha - 1
        M_t = half_gaussian_moments(mu_t, torch.sqrt(S2_t), m)
        # print("M_t: ", M_t)
        grad_M_t = grad_half_gaussian_moments(grad_mu_t, mu_t, torch.sqrt(S2_t), m)
        # print("grad_M_t: ", grad_M_t)

        if (M_t == 0).any():
            print("M_t is zero")
            grad_log_M_t = torch.zeros_like(grad_M_t)
        else:
            grad_log_M_t = grad_M_t / M_t
        # print("grad_log_M_t: ", grad_log_M_t)

        grad_log_f_t = -beta / scaling_t
        # print("grad_log_f_t: ", grad_log_f_t)

        prior_score_t = grad_log_M_t + grad_log_f_t

        return prior_score_t

    return vp_diff_gamma_score


def half_gaussian_moments(mu, sigma, m):
    # m must be integer
    assert m % 1 == 0
    if m == 0:
        # print("x=mu/sqrt(2*sigma^2): ", mu / torch.sqrt(2 * sigma**2))
        # print("erf: ", torch.special.erf(mu / torch.sqrt(2 * sigma**2)))
        return (
            (torch.special.erf(mu / torch.sqrt(2 * sigma**2)) + 1)
            * sigma
            * math.sqrt(math.pi / 2)
        )
    elif m == 1:
        return sigma**2 * torch.exp(
            -(mu**2) / (2 * sigma**2)
        ) + mu * half_gaussian_moments(mu, sigma, 0)
    else:
        assert m > 1
        return mu * half_gaussian_moments(mu, sigma, m - 1) + sigma**2 * (
            m - 1
        ) * half_gaussian_moments(mu, sigma, m - 2)

def grad_half_gaussian_moments(grad_mu, mu, sigma, m):
    # m must be integer
    assert m % 1 == 0
    if m == 0:
        return grad_mu * torch.exp(-(mu**2) / (2 * sigma**2)) 
    elif m == 1:
        return grad_mu * half_gaussian_moments(mu, sigma, 0) 
    else:
        return (
            grad_mu * half_gaussian_moments(mu, sigma, m - 1)
            + mu * grad_half_gaussian_moments(grad_mu, mu, sigma, m - 1)
            + (sigma**2) * (m - 1) * grad_half_gaussian_moments(grad_mu, mu, sigma, m - 2)
        )


if __name__ == "__main__":
    from nse import NSE

    nse = NSE(1, 1)
    beta = 2  # small beta gives big variance (heavy tail)
    alpha = 1  # small alpha gives high skewness (heavy tail)
    prior = torch.distributions.Gamma(alpha, beta, validate_args=False)
    diffused_prior_score = get_vpdiff_gamma_score(prior.concentration, prior.rate, nse)

    # t = torch.tensor(0.0)
    # theta_t = prior.sample((5,))
    # t = torch.tensor(0.1)
    theta_t = torch.randn((5,))
    t_ = torch.linspace(1, 0, 10)
    for t in t_:
        prior_score_t = diffused_prior_score(theta_t, t)
        print(prior_score_t)
        print((alpha - 1) / theta_t - beta)
        print()
