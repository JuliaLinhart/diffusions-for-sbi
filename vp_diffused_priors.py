import torch

from sbi.utils import BoxUniform
from scipy.stats import norm


def get_vpdiff_uniform_score(a, b, nse):
    # score of diffused prior: grad_t log prior_t (theta_t)
    #
    # prior_t = int p_{t|0}(theta_t|theta) p(theta)dtheta
    #         = uniform_cst * int_[a,b] p_{t|0}(theta_t|theta) dtheta
    # where p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
    #
    # ---> prior_t: uniform_cst * f_1(theta_t_1) * f_2(theta_t_2)
    # ---> grad log prior_t: (f_1_prime / f_1, f_2_prime / f_2)
    norm = torch.distributions.Normal(loc=torch.zeros((1,), device=a.device),
                                      scale=torch.ones((1,), device=a.device))
    norm.pdf = lambda x: torch.exp(norm.log_prob(x))
    def vpdiff_uniform_score(theta_t, t):
        # device
        device = theta_t.device


        # reshape theta_t
        thetas = {}
        for i in range(len(a)):
            thetas[i] = theta_t[:, i].unsqueeze(1)

        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
        # with _t = theta_0 * alpha_t
        alpha_t = nse.alpha(t)
        sigma_t = nse.sigma(t)

        # N(theta_t|mu_t, sigma^2_t) = N(mu_t|theta_t, sigma^2_t)
        # int N(theta_t|mu_t, sigma^2_t) dtheta = int N(mu_t|theta_t, sigma^2_t) dmu_t / alpha_t
        # theta in [a, b] -> mu_t in [a, b] * alpha_t

        prior_score_t = {}
        for i in range(len(a)):
            f = (
                norm.cdf((b[i] * alpha_t - thetas[i]) / sigma_t)
                - norm.cdf((a[i] * alpha_t - thetas[i]) / sigma_t)
            ) / alpha_t

            # derivative of norm_cdf w.r.t. theta_t
            f_prime = (
                -1
                / (sigma_t)
                * (
                    norm.pdf((b[i] * alpha_t - thetas[i]) / sigma_t)
                    - norm.pdf((a[i] * alpha_t - thetas[i]) / sigma_t)
                )
                / alpha_t
            )

            # score of diffused prior: grad_t log prior_t (theta_t)
            prior_score_t[i] = f_prime / (f + 1e-6)  # (batch_size, 1)

        prior_score_t = torch.cat(
            [ps for ps in prior_score_t.values()], dim=1
        )  # (batch_size, dim_theta)

        return prior_score_t

    return vpdiff_uniform_score
