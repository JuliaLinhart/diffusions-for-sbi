import torch


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
        loc=torch.zeros((1,), device=a.device), scale=torch.ones((1,), device=a.device)
    )
    norm.pdf = lambda x: torch.exp(norm.log_prob(x))

    def vpdiff_uniform_score(theta_t, t):
        # reshape theta_t
        thetas = {}
        for i in range(len(a)):
            thetas[i] = theta_t[:, i].unsqueeze(1)

        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
        # with _t = theta_0 * scaling_t
        scaling_t = nse.alpha(t) ** 0.5
        sigma_t = nse.sigma(t)

        # N(theta_t|mu_t, sigma^2_t) = N(mu_t|theta_t, sigma^2_t)
        # int N(theta_t|mu_t, sigma^2_t) dtheta = int N(mu_t|theta_t, sigma^2_t) dmu_t / scaling_t
        # theta in [a, b] -> mu_t in [a, b] * scaling_t

        prior_score_t = {}
        for i in range(len(a)):
            f = (
                norm.cdf((b[i] * scaling_t - thetas[i]) / sigma_t)
                - norm.cdf((a[i] * scaling_t - thetas[i]) / sigma_t)
            ) / scaling_t

            # derivative of norm_cdf w.r.t. theta_t
            f_prime = (
                -1
                / (sigma_t)
                * (
                    norm.pdf((b[i] * scaling_t - thetas[i]) / sigma_t)
                    - norm.pdf((a[i] * scaling_t - thetas[i]) / sigma_t)
                )
                / scaling_t
            )

            # score of diffused prior: grad_t log prior_t (theta_t)
            prior_score_t[i] = f_prime / (f + 1e-6)  # (batch_size, 1)

        prior_score_t = torch.cat(
            [ps for ps in prior_score_t.values()], dim=1
        )  # (batch_size, dim_theta)

        return prior_score_t

    return vpdiff_uniform_score


def get_vpdiff_gaussian_score(mean, cov, nse):
    # score of diffused prior: grad_t log prior_t (theta_t)
    # for Gaussian prior p(theta) = N(theta | mean, cov)

    def vpdiff_gaussian_score(theta_t, t):

        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t I)
        # with mu_t = theta * scaling_t
        scaling_t = nse.alpha(t) ** 0.5
        sigma_t = nse.sigma(t)

        # from Bishop 2006 (2.115)
        # p_t(theta_t) = int p_{t|0}(theta_t|theta) p(theta)dtheta
        # = N(theta_t | scaling_t * mean, sigma^2_t I + scaling_t^2 * cov)
        loc=scaling_t * mean
        covariance_matrix=sigma_t**2 * torch.eye(2, device=mean.device) + scaling_t**2 * cov

        # grad_theta_t log N(theta_t | loc, cov) = - cov^{-1} * (theta_t - loc)
        prior_score_t = -(theta_t - loc) @ torch.linalg.inv(covariance_matrix)
        return prior_score_t

    return vpdiff_gaussian_score

if __name__ == '__main__':
    from nse import NSE
    from tasks.toy_examples.prior import GaussianPrior
    theta_t = torch.randn((1000, 2))
    t = torch.tensor([0.1])
    nse = NSE(2,2)
    prior = GaussianPrior()
    diffused_prior_score = get_vpdiff_gaussian_score(prior.prior.loc, prior.prior.covariance_matrix, nse)

    prior_score_t = diffused_prior_score(theta_t, t)
    print(prior_score_t.shape)