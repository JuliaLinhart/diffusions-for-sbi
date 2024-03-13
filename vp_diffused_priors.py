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
        loc=torch.zeros((1,), device=a.device), scale=torch.ones((1,), device=a.device), validate_args=False
    )

    def vpdiff_uniform_score(theta, t):
        # transition kernel p_{t|0}(theta_t) = N(theta_t | mu_t, sigma^2_t)
        # with _t = theta_0 * scaling_t
        scaling_t = nse.alpha(t) ** 0.5
        sigma_t = nse.sigma(t)

        # N(theta_t|mu_t, sigma^2_t) = N(mu_t|theta_t, sigma^2_t)
        # int N(theta_t|mu_t, sigma^2_t) dtheta = int N(mu_t|theta_t, sigma^2_t) dmu_t / scaling_t
        # theta in [a, b] -> mu_t in [a, b] * scaling_t
        f = (norm.cdf((b * scaling_t - theta) / sigma_t) - norm.cdf((a * scaling_t - theta) / sigma_t)) / scaling_t
        f_prime = -1/sigma_t * (torch.exp(norm.log_prob((b * scaling_t - theta) / sigma_t)) - torch.exp(norm.log_prob((a * scaling_t - theta) / sigma_t)))/ scaling_t

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
        loc=scaling_t * mean
        covariance_matrix=sigma_t**2 * torch.eye(theta.shape[-1], device=mean.device) + scaling_t**2 * cov

        # grad_theta_t log N(theta_t | loc, cov) = - cov^{-1} * (theta_t - loc)
        prior_score_t = -(theta - loc) @ torch.linalg.inv(covariance_matrix)
        return prior_score_t

    return vpdiff_gaussian_score

def get_vpdiff_gamma_score(alpha, beta, nse):
    # score of diffused prior: grad_t log prior_t (theta_t)
    # for Gamma prior p(theta) = Gamma(theta | alpha, beta)

    def gamma_n(x, alpha, beta, mu, scale):
        gamma_pdf = torch.distributions.Gamma(alpha, beta, validate_args=False).log_prob(x).exp()
        normal_pdf = torch.distributions.Normal(mu, scale, validate_args=False).log_prob(x).exp()
        return gamma_pdf * normal_pdf


    def integrate_gamma_n(alpha, beta, mu, scale):
        from scipy.integrate import quad
        import numpy as np

        # print(alpha, beta, mu, scale)
        def integrand(x, mu_):
            return gamma_n(x, alpha, beta, mu_, scale)
        
        integral = []
        for mu_ in mu:
            # between 0 and infinity
            integral.append(quad(integrand, 0, np.inf, args=(mu_))[0])
        return torch.tensor(integral)


    def vp_diff_gamma_score(theta, t):
        # p_t(theta_t | alpha, beta) = int p_{t|0}(theta_t|theta) Gamma(theta | alpha, beta) dtheta
        # p_{t|0}(theta_t|theta)= N(theta_t | scaling_t * theta, sigma_t^2 I)

        # grad_theta_t log p_t(theta_t) = grad_theta_t p_t(theta_t) / p_t(theta_t)

        # grad_theta_t N(theta_t | scaling_t * theta, sigma_t^2) = - (theta_t - scaling_t * theta) / sigma_t^2
        # grad_theta_t p_t(theta_t | alpha, beta) = - (theta_t / sigma_t^2) * p_t(theta_t | alpha, beta)
        #     + scaling_t/sigma_t^2 * int theta * Gamma(theta | alpha, beta) N(theta_t | mu_t, sigma_t^2) dtheta

        # but theta * Gamma(theta | alpha, beta) = Gamma(theta | alpha+1, beta)
        # so grad_theta_t log p_t(theta_t) = - (theta_t / sigma_t^2) + scaling_t/sigma_t^2 * p_t(theta_t | alpha+1, beta) / p_t(theta_t | alpha, beta)

        scaling_t = nse.alpha(t) ** 0.5
        sigma_t = nse.sigma(t)

        first_term = - (theta / (sigma_t**2))
        # N(theta_t | scaling_t * theta, sigma_t^2 I) = N(theta | theta_t / scaling_t, sigma_t^2 I / scaling_t^2)
        
        int_nominator = integrate_gamma_n(alpha+1, beta, mu=theta/scaling_t, scale=sigma_t/scaling_t) * alpha / beta
        int_denominator = integrate_gamma_n(alpha, beta, mu=theta/scaling_t, scale=sigma_t/scaling_t)
        # print(int_nominator, int_denominator)
        second_term = (scaling_t/(sigma_t**2)) * (int_nominator) /(int_denominator)
        # print(first_term, second_term)
        prior_score_t = first_term + second_term

        return prior_score_t

    return vp_diff_gamma_score



if __name__ == '__main__':
    from nse import NSE
    from tasks.toy_examples.prior import UniformPrior
    nse = NSE(1,1)
    beta = 2 # small beta gives big variance (heavy tail)
    alpha = 0.5 # small alpha gives high skewness (heavy tail)
    prior = torch.distributions.Gamma(alpha, beta, validate_args=False)
    diffused_prior_score = get_vpdiff_gamma_score(prior.concentration, prior.rate, nse)

    t = torch.tensor(0.001)
    theta_t = prior.sample((10,))
    prior_score_t = diffused_prior_score(theta_t, t)
    print(prior_score_t)
    print((alpha-1)/theta_t - beta)
