import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from tqdm import tqdm
from functools import partial
from nse import assure_positive_definitness


def log_L(
        means_posterior_backward,
        covar_posteriors_backward,
        mean_prior_backward,
        covar_prior_backward,
):
    """
    Calculates all the factors dependent of theta of log L_theta as defined in (slack document for now...)
    Following http://www.lucamartino.altervista.org/2003-003.pdf
    Parameters
    ----------
    means_posterior_backward: torch.Tensor
        (*, n_observations, dim_theta)
    covar_posteriors_backward: torch.Tensor
        (*, n_observations, dim_theta, dim_theta)
    mean_prior_backward: torch.Tensor
        (*, dim_theta)
    covar_prior_backward: torch.Tensor
        (*, dim_theta, dim_theta)

    Returns
    -------

    """

    def from_canonical_to_sufficient(mean, covar):
        eta = torch.linalg.solve(A=covar, B=mean[..., None])[..., 0]
        return (
            torch.linalg.inv(covar),
            eta,
            -0.5 * (eta[..., None].mT @ covar @ eta[..., None])[..., 0, 0],
        )

    n_observations = means_posterior_backward.shape[-2]
    lambdas_posterior, etas_posterior, zetas_posterior = from_canonical_to_sufficient(
        means_posterior_backward, covar_posteriors_backward
    )
    lda_prior, eta_prior, zeta_prior = from_canonical_to_sufficient(
        mean_prior_backward, covar_prior_backward
    )

    sum_zetas = zetas_posterior.sum(axis=1) + (1 - n_observations) * zeta_prior

    final_gaussian_etas = (1 - n_observations) * eta_prior + etas_posterior.sum(axis=1)
    final_gaussian_ldas = (1 - n_observations) * lda_prior + lambdas_posterior.sum(
        axis=1
    )

    inv_lda_eta = torch.linalg.solve(B=final_gaussian_etas[..., None], A=final_gaussian_ldas)
    final_gaussian_zeta = -0.5 * (
            0#-torch.linalg.slogdet(final_gaussian_ldas).logabsdet
            + (
                    final_gaussian_etas[..., None].mT
                    @ inv_lda_eta
            )[..., 0, 0]
    )
    return sum_zetas - final_gaussian_zeta, final_gaussian_etas, final_gaussian_ldas


def mean_backward(theta, t, score_fn, nse, **kwargs):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    return (
            1
            / (alpha_t**0.5)
            * (theta + sigma_t**2 * score_fn(theta=theta, t=t, **kwargs))
    )


def sigma_backward(t, dist_cov, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    return torch.linalg.inv(torch.linalg.inv(dist_cov) + (alpha_t / sigma_t ** 2) * torch.eye(dist_cov.shape[-1]).to(dist_cov.device))
    # return (
    #     ((sigma_t**2)
    #     / alpha_t)
    #     * (
    #         torch.eye(2).to(alpha_t.device)
    #         + (sigma_t**2)
    #         * (-1)
    #         * torch.linalg.inv(
    #             (
    #                 nse.alpha(t) * dist_cov.to(alpha_t.device)
    #                 + (sigma_t**2)* torch.eye(2).to(alpha_t.device)
    #             )
    #         )
    #     )
    # )


def tweedies_approximation(x, theta, t, score_fn, nse, dist_cov_est=None, mode='JAC', clip_mean_bounds = (None, None)):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    if mode == 'JAC':
        def score_jac(theta, x):
            score = score_fn(theta=theta, t=t, x=x)
            return score, score
        jac_score, score = vmap(lambda theta: vmap(jacrev(score_jac, has_aux=True), in_dims=(None, 0))(theta, x))(theta)
        cov = (sigma_t**2 / alpha_t) * (torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2)*jac_score)
    elif mode == 'GAUSS':
        cov = torch.linalg.inv(torch.linalg.inv(dist_cov_est) + (alpha_t / sigma_t ** 2) * torch.eye(theta.shape[-1]).to(dist_cov_est.device))#(1 - alpha_t) * dist_cov_est
        cov = cov[None].repeat(theta.shape[0], 1, 1, 1)
        score = vmap(lambda theta: vmap(partial(score_fn, t=t),
                                        in_dims=(None, 0),
                                        randomness='different')(theta, x),
                     randomness='different')(theta)
    elif mode == 'PSEUDO':
        cov = (1 - alpha_t) * dist_cov_est
        cov = cov[None].repeat(theta.shape[0], 1, 1, 1)
        score = vmap(lambda theta: vmap(partial(score_fn, t=t),
                                        in_dims=(None, 0),
                                        randomness='different')(theta, x),
                     randomness='different')(theta)
    else:
        raise NotImplemented("Available methods are GAUSS, PSEUDO, JAC")
    mean = (1 / (alpha_t**0.5) * (theta[:, None] + sigma_t**2 * score))
    if clip_mean_bounds[0]:
        mean = mean.clip(*clip_mean_bounds)
    return cov, mean, score

def tweedies_approximation_prior(theta, t, score_fn, nse, mode='vmap'):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    if mode == 'vmap':
        def score_jac(theta):
            score = score_fn(theta=theta, t=t)
            return score, score
        jac_score, score = vmap(jacrev(score_jac, has_aux=True))(theta)
    else:
        raise NotImplemented
    mean = 1 / (alpha_t**0.5) * (theta + sigma_t**2 * score)
    cov = (sigma_t**2 / alpha_t) * (torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2)*jac_score)
    return cov, mean, score


def diffused_tall_posterior_score(
        theta,
        t,
        prior,
        prior_score_fn,
        x_obs,
        nse,
        dist_cov_est=None,
        cov_mode='JAC',
        psd_clipping=True,
        scale_gradlogL=False,
        warmup_alpha=.5,
        debug=False,
        score_fn=None,
        prior_type="gaussian",
):
    # device
    n_obs = x_obs.shape[0]

    theta.requires_grad = True
    theta.grad = None

    sigma_0_t, mean_0_t, scores = tweedies_approximation(x=x_obs,
                                                         theta=theta,
                                                         nse=nse,
                                                         t=t,
                                                         score_fn=nse.score if score_fn is None else score_fn,
                                                         dist_cov_est=dist_cov_est,
                                                         mode=cov_mode)
    # if mean_clipping_range[0]:
    #     mean_0_t = mean_0_t.clip(*mean_clipping_range)

    #scores = nse.score(theta[:, None], x_obs[None], t=t)
    if prior_type == "gaussian":
        mean_prior_0_t = mean_backward(theta, t, prior_score_fn, nse)
        sigma_prior_0_t = sigma_backward(t, prior.covariance_matrix, nse).repeat(
            theta.shape[0], 1, 1
        )
    else:
        sigma_prior_0_t, mean_prior_0_t, _ = tweedies_approximation_prior(theta=theta,
                                                                        t=t,
                                                                        score_fn=prior_score_fn,
                                                                        nse=nse)
    
    prior_score = prior_score_fn(theta, t)
    total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)

    if (nse.alpha(t)**.5 > warmup_alpha) and (n_obs > 1):
        # sigma_prior_0_t = #= assure_positive_definitness(sigma_prior_0_t.detach(), lim_inf=psd_clipping_range[0],
        #                                               lim_sup=psd_clipping_range[1])
        if psd_clipping:
            sigma_prior_eigvals = torch.linalg.eigvals(sigma_prior_0_t).real
            lim_sup_sigma = (n_obs / (n_obs - 1)) * sigma_prior_eigvals.min()
            if prior_type != "gaussian":
                lim_sup_sigma = lim_sup_sigma.item()
            sigma_0_t = assure_positive_definitness(sigma_0_t.detach(), lim_sup=lim_sup_sigma*.99)

        logL, _, lda = log_L(
            mean_0_t,
            sigma_0_t.detach(),
            mean_prior_0_t,
            sigma_prior_0_t.detach()
        )
        logL.sum().backward()
        gradlogL = theta.grad
        #is_positive = (torch.linalg.eigvals(lda).real.min(dim=-1).values > 0)
        if scale_gradlogL:
            gradlogL = gradlogL*(nse.alpha(t))
        total_score = total_score + gradlogL #* ((is_positive[..., None]).float())
        #total_score = total_score + gradlogL
    else:
        gradlogL = torch.zeros_like(total_score)
        lda = torch.zeros_like(sigma_0_t[:, 0])

    if debug:
        return (
            total_score.detach(),
            gradlogL.detach().cpu(),
            lda.detach().cpu(),
            scores.detach().cpu(),
            mean_0_t.detach().cpu(),
            sigma_0_t.detach().cpu(),
        )
    else:
        return total_score


def euler_sde_sampler(score_fn, nsamples, dim_theta, beta, device="cpu", debug=False, theta_clipping_range=(None, None)):
    theta_t = torch.randn((nsamples, dim_theta)).to(device)  # (nsamples, 2)
    time_pts = torch.linspace(1, 0, 1000).to(device)  # (ntime_pts,)
    theta_list = [theta_t]
    gradlogL_list = []
    lda_list = []
    posterior_scores_list = []
    means_posterior_backward_list = []
    sigma_posterior_backward_list = []
    for i in tqdm(range(len(time_pts) - 1)):
        t = time_pts[i]
        dt = time_pts[i + 1] - t

        # calculate the drift and diffusion terms
        f = -0.5 * beta(t) * theta_t
        g = beta(t) ** 0.5
        if debug:
            (
                score,
                gradlogL,
                lda,
                posterior_scores,
                means_posterior_backward,
                sigma_posterior_backward,
            ) = score_fn(theta_t, t, debug=True)
        else:
            score = score_fn(theta_t, t)
        score = score.detach()

        drift = f - g * g * score
        diffusion = g

        # euler-maruyama step
        theta_t = (
                theta_t.detach()
                + drift * dt
                + diffusion * torch.randn_like(theta_t) * torch.abs(dt) ** 0.5
        )
        if theta_clipping_range[0] is not None:
            theta_t = theta_t.clip(*theta_clipping_range)
        theta_list.append(theta_t.detach().cpu())
        if debug:
            gradlogL_list.append(gradlogL)
            lda_list.append(lda)
            posterior_scores_list.append(posterior_scores)
            means_posterior_backward_list.append(means_posterior_backward)
            sigma_posterior_backward_list.append(sigma_posterior_backward)

    theta_list[0] = theta_list[0].detach().cpu()
    if debug:
        return (
            theta_t,
            torch.stack(theta_list),
            torch.stack(gradlogL_list),
            torch.stack(lda_list),
            torch.stack(posterior_scores_list),
            torch.stack(means_posterior_backward_list),
            torch.stack(sigma_posterior_backward_list),
        )
    else:
        return theta_t, theta_list
    
