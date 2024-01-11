import numpy as np
import matplotlib.pyplot as plt
import torch
from tasks.toy_examples.data_generators import SBIGaussian2d
import sys
#from vp_diffused_priors import get_vpdiff_gaussian_score
from nse import NSE


torch.manual_seed(1)
sys.path.append("../")

N_TRAIN = 10_000
N_SAMPLES = 4096


def log_L(means_posterior_backward,
          covar_posteriors_backward,
          mean_prior_backward,
          covar_prior_backward):
    '''
    Calculates all the factors dependent of theta of log L_theta as defined in (slack document for now...) Following http://www.lucamartino.altervista.org/2003-003.pdf
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

    '''

    def from_canonical_to_sufficient(mean, covar):
        lda = torch.linalg.inv(covar)
        eta = (lda @ mean[..., None])[..., 0]
        return lda, eta, -.5 * (
                    -torch.linalg.slogdet(lda).logabsdet + (mean[..., None].mT @ lda @ mean[..., None])[..., 0, 0])

    n_observations = means_posterior_backward.shape[-2]
    lambdas_posterior, etas_posterior, zetas_posterior = from_canonical_to_sufficient(means_posterior_backward,
                                                                                      covar_posteriors_backward)
    lda_prior, eta_prior, zeta_prior = from_canonical_to_sufficient(mean_prior_backward, covar_prior_backward)

    sum_zetas = zetas_posterior.sum(axis=1) + (1 - n_observations) * zeta_prior

    final_gaussian_etas = (1 - n_observations) * eta_prior + etas_posterior.sum(axis=1)
    final_gaussian_ldas = (1 - n_observations) * lda_prior + lambdas_posterior.sum(axis=1)
    final_gaussian_zeta = -.5 * (-torch.linalg.slogdet(final_gaussian_ldas).logabsdet
                                 + (final_gaussian_etas[..., None].mT @ torch.linalg.inv(final_gaussian_ldas) @
                                    final_gaussian_etas[..., None])[..., 0, 0])
    return sum_zetas - final_gaussian_zeta


def zeta_all(means_posterior_backward,
          covar_posteriors_backward,
          mean_prior_backward,
          covar_prior_backward):
    '''
    Calculates all the factors dependent of theta of log L_theta as defined in (slack document for now...) Following http://www.lucamartino.altervista.org/2003-003.pdf
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

    '''

    def from_canonical_to_sufficient(mean, covar):
        lda = torch.linalg.inv(covar)
        eta = (lda @ mean[..., None])[..., 0]
        return lda, eta, -.5 * (
                    -torch.linalg.slogdet(lda).logabsdet + (mean[..., None].mT @ lda @ mean[..., None])[..., 0, 0])

    n_observations = means_posterior_backward.shape[-2]
    lambdas_posterior, etas_posterior, zetas_posterior = from_canonical_to_sufficient(means_posterior_backward,
                                                                                      covar_posteriors_backward)
    lda_prior, eta_prior, zeta_prior = from_canonical_to_sufficient(mean_prior_backward, covar_prior_backward)

    final_gaussian_etas = (1 - n_observations) * eta_prior + etas_posterior.sum(axis=1)
    final_gaussian_ldas = (1 - n_observations) * lda_prior + lambdas_posterior.sum(axis=1)
    final_gaussian_zeta = -.5 * (-torch.linalg.slogdet(final_gaussian_ldas).logabsdet
                                 + (final_gaussian_etas[..., None].mT @ torch.linalg.inv(final_gaussian_ldas) @
                                    final_gaussian_etas[..., None])[..., 0, 0])
    return final_gaussian_zeta

def mean_backward(theta, t, distribution, nse):
    mu_dist = distribution.loc
    sigma_dist = distribution.covariance_matrix
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    c = (sigma_t ** 2) * torch.linalg.inv((alpha_t * sigma_dist + (sigma_t ** 2) * torch.eye(2)))
    prior_term = c @ mu_dist
    theta_term = 1 / (alpha_t ** .5) * ((torch.eye(2) - c) @ theta.mT).mT

    return prior_term + theta_term


def sigma_backward(t, distribution, nse):
    sigma_dist = distribution.covariance_matrix
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    c = (sigma_t ** 2) * torch.linalg.inv((alpha_t * sigma_dist + (sigma_t ** 2) * torch.eye(2)))

    return ((sigma_t ** 2) / alpha_t) * (torch.eye(2) - c)


def eta(theta, t, posterior_precision, prior, means_posterior, mean_prior, sigma_posterior, sigma_prior, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    c_post = (sigma_t ** 2) * torch.linalg.inv((nse.alpha(t) * sigma_posterior + sigma_t ** 2 * torch.eye(2)))
    c_prior = (sigma_t ** 2) * torch.linalg.inv((nse.alpha(t) * sigma_prior + sigma_t ** 2 * torch.eye(2)))

    post_term = (posterior_precision @ c_post @ means_posterior.sum(axis=1).mT).mT
    prior_term = (prior.precision_matrix @ c_prior @ mean_prior.mT).mT

    return (1 - means_posterior.shape[1]) * prior_term + post_term + (alpha_t ** .5) / (sigma_t ** 2) * theta


def lda(t, n, sigma_posterior, sigma_prior, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    return alpha_t / (sigma_t ** 2) * torch.eye(2) + (alpha_t ** 2) * sigma_prior - n * (alpha_t**2) * (sigma_prior - sigma_posterior)


if __name__ == '__main__':
    # Task
    task = SBIGaussian2d(prior_type="gaussian")
    # Prior and Simulator
    prior = task.prior
    simulator = task.simulator

    # Observation
    theta_true = torch.FloatTensor([-5, 150])  # true parameters
    x_obs = simulator(theta_true)
    x_obs_100 = torch.cat([simulator(theta_true).reshape(1, -1) for i in range(100)], dim=0)

    theta = torch.randn((N_SAMPLES, 2))
    theta.requires_grad = True
    t = torch.tensor([.01])

    nse = NSE(2, 2)

    means_posterior_backward = []
    for x in x_obs_100:
        posterior = task.true_posterior(x)
        means_posterior_backward.append(mean_backward(theta, t, posterior, nse))
    means_posterior_backward = torch.stack(means_posterior_backward).permute(1, 0, 2)

    sigma_posterior_backward = sigma_backward(t, posterior, nse).repeat(N_SAMPLES, 100, 1, 1)

    mean_prior_backward = mean_backward(theta, t, prior.prior, nse)
    sigma_prior_backward = sigma_backward(t, prior.prior, nse).repeat(N_SAMPLES, 1, 1)

    # Veryfing Zeta calculation
    #
    def zeta(theta, t, dist, nse):
        mean = mean_backward(theta, t, dist, nse)
        covar = sigma_backward(t, dist, nse)
        precision_matrix = torch.linalg.inv(covar)
        return -.5 * (-torch.linalg.slogdet(precision_matrix).logabsdet + (mean[..., None].mT @ (precision_matrix @ mean[..., None]))[..., 0, 0])

    theta.grad = None
    # grad zetas via automatic differentiation
    zeta_prior = zeta(theta, t, prior.prior, nse)
    zeta_prior.sum().backward()

    mean_prior_backward = mean_backward(theta, t, prior.prior, nse)
    sigma_prior_backward = sigma_backward(t, prior.prior, nse).repeat(N_SAMPLES, 1, 1)

    # grad zetas via analytic formula
    zeta_prior_ana = -(nse.alpha(t) ** .5) / (nse.sigma(t) ** 2) * mean_prior_backward
    error = (torch.linalg.norm(zeta_prior_ana - theta.grad, axis=-1) / torch.linalg.norm(zeta_prior_ana, axis=-1))
    print(error.std(), error.mean())

    theta.grad = None
    zeta_posterior = zeta(theta, t, posterior, nse)
    zeta_posterior.sum().backward()
    zeta_posterior_ana = -(nse.alpha(t) ** .5) / (nse.sigma(t) ** 2) * means_posterior_backward[:, -1]
    error = (torch.linalg.norm(zeta_posterior_ana - theta.grad, axis=-1) / torch.linalg.norm(zeta_posterior_ana, axis=-1))
    print(error.std(), error.mean())

    # Veryfing calculation zeta all

    # zall = zeta_all(means_posterior_backward, sigma_posterior_backward, mean_prior_backward, sigma_prior_backward)
    # zall.sum().backward()
    # grad_zeta_all = theta.grad
    #
    # lda_ = lda(t, 100, sigma_posterior_backward[0, 0], sigma_prior_backward[0], nse)
    # eta_ = eta(theta, t, task.true_posterior(x_obs).precision_matrix, prior.prior, means_posterior_backward, mean_prior_backward,
    #            sigma_posterior_backward[0, 0], sigma_prior_backward[0], nse)
    # grad_zeta_all_ana = -(nse.alpha(t) ** .5) / (nse.sigma(t) ** 2) * (torch.linalg.inv(lda_) @ eta_.mT).mT
    # error = (torch.linalg.norm(grad_zeta_all_ana -grad_zeta_all, axis=-1) / torch.linalg.norm(grad_zeta_all_ana, axis=-1))
    # print(error.std(), error.mean())




