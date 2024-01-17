import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score
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
        lda = torch.linalg.inv(covar)
        eta = (lda @ mean[..., None])[..., 0]
        return (
            lda,
            eta,
            -0.5
            * (
                    0#-torch.linalg.slogdet(lda).logabsdet
                    + (mean[..., None].mT @ lda @ mean[..., None])[..., 0, 0]
            ),
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
    final_gaussian_zeta = -0.5 * (
            0#-torch.linalg.slogdet(final_gaussian_ldas).logabsdet
            + (
                    final_gaussian_etas[..., None].mT
                    @ torch.linalg.inv(final_gaussian_ldas)
                    @ final_gaussian_etas[..., None]
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
    return torch.linalg.inv(torch.linalg.inv(dist_cov) + (alpha_t / sigma_t ** 2) * torch.eye(2).to(dist_cov.device))
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


def tweedies_approximation(x, theta, t, score_fn, nse, mode='vmap'):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    if mode == 'vmap':
        def score_jac(theta, x):
            score = score_fn(theta=theta, t=t, x=x)
            return score, score
        jac_score, score = vmap(lambda theta: vmap(jacrev(score_jac, has_aux=True), in_dims=(None, 0))(theta, x))(theta)
    else:
        raise NotImplemented
    mean = 1 / (alpha_t**0.5) * (theta[:, None] + sigma_t**2 * score)
    cov = (sigma_t**2 / alpha_t) * (torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2)*jac_score)
    return cov, mean, score


def sigma_backward_autodiff(theta, x, t, score_fn, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    def mean_to_jac(theta, x):
        score = score_fn(theta=theta, t=t, x=x)
        # mu = mean_backward(theta, t, score_fn, nse, x=x)
        mu = 1 / (alpha_t**0.5) * (theta + sigma_t**2 * score)
        return mu, (mu, score)

    grad_mean, _ = vmap(jacrev(mean_to_jac, has_aux=True))(theta, x)

    # # extract diag terms of grad_mean and create diag_matrix
    # diag_terms = torch.stack([grad_mean[:,0,0], grad_mean[:,1,1]],dim=-1)
    # # clip negative values
    # diag_terms = diag_terms.abs().clip(1e-1,1e20)

    # grad_mean = torch.diag_embed(diag_terms)
    return (sigma_t**2 / (alpha_t**0.5)) * grad_mean
    # return torch.eye(2).repeat(theta.shape[0], 1, 1).to(theta.device)


def diffused_tall_posterior_score(
        theta,
        t,
        prior,
        x_obs,
        nse,
        debug=False,
):
    # device
    prior.loc = prior.loc.to(theta.device)
    prior.covariance_matrix = prior.covariance_matrix.to(theta.device)
    n_obs = x_obs.shape[0]

    prior_score_fn = get_vpdiff_gaussian_score(prior.loc, prior.covariance_matrix, nse)

    theta.requires_grad = True
    theta.grad = None

    sigma_0_t, mean_0_t, scores = tweedies_approximation(x=x_obs,
                                                         theta=theta,
                                                         nse=nse,
                                                         t=t,
                                                         score_fn=nse.score)

    scores = nse.score(theta[:, None], x_obs[None], t=t)
    mean_prior_0_t = mean_backward(theta, t, prior_score_fn, nse)
    sigma_prior_0_t = sigma_backward(t, prior.covariance_matrix, nse).repeat(
        theta.shape[0], 1, 1
    )
    prior_score = prior_score_fn(theta, t)
    #sigma_prior_0_t = assure_positive_definitness(sigma_prior_0_t.detach())

    total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)


    # This works but it's strange!
    #smallest_eig = torch.linalg.eigvals(lda).real.min(dim=-1).values
    if nse.alpha(t)**.5 > .5:#(nse.sigma(t) / (nse.alpha(t)**.5)) < .05: #smallest_eig > 0).all():
        sigma_prior_eigvals = torch.linalg.eigvals(sigma_prior_0_t).real[0]
        lim_sup_sigma = (n_obs / (n_obs - 1)) * sigma_prior_eigvals.min()
        sigma_0_t = assure_positive_definitness(sigma_0_t.detach(), lim_sup=lim_sup_sigma*1)
        logL, _, lda = log_L(
            mean_0_t,
            sigma_0_t.detach(),
            mean_prior_0_t,
            sigma_prior_0_t.detach(),
        )
        logL.sum().backward()
        gradlogL = theta.grad
        is_positive = (torch.linalg.eigvals(lda).real.min(dim=-1).values > 0)
        total_score = total_score + gradlogL * ((is_positive[..., None]).float())
        #total_score = total_score + gradlogL
    else:
        gradlogL = torch.zeros_like(total_score)
        lda = torch.zeros_like(sigma_0_t[:, 0])
    #
    # sigma_prior_eigvals = torch.linalg.eigvals(sigma_prior_0_t).real[0]
    # lim_sup_sigma = (n_obs / (n_obs - 1)) * sigma_prior_eigvals.min()
    # sigma_0_t = assure_positive_definitness(sigma_0_t.detach(), lim_inf=nse.sigma(t), lim_sup=lim_sup_sigma*2)
    # logL, _, lda = log_L(
    #     mean_0_t,
    #     sigma_0_t.detach(),
    #     mean_prior_0_t,
    #     sigma_prior_0_t.detach(),
    # )
    # logL.sum().backward()
    # gradlogL = theta.grad
    # is_positive = (torch.linalg.eigvals(lda).real.min(dim=-1).values > 0)
    # total_score = total_score + gradlogL*((is_positive[..., None]).float())
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


def euler_sde_sampler(score_fn, nsamples, beta, device="cpu", ana=False, debug=False):
    theta_t = torch.randn((nsamples, 2)).to(device)  # (nsamples, 2)
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
            score = score_fn(theta_t, t, ana=ana)
        score = score.detach()

        drift = f - g * g * score
        diffusion = g

        # euler-maruyama step
        theta_t = (
                theta_t.detach()
                + drift * dt
                + diffusion * torch.randn_like(theta_t) * torch.abs(dt) ** 0.5
        ).clip(-3, 3)
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


def build_analytical_quantities(x, theta, posterior_fn_ana, theta_mean, theta_std, nse):
    # Analytical score
    posterior_loc, posterior_cov = posterior_fn_ana(x.cpu())
    # rescale the mean and covariance of the posterior
    if theta_mean is not None and theta_std is not None:
        posterior_loc = (posterior_loc - theta_mean) / theta_std
        posterior_cov = (
                torch.diag(1 / theta_std)
                @ posterior_cov
                @ torch.diag(1 / theta_std)
        )
    posterior_score_fn_ana = get_vpdiff_gaussian_score(
        posterior_loc, posterior_cov, nse
    )
    t = torch.linspace(0, 1, theta.shape[0]).to(theta.device)
    sigma_posterior_ana = vmap(partial(sigma_backward, dist_cov=posterior_cov, nse=nse))(t)
    posterior_score_ana = vmap(posterior_score_fn_ana)(theta, t)
    mean_backward_ana = vmap(partial(mean_backward, score_fn=posterior_score_fn_ana,nse=nse))(theta, t)

    return posterior_score_ana, mean_backward_ana,  sigma_posterior_ana



if __name__ == "__main__":
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from nse import NSE, NSELoss
    from sm_utils import train

    from tqdm import tqdm
    from functools import partial

    torch.manual_seed(1)

    N_TRAIN = 10_000
    N_SAMPLES = 4096
    N_OBS = 100

    # Task
    task = SBIGaussian2d(prior_type="gaussian")
    # Prior and Simulator
    prior = task.prior
    simulator = task.simulator

    # Observations
    theta_true = torch.FloatTensor([-5, 150])  # true parameters
    x_obs = simulator(theta_true)  # x_obs ~ simulator(theta_true)
    x_obs_100 = torch.cat(
        [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
    )

    # True posterior: p(theta|x_obs)
    true_posterior = task.true_posterior(x_obs)

    # Train data
    theta_train = task.prior.sample((N_TRAIN,))
    x_train = simulator(theta_train)

    # normalize theta
    theta_train_ = (theta_train - theta_train.mean(axis=0)) / theta_train.std(axis=0)

    # normalize x
    x_train_ = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_obs_ = (x_obs - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_obs_100_ = (x_obs_100 - x_train.mean(axis=0)) / x_train.std(axis=0)

    # # train score network
    # dataset = torch.utils.data.TensorDataset(theta_train_.cuda(), x_train_.cuda())
    # score_net = NSE(theta_dim=2, x_dim=2, hidden_features=[128, 256, 128]).cuda()

    # avg_score_net = train(
    #     model=score_net,
    #     dataset=dataset,
    #     loss_fn=NSELoss(score_net),
    #     n_epochs=200,
    #     lr=1e-3,
    #     batch_size=256,
    #     prior_score=False, # learn the prior score via the classifier-free guidance approach
    # )
    # score_net = avg_score_net.module
    # torch.save(score_net, "score_net.pkl")

    # load score network
    score_net = torch.load("score_net.pkl").cuda()

    # normalize prior
    loc_ = (prior.prior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    cov_ = (
            torch.diag(1 / theta_train.std(axis=0))
            @ prior.prior.covariance_matrix
            @ torch.diag(1 / theta_train.std(axis=0))
    )
    prior_ = torch.distributions.MultivariateNormal(
        loc=loc_.cuda(), covariance_matrix=cov_.cuda()
    )

    #Normalize posterior
    loc_ = (true_posterior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    cov_ = (
            torch.diag(1 / theta_train.std(axis=0))
            @ true_posterior.covariance_matrix
            @ torch.diag(1 / theta_train.std(axis=0))
    )
    true_posterior_ = torch.distributions.MultivariateNormal(
        loc=loc_.cuda(), covariance_matrix=cov_.cuda()
    )

    #normalize posterior

    # score function for tall posterior (learned and analytical)
    score_fn = partial(
        diffused_tall_posterior_score,
        prior=prior_, # normalized prior# analytical posterior
        x_obs=x_obs_100_[:N_OBS].cuda(), # observations
        nse=score_net, # trained score network
    )

    # compute results for learned and analytical score during sampling
    # where each euler step is updated with the learned tall posterior score
    (
        theta_learned,
        all_theta_learned,
        gradlogL,
        lda,
        posterior_scores,
        means_posterior_backward,
        sigma_posterior_backward,
    ) = euler_sde_sampler(
        score_fn, N_SAMPLES, beta=score_net.beta, device="cuda:0", debug=True
    )
    scores_ana, means_ana, sigmas_ana = vmap(partial(build_analytical_quantities,
                                                     theta=all_theta_learned,
                                                     posterior_fn_ana=partial(task.true_posterior, return_loc_cov=True),
                                                     theta_mean=theta_train.mean(axis=0),
                                                     theta_std=theta_train.std(axis=0),
                                                     nse=score_net))(
        x_obs_100_[:N_OBS],
    )
    scores_ana, means_ana, sigmas_ana = scores_ana.permute(1, 2, 0, 3), means_ana.permute(1, 2, 0, 3), sigmas_ana.permute(1, 0, 2, 3)
    prior_loc = prior.prior.loc.to(all_theta_learned.device)
    prior_covariance_matrix = prior.prior.covariance_matrix.to(all_theta_learned.device)
    n_obs = x_obs.shape[0]
    t = torch.linspace(0, 1, all_theta_learned.shape[0])

    prior_score_fn = get_vpdiff_gaussian_score(prior_loc, prior_covariance_matrix, score_net)
    means_prior = vmap(partial(mean_backward, score_fn=prior_score_fn, nse=score_net))(all_theta_learned, t)
    sigmas_prior = vmap(partial(sigma_backward, dist_cov=prior_covariance_matrix, nse=score_net))(t)
    prior_score = vmap(prior_score_fn)(all_theta_learned, t)

    _, _, lda_ana = vmap(log_L)(means_ana,
                                sigmas_ana[:, None].repeat(1, all_theta_learned.shape[1], 1, 1, 1),
                                means_prior,
                                sigmas_prior[:, None].repeat(1, all_theta_learned.shape[1], 1, 1))
    lda_ana = lda_ana[:, 0]
    lda_ana_eigvals = torch.linalg.eigvals(lda_ana).real
    lda_eigvals = torch.linalg.eigvals(lda).real

    plt.plot(score_net.alpha(t)**.5, lda_ana_eigvals.min(dim=-1).values, 'bx')
    plt.scatter((torch.flip(score_net.alpha(t)**.5, dims=(0,)))[:-1][:, None].repeat(1, lda_eigvals.shape[1]), lda_eigvals.min(dim=-1).values, color='blue')
    plt.plot(score_net.alpha(t) ** .5, lda_ana_eigvals.max(dim=-1).values, 'rx')
    plt.scatter((torch.flip(score_net.alpha(t) ** .5, dims=(0,)))[:-1][:, None].repeat(1, lda_eigvals.shape[1]),
                lda_eigvals.max(dim=-1).values, color='red')

    #plt.plot(torch.flip(t, dims=(0,)), lda_ana_eigvals.min(dim=-1).values, color='orange')
    #plt.scatter(t[:-1][:, None].repeat(1, lda_eigvals.shape[1]), lda_eigvals.min(dim=-1).values, color='orange')
    plt.yscale('log')
    plt.xlim(.8, 1)
    plt.show()

    plt.plot(torch.flip(t, dims=(0,)), score_net.alpha(t)**.5 / score_net.sigma(t))
    plt.yscale('log')
    plt.show()
    # compute analytical samples only
    # theta_ana, all_theta_ana = euler_sde_sampler(
    #     score_fn, N_SAMPLES, beta=score_net.beta, device="cuda:0", ana=True
    # )

    #posterior_scores_ana = vmap(vmap())
    # gradlogL_ana =
    # lda_ana,
    # posterior_scores_ana,
    # means_posterior_backward_ana,
    # sigma_posterior_backward_ana,
    #
    # results_dict = {
    #     "all_theta_learned": all_theta_learned,
    #     "all_theta_ana": all_theta_ana,
    #     "gradlogL": gradlogL,
    #     "lda": lda,
    #     "posterior_scores": posterior_scores,
    #     "means_posterior_backward": means_posterior_backward,
    #     "sigma_posterior_backward": sigma_posterior_backward,
    #     "gradlogL_ana": gradlogL_ana,
    #     "lda_ana": lda_ana,
    #     "posterior_scores_ana": posterior_scores_ana,
    #     "means_posterior_backward_ana": means_posterior_backward_ana,
    #     "sigma_posterior_backward_ana": sigma_posterior_backward_ana,
    # }
    # torch.save(results_dict, f"results_dict_n_obs_{N_OBS}_assure_psd_prior.pkl")

    # norm of the difference between the learned and analytic quantities
    # means_posterior_diff = torch.linalg.norm(
    #     means_posterior_backward_ana - means_posterior_backward, axis=-1
    # )
    # sigma_posterior_diff = torch.linalg.norm(
    #     sigma_posterior_backward_ana - sigma_posterior_backward, axis=(-2, -1)
    # )
    # posterior_scores_diff = torch.linalg.norm(
    #     posterior_scores_ana - posterior_scores, dim=-1
    # )
    # gradlogL_diff = torch.linalg.norm(gradlogL_ana - gradlogL, dim=-1)

    # 
    theta_learned = theta_learned.detach().cpu()
    all_theta_learned = all_theta_learned * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]
    # unnormalize sample s
    theta_learned = theta_learned * theta_train.std(axis=0) + theta_train.mean(axis=0)
    #s = posterior_scores_diff.mean(dim=0)
    # s = posterior_scores_diff.max(dim=0).values
    #s = s - s.min()
    plt.scatter(*true_posterior.sample((N_SAMPLES,)).T, label="True (n=1)", alpha=0.1)
    plt.scatter(*all_theta_learned[-1].T, label=f"DDGM (n={N_OBS})",)#=s.clip(0, 100))
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.ylim(140, 170)
    plt.xlim(10, -10)
    plt.legend()
    #plt.savefig(f"samples_n_obs_{N_OBS}_assure_psd.png")
    #plt.clf()
    plt.show()


    # # Erreur des score en fonction de la distance a theta true
    # plt.scatter(
    #     posterior_scores_diff.max(dim=0).values,
    #     torch.linalg.norm(theta_learned - theta_true[None,], axis=-1),
    # )
    # # plt.yscale('log')
    # plt.ylabel("Distance to true theta")
    # plt.xlabel("Max score error through sampling")
    # plt.show()

    # plt.scatter(
    #     posterior_scores_diff.mean(dim=0),
    #     torch.linalg.norm(theta_learned - theta_true[None,], axis=-1),
    # )
    # # plt.yscale('log')
    # plt.ylabel("Distance to true theta")
    # plt.xlabel("Mean score error through sampling")
    # plt.show()

    # # Erreur de score a la premiere iteration
    # plt.scatter(
    #     posterior_scores_diff[100],
    #     torch.linalg.norm(all_theta_learned[100].detach().cpu(), dim=-1),
    # )
    # plt.show()

    # # Tout erreur vs tout theta
    # plt.scatter(
    #     torch.linalg.norm(all_theta_learned[:-1], dim=-1).flatten(),
    #     posterior_scores_diff.flatten(),
    # )
    # plt.show()

    # get the index of the worst learned theta (furthest from the true theta)
    # indices_best_to_worst = torch.argsort(
    #     torch.linalg.norm(theta_learned - theta_true[None, ...], axis=-1)
    # )
    # ind = indices_best_to_worst[-1]
    # # # plt.plot(posterior_scores_diff[:, ind])
    # # # plt.show()
    #
    # # # plt.plot(torch.linalg.norm(all_theta_learned[:, ind], dim=-1))
    # # # plt.show()
    #
    # # index before the norm of the learned theta becomes greater than 4.5
    # ind_diff = (
    #     torch.diff(torch.linalg.norm(all_theta_learned[:, ind], axis=-1) > 4.5).float().argmax()
    # )
    # print(ind_diff)
    # range = torch.arange(ind_diff - 3, ind_diff + 4)
    #
    # print(torch.linalg.norm(all_theta_learned[:, ind], axis=-1)[range])
    # print(posterior_scores_diff[range, ind])
    # print(means_posterior_diff[range, ind].max(axis=-1).values)
    # print(sigma_posterior_diff[range, ind].max(axis=-1).values)
    # print()
    # print(torch.linalg.norm(gradlogL, axis=-1)[range, ind])
    # print(torch.linalg.norm(gradlogL_ana, axis=-1)[range, ind])
    # print(gradlogL_diff[range, ind])
    # print()
    # print(torch.linalg.eigvals(sigma_posterior_backward[range, ind][:,0,:,:]))
    # print(torch.linalg.eigvals(sigma_posterior_backward_ana[range, ind][:,0,:,:]))