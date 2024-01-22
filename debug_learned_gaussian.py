import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score
from nse import assure_positive_definitness
from debug_learned_gaussian_old import diffused_tall_posterior_score as diffused_tall_posterior_score_old
from ot.sliced import max_sliced_wasserstein_distance
from tqdm import tqdm
from functools import partial


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
        cov = torch.linalg.inv(torch.linalg.inv(dist_cov_est) + (alpha_t / sigma_t ** 2) * torch.eye(2).to(dist_cov_est.device))#(1 - alpha_t) * dist_cov_est
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


def diffused_tall_posterior_score(
        theta,
        t,
        prior,
        x_obs,
        nse,
        dist_cov_est=None,
        cov_mode='JAC',
        psd_clipping=True,
        scale_gradlogL=False,
        warmup_alpha=.5,
        debug=False,
        score_fn=None
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
                                                         score_fn=nse.score if score_fn is None else score_fn,
                                                         dist_cov_est=dist_cov_est,
                                                         mode=cov_mode)
    # if mean_clipping_range[0]:
    #     mean_0_t = mean_0_t.clip(*mean_clipping_range)

    #scores = nse.score(theta[:, None], x_obs[None], t=t)
    mean_prior_0_t = mean_backward(theta, t, prior_score_fn, nse)
    sigma_prior_0_t = sigma_backward(t, prior.covariance_matrix, nse).repeat(
        theta.shape[0], 1, 1
    )
    prior_score = prior_score_fn(theta, t)
    total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)

    if nse.alpha(t)**.5 > warmup_alpha:
        # sigma_prior_0_t = #= assure_positive_definitness(sigma_prior_0_t.detach(), lim_inf=psd_clipping_range[0],
        #                                               lim_sup=psd_clipping_range[1])
        if psd_clipping:
            sigma_prior_eigvals = torch.linalg.eigvals(sigma_prior_0_t).real
            lim_sup_sigma = (n_obs / (n_obs - 1)) * sigma_prior_eigvals.min()
            sigma_0_t = assure_positive_definitness(sigma_0_t.detach(), lim_inf=0, lim_sup=lim_sup_sigma*.99)

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


def euler_sde_sampler(score_fn, nsamples, beta, device="cpu", ana=False, debug=False, theta_clipping_range=(None, None)):
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


def compute_approx_and_ana(score_fn, N_SAMPLES, score_net):
    (
        theta_learned,
        all_theta_learned,
        gradlogL,
        lda,
        posterior_scores,
        means_posterior_backward,
        sigma_posterior_backward,
    ) = euler_sde_sampler(
        score_fn, N_SAMPLES, beta=score_net.beta, device="cuda:0", debug=True, theta_clipping_range=(-3, 3),
    )
    scores_ana, means_ana, sigmas_ana = vmap(partial(build_analytical_quantities,
                                                     theta=all_theta_learned,
                                                     posterior_fn_ana=partial(task.true_posterior, return_loc_cov=True),
                                                     theta_mean=theta_train.mean(axis=0),
                                                     theta_std=theta_train.std(axis=0),
                                                     nse=score_net))(
        x_obs_100_[:N_OBS],
    )
    scores_ana, means_ana, sigmas_ana = scores_ana.permute(1, 2, 0, 3), means_ana.permute(1, 2, 0,
                                                                                          3), sigmas_ana.permute(1, 0,
                                                                                                                 2, 3)
    # prior_loc = prior.prior.loc.to(all_theta_learned.device)
    # prior_covariance_matrix = prior.prior.covariance_matrix.to(all_theta_learned.device)
    # n_obs = x_obs.shape[0]
    # t = torch.linspace(0, 1, all_theta_learned.shape[0])
    # prior_score_fn = get_vpdiff_gaussian_score(prior_loc, prior_covariance_matrix, score_net)
    # means_prior = vmap(partial(mean_backward, score_fn=prior_score_fn, nse=score_net))(all_theta_learned, t)
    # sigmas_prior = vmap(partial(sigma_backward, dist_cov=prior_covariance_matrix, nse=score_net))(t)
    # prior_score = vmap(prior_score_fn)(all_theta_learned, t)
    # _, _, lda_ana = vmap(partial(log_L, psd_clipping=(0, 1e20)))(means_ana,
    #                                                                    sigmas_ana[:, None].repeat(1,
    #                                                                                               all_theta_learned.shape[
    #                                                                                                   1], 1, 1, 1),
    #                                                                    means_prior,
    #                                                                    sigmas_prior[:, None].repeat(1,
    #                                                                                                 all_theta_learned.shape[
    #                                                                                                     1], 1, 1))
    # lda_ana = lda_ana[:, 0]
    # lda_ana_eigvals = torch.linalg.eigvals(lda_ana).real
    # lda_eigvals = torch.linalg.eigvals(lda).real
    return all_theta_learned#, lda_ana_eigvals, lda_eigvals

def gaussien_wasserstein(X1, X2):
    mean1 = torch.mean(X1, dim=1)
    mean2 = torch.mean(X2, dim=1)
    cov1 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X1)
    cov2 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X2)
    return torch.linalg.norm(mean1 - mean2, dim=-1)**2 + torch.linalg.matrix_norm(cov1 - cov2, dim=(-2, -1))**2


if __name__ == "__main__":
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from nse import NSE, NSELoss
    from sm_utils import train



    torch.manual_seed(1)
    N_TRAIN = 10_000
    N_SAMPLES = 4096

    TYPE_COV_EST = 'DDIM'
    EPS_PERT = 3e-2
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
    # Normalize posterior
    loc_ = (true_posterior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    cov_ = (
            torch.diag(1 / theta_train.std(axis=0))
            @ true_posterior.covariance_matrix
            @ torch.diag(1 / theta_train.std(axis=0))
    )
    true_posterior_ = torch.distributions.MultivariateNormal(
        loc=loc_.cuda(), covariance_matrix=cov_.cuda()
    )

    for N_OBS in [15, 30, 40, 50, 60, 70, 80, 90]:
        cov_est = vmap(lambda x: score_net.ddim(shape=(1000,), x=x, steps=100, eta=.5), randomness='different')(x_obs_100_[:N_OBS].cuda())
        #normalize posterior

        cov_est = vmap(lambda x:torch.cov(x.mT))(cov_est)
        print((torch.linalg.norm(cov_est - cov_.cuda(), dim=(-2, -1)) / torch.linalg.norm(cov_.cuda())).mean())
        ana_score = lambda theta, x, t, **kwargs: get_vpdiff_gaussian_score(mean=x,
                                                                            cov=cov_.cuda(),
                                                                            nse=score_net)(theta, t)
        # score function for tall posterior (learned and analytical)
        score_fn_ana = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=cov_.cuda().repeat(N_OBS, 1, 1),#cov_est,
            cov_mode='GAUSS',
            warmup_alpha=0,
            psd_clipping=False,
            score_fn=ana_score
        )
        eps = torch.randn_like(cov_est) / 2
        eps = eps.mT @ eps
        score_fn_ana_cov_est = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=cov_.cuda().repeat(N_OBS, 1, 1) + (torch.trace(cov_)/2)*EPS_PERT * eps,#cov_est,
            cov_mode='GAUSS',
            warmup_alpha=0.5,
            psd_clipping=True,
            score_fn=ana_score
        )
        score_fn_ana_perturb = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=cov_.cuda().repeat(N_OBS, 1, 1),
            cov_mode='GAUSS',
            warmup_alpha=0.5,
            psd_clipping_range=False,
            score_fn=lambda theta, x, t, **kwargs: ana_score(theta, x, t) + EPS_PERT*torch.randn_like(theta)*theta
        )
        score_fn_net = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=torch.eye(2).cuda().repeat(N_OBS, 1, 1),
            cov_mode='GAUSS',
            warmup_alpha=0.5,
            psd_clipping=True,
        )
        score_fn_gauss = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=cov_est,
            cov_mode='GAUSS',
            warmup_alpha=.5,
            psd_clipping=True,
            scale_gradlogL=True,
        )
        score_fn_gauss_warmup = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=cov_est,
            cov_mode='GAUSS',
            warmup_alpha=0.5,
            psd_clipping=True,
        )
        score_fn_jac = partial(
            diffused_tall_posterior_score,
            prior=prior_,  # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
            dist_cov_est=None,
            cov_mode='JAC',
            warmup_alpha=0.5,
            psd_clipping=True,
        )
        score_fn_jac_scale = partial(
            diffused_tall_posterior_score,
            prior=prior_,  # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
            dist_cov_est=None,
            cov_mode='JAC',
            warmup_alpha=0.5,
            psd_clipping=True,
            scale_gradlogL=True,
        )
        score_fn_jac_old = partial(
            diffused_tall_posterior_score_old,
            prior=prior_,  # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
        )
        # score_fn_jac = partial(
        #     diffused_tall_posterior_score,
        #     prior=prior_, # normalized prior# analytical posterior
        #     x_obs=x_obs_100_[:N_OBS].cuda(), # observations
        #     nse=score_net, # trained score network
        #     dist_cov_est=cov_est,
        #     cov_mode='JAC',
        #     warmup_alpha=0.5,
        #     mean_clipping_range=(-3, 3),
        #     psd_clipping_range=(0, 1e8),
        # )
        # compute results for learned and analytical score during sampling
        # where each euler step is updated with the learned tall posterior score
        samples_per_alg = {}
        for name, score_fun in [
            ('Full Analytical', score_fn_ana),
            #(f'Approx Cov (eps={EPS_PERT})', score_fn_ana_cov_est),
            #(f'Approx score (mean) (eps={EPS_PERT})', score_fn_ana_perturb),
            #('NN / Id', score_fn_net),
            ('NN / Gaussian cov', score_fn_gauss),
            #('NN / Gaussian cov Wup', score_fn_gauss_warmup),
            #('NN / Gaussian cov clip', score_fn_gauss_psd_clip),
            ('2 order Tweedie clip_old', score_fn_jac_old),
            #('2 order Tweedie scale', score_fn_jac_scale),
            ('2 order Tweedie clip', score_fn_jac),

        ]:
            all_thetas_ = compute_approx_and_ana(score_fn=score_fun, N_SAMPLES=N_SAMPLES,
                                                 score_net=score_net)
            samples_per_alg[name] = all_thetas_ * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'N={N_OBS}')
        ind_samples = torch.randint(low=0, high=N_SAMPLES, size=(500,))
        for ax, (name, all_thetas) in zip(axes.flatten(), samples_per_alg.items()):
            ax.scatter(*samples_per_alg['Full Analytical'][-1, ind_samples].T, label='Ground truth')
            ax.scatter(*all_thetas[-1, ind_samples].T, label=name, alpha=.1)
            ax.set_ylim(140, 160)
            ax.set_xlim(0, -10)
            ax.set_title(name)
            leg = ax.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
        for ax in axes[-1]:
            ax.set_xlabel("theta_1")
        for ax in axes[:, 0]:
            ax.set_ylabel("theta_2")
        fig.show()

        sw_fun = vmap(lambda x1, x2: max_sliced_wasserstein_distance(x1, x2, n_projections=1000), randomness='same')
        ref_samples = samples_per_alg['Full Analytical']
        sws = {}
        for name, samples in samples_per_alg.items():
            if name != 'Full Analytical':
                sws[name] = gaussien_wasserstein(ref_samples, samples)
        fig, ax = plt.subplots(1, 1)
        for name, sw in sws.items():
            ax.plot(sw, label=name)
        #ax.set_ylim(0, 100)
        fig.legend()
        fig.show()
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