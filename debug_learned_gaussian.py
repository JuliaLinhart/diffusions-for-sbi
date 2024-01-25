import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score
from nse import assure_positive_definitness
from debug_learned_gaussian_old import diffused_tall_posterior_score as diffused_tall_posterior_score_old
from ot.sliced import max_sliced_wasserstein_distance
from tqdm import tqdm
from functools import partial


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    L, V = torch.linalg.eig(matrix)
    L = L.real
    V = V.real
    return V @ torch.diag_embed(L.pow(p)) @ torch.linalg.inv(V)



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
    eye = torch.eye(dist_cov.shape[-1]).to(alpha_t.device)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    if (alpha_t**.5) < 0.5:
        return torch.linalg.inv(torch.linalg.inv(dist_cov) + (alpha_t / sigma_t ** 2) * eye)
    return (((sigma_t ** 2) / alpha_t) * eye
            - (((sigma_t ** 2)**2 / alpha_t)) * torch.linalg.inv(
                alpha_t * (dist_cov.to(alpha_t.device) - eye)
                + eye))


def prec_matrix_backward(t, dist_cov, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    eye = torch.eye(dist_cov.shape[-1]).to(alpha_t.device)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    #return torch.linalg.inv(dist_cov * alpha_t + (sigma_t**2) * eye)
    return (torch.linalg.inv(dist_cov) + (alpha_t / sigma_t ** 2) * eye)



def tweedies_approximation(x, theta, t, score_fn, nse, dist_cov_est=None, mode='JAC', clip_mean_bounds = (None, None)):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    if mode == 'JAC':
        def score_jac(theta, x):
            score = score_fn(theta=theta, t=t, x=x)
            return score, score
        jac_score, score = vmap(lambda theta: vmap(jacrev(score_jac, has_aux=True), in_dims=(None, 0))(theta, x))(theta)
        cov = (sigma_t**2 / alpha_t) * (torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2)*jac_score)
        prec = torch.linalg.inv(cov)
    elif mode == 'GAUSS':
        prec = prec_matrix_backward(t=t, dist_cov=dist_cov_est, nse=nse)
        # eye = torch.eye(dist_cov_est.shape[-1]).to(alpha_t.device)
        # # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
        # cov = torch.linalg.inv(torch.linalg.inv(dist_cov_est) + (alpha_t / sigma_t ** 2) * eye)
        prec = prec[None].repeat(theta.shape[0], 1, 1, 1)
        score = vmap(lambda theta: vmap(partial(score_fn, t=t),
                                        in_dims=(None, 0),
                                        randomness='different')(theta, x),
                     randomness='different')(theta)
    else:
        raise NotImplemented("Available methods are GAUSS, PSEUDO, JAC")
    mean = (1 / (alpha_t**0.5) * (theta[:, None] + sigma_t**2 * score))
    if clip_mean_bounds[0]:
        mean = mean.clip(*clip_mean_bounds)
    return prec, mean, score


def diffused_tall_posterior_score(
        theta,
        t,
        prior,
        x_obs,
        nse,
        score_fn=None,
        dist_cov_est=None,
        cov_mode='JAC',
):
    # device
    prior.loc = prior.loc.to(theta.device)
    prior.covariance_matrix = prior.covariance_matrix.to(theta.device)
    n_obs = x_obs.shape[0]

    prior_score_fn = get_vpdiff_gaussian_score(prior.loc, prior.covariance_matrix, nse)

    # theta.requires_grad = True
    # theta.grad = None
    alpha_t = nse.alpha(t)
    # Tweedies approx for p_{0|t}
    prec_0_t, mean_0_t, scores = tweedies_approximation(x=x_obs,
                                                        theta=theta,
                                                        nse=nse,
                                                        t=t,
                                                        score_fn=nse.score if score_fn is None else score_fn,
                                                        dist_cov_est=dist_cov_est,
                                                        mode=cov_mode)
    mean_prior_0_t = mean_backward(theta, t, prior_score_fn, nse)
    prec_prior_0_t = prec_matrix_backward(t, prior.covariance_matrix, nse).repeat(
        theta.shape[0], 1, 1
    )
    prior_score = prior_score_fn(theta, t)
    prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
    prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
    lda = prec_prior_0_t*(1-n_obs) + prec_0_t.sum(dim=1)
    weighted_scores = prec_score_prior + (prec_score_post - prec_score_prior[:, None]).sum(dim=1)

    total_score = torch.linalg.solve(A=lda, B=weighted_scores)
    return total_score #/ (1 + (1/n_obs)*torch.abs(total_score))


def euler_sde_sampler(score_fn, nsamples, beta, device="cpu", theta_clipping_range=(None, None)):
    theta_t = torch.randn((nsamples, 2)).to(device)  # (nsamples, 2)
    time_pts = torch.linspace(1, 0, 1000).to(device)  # (ntime_pts,)
    theta_list = [theta_t.clone().cpu()]
    for i in tqdm(range(len(time_pts) - 1)):
        t = time_pts[i]
        dt = time_pts[i + 1] - t

        # calculate the drift and diffusion terms
        f = -0.5 * beta(t) * theta_t
        g = beta(t) ** 0.5

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

    return torch.stack(theta_list)

def gaussien_wasserstein(ref_mu, ref_cov, X2):
    mean2 = torch.mean(X2, dim=1)
    sqrtcov1 = _matrix_pow(ref_cov, .5)
    cov2 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X2)
    covterm = torch.func.vmap(torch.trace)(ref_cov + cov2 - 2 * _matrix_pow(sqrtcov1 @ cov2 @ sqrtcov1, .5))
    return (1*torch.linalg.norm(ref_mu - mean2, dim=-1)**2 + 1*covterm)**.5


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

    for N_OBS in [2, 30, 40, 50, 60, 70, 80, 90]:
        cov_est = vmap(lambda x: score_net.ddim(shape=(1000,), x=x, steps=100, eta=.5), randomness='different')(x_obs_100_[:N_OBS].cuda())
        #normalize posterior
        cov_est = vmap(lambda x:torch.cov(x.mT))(cov_est)


        t = torch.linspace(0, 1, 1000)
        lik_cov = torch.FloatTensor([[1, task.rho], [task.rho, 1]]).cuda()
        lik_cov_ = (
                torch.diag(1 / theta_train.std(axis=0)).cuda()
                @ lik_cov
                @ torch.diag(1 / theta_train.std(axis=0)).cuda()
        ).cuda()
        posterior_cov_0 = torch.linalg.inv((N_OBS * torch.linalg.inv(lik_cov) + torch.linalg.inv(prior.prior.covariance_matrix.cuda())))

        posterior_cov_0_ = torch.linalg.inv((N_OBS * torch.linalg.inv(lik_cov_) + torch.linalg.inv(prior_.covariance_matrix)))
        #posterior_cov_diffused = (posterior_cov_0[None] * score_net.alpha(t)[:, None, None] +
        #                          (1 - score_net.alpha(t))[:, None, None] * torch.eye(posterior_cov_0.shape[0])[None])
        posterior_mean_0 = posterior_cov_0 @ (torch.linalg.inv(prior.prior.covariance_matrix.cuda()) @ prior.prior.loc[:, None].cuda() +
                                              torch.linalg.inv(lik_cov) @ x_obs_100[:N_OBS].sum(dim=0).cuda()[:, None])[..., 0]
        posterior_mean_0_ = posterior_cov_0_ @ (torch.linalg.inv(prior_.covariance_matrix) @ prior_.loc[:, None] +
                                              torch.linalg.inv(lik_cov_) @ x_obs_100_[:N_OBS].sum(dim=0).cuda()[:, None])[..., 0]
        # posterior_mean_0_ = posterior_mean_0 - theta_train.mean(dim=0)
        # posterior_cov_0_ = (
        #         torch.diag(1 / theta_train.std(axis=0))
        #         @ posterior_cov_0
        #         @ torch.diag(1 / theta_train.std(axis=0))
        # )
        posterior_cov_diffused_ = (posterior_cov_0_[None] * score_net.alpha(t)[:, None, None].cuda() +
                                   (1 - score_net.alpha(t).cuda())[:, None, None] * torch.eye(posterior_cov_0_.shape[0])[None].cuda())
        posterior_mean_diffused_ = posterior_mean_0_[None] * score_net.alpha(t)[:, None].cuda()


        score_fn_gauss = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            dist_cov_est=cov_est,
            cov_mode='GAUSS',
        )

        score_fn_jac = partial(
            diffused_tall_posterior_score,
            prior=prior_, # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(), # observations
            nse=score_net, # trained score network
            cov_mode='JAC',
        )
        mse_scores = {'GAUSS': [], "JAC": []}
        for mu_, cov_, t in zip(posterior_mean_diffused_,
                                posterior_cov_diffused_,
                                torch.linspace(0, 1, 1000)):
            dist = torch.distributions.MultivariateNormal(loc=mu_, covariance_matrix=cov_)
            ref_samples = dist.sample((1000,))
            ref_samples.requires_grad_(True)
            dist.log_prob(ref_samples).sum().backward()
            real_score = ref_samples.grad.clone()
            ref_samples.grad = None
            ref_samples.requires_grad_(False)

            approx_score_gauss = score_fn_gauss(ref_samples.cuda(), t.cuda())
            approx_score_jac = score_fn_jac(ref_samples.cuda(), t.cuda())
            error_gauss = torch.linalg.norm(approx_score_gauss - real_score, axis=-1)**2
            error_jac = torch.linalg.norm(approx_score_jac - real_score, axis=-1)**2
            error_mean = torch.linalg.norm(real_score - real_score.mean(dim=0)[None], dim=-1)**2
            r2_score_gauss = 1 - (error_gauss.sum() / error_mean.sum())
            r2_score_jac = 1 - (error_jac.sum() / error_mean.sum())
            mse_scores["GAUSS"].append(r2_score_gauss.item())
            mse_scores["JAC"].append(r2_score_jac.item())
        plt.plot(torch.linspace(0, 1, 1000),mse_scores["GAUSS"], label='GAUSS', alpha=.8)
        plt.plot(torch.linspace(0, 1, 1000),mse_scores["JAC"], label='JAC', alpha=.8)
        plt.suptitle(N_OBS)
        plt.legend()
        plt.ylim(-1.1, 1.1)
        #plt.yscale('log')
        plt.ylabel('R2 score')
        plt.xlabel('Diffusion time')
        plt.show()
        # compute results for learned and analytical score during sampling
        # where each euler step is updated with the learned tall posterior score
        samples_per_alg = {}
        for name, score_fun in [
            ('NN / Gaussian cov', score_fn_gauss),
            ('JAC', score_fn_jac),
            #(f'Approx Cov (eps={EPS_PERT})', score_fn_ana_cov_est),
            #(f'Approx score (mean) (eps={EPS_PERT})', score_fn_ana_perturb),
            #('NN / Id', score_fn_net),
            #('NN / Gaussian cov Wup', score_fn_gauss_warmup),
            #('NN / Gaussian cov clip', score_fn_gauss_psd_clip),
            #('2 order Tweedie clip_old', score_fn_jac_old),
            #('2 order Tweedie scale', score_fn_jac_scale),
            #('2 order Tweedie clip', score_fn_jac),

        ]:
            samples_ = euler_sde_sampler(
                score_fun, N_SAMPLES, beta=score_net.beta, device="cuda:0", theta_clipping_range=(-3, 3),
            )
            samples_per_alg[name] = samples_ * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]

        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'N={N_OBS}')
        ref_samples = torch.distributions.MultivariateNormal(loc=posterior_mean_0, covariance_matrix=posterior_cov_0).sample((500,)).cpu()
        ind_samples = torch.randint(low=0, high=N_SAMPLES, size=(1000,))
        for ax, (name, all_thetas) in zip(axes.flatten(), samples_per_alg.items()):
            ax.scatter(*ref_samples.T, label='Ground truth')
            ax.scatter(*all_thetas[-1, ind_samples].T, label=name, alpha=.8)
            ax.set_ylim(140, 160)
            ax.set_xlim(0, -10)
            ax.set_title(name)
            leg = ax.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
        # for ax in axes[-1]:
        #     ax.set_xlabel("theta_1")
        # for ax in axes[:, 0]:
        #     ax.set_ylabel("theta_2")
        fig.show()
        #
        sw_fun = vmap(lambda x1, x2: max_sliced_wasserstein_distance(x1, x2, n_projections=1000), randomness='same')
        sws = {}
        for name, samples in samples_per_alg.items():
            if name != 'Full Analytical':
                sws[name] = gaussien_wasserstein(ref_mu=torch.flip(posterior_mean_diffused, dims=(0,)),
                                                 ref_cov=torch.flip(posterior_cov_diffused, dims=(0,)),
                                                 X2=samples)
        fig, ax = plt.subplots(1, 1)
        for name, sw in sws.items():
            ax.plot(sw, label=name)
        #ax.set_xlim(900, 1000)
        ax.set_yscale('log')
        fig.legend()
        fig.show()
