import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score


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
                    -torch.linalg.slogdet(lda).logabsdet
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
            -torch.linalg.slogdet(final_gaussian_ldas).logabsdet
            + (
                    final_gaussian_etas[..., None].mT
                    @ torch.linalg.inv(final_gaussian_ldas)
                    @ final_gaussian_etas[..., None]
            )[..., 0, 0]
    )
    return sum_zetas - final_gaussian_zeta


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

    return (
            (sigma_t**2)
            / alpha_t
            * (
                    torch.eye(2).to(alpha_t.device)
                    + sigma_t**2
                    * (-1)
                    * torch.linalg.inv(
                (
                        nse.alpha(t) * dist_cov.to(alpha_t.device)
                        + sigma_t**2 * torch.eye(2).to(alpha_t.device)
                )
            )
            )
    )


def sigma_backward_autodiff(theta, x, t, score_fn, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    def mean_to_jac(theta, x):
        score = score_fn(theta=theta, t=t, x=x)
        # mu = mean_backward(theta, t, score_fn, nse, x=x)
        mu = 1/(alpha_t**.5) * (theta + sigma_t**2 * score)
        return mu, (mu, score)

    grad_mean, _ = vmap(jacrev(mean_to_jac, has_aux=True))(theta, x)
    return (sigma_t**2 / (alpha_t**0.5)) * grad_mean
    # return torch.eye(2).repeat(theta.shape[0], 1, 1).to(theta.device)


def diffused_tall_posterior_score(
        theta,
        t,
        prior,
        posterior_fn_ana,
        x_obs,
        x_obs_,
        nse,
        theta_mean=None,
        theta_std=None,
        debug=False,
):
    # device
    prior.loc = prior.loc.to(theta.device)
    prior.covariance_matrix = prior.covariance_matrix.to(theta.device)

    n_obs = len(x_obs)

    prior_score_fn = get_vpdiff_gaussian_score(prior.loc, prior.covariance_matrix, nse)

    theta.requires_grad = True
    theta.grad = None

    means_posterior_backward = []
    sigmas_posterior_backward = []
    posterior_scores = []
    means_posterior_backward_ana = []
    sigmas_posterior_backward_ana = []
    posterior_scores_ana = []
    for i in range(n_obs):
        #Analytical score
        posterior = posterior_fn_ana(x_obs[i].cpu())
        # rescale the mean and covariance of the posterior
        if theta_mean is not None and theta_std is not None:
            loc = (posterior.loc - theta_mean) / theta_std
            cov = (
                    torch.diag(1 / theta_std)
                    @ posterior.covariance_matrix
                    @ torch.diag(1 / theta_std)
            )
            posterior = torch.distributions.MultivariateNormal(
                loc=loc.to(theta.device), covariance_matrix=cov.to(theta.device)
            )
        posterior_score_fn_ana = get_vpdiff_gaussian_score(
            posterior.loc, posterior.covariance_matrix, nse
        )
        sigma_posterior_ana = sigma_backward(t, posterior.covariance_matrix, nse).repeat(
            theta.shape[0], 1, 1
        )
        sigmas_posterior_backward_ana.append(
            sigma_posterior_ana
        )
        posterior_scores_ana.append(posterior_score_fn_ana(theta=theta, t=t))
        means_posterior_backward_ana.append(
            mean_backward(theta, t, posterior_score_fn_ana, nse)
        )

        #Learned score
        posterior_score_fn = nse.score
        x = x_obs_[i].to(theta.device).repeat(theta.shape[0], 1)
        kwargs = {"x": x_obs_[i].to(theta.device)}
        sigmas_posterior_backward.append(
            sigma_backward_autodiff(theta, x, t, posterior_score_fn, nse)
        )

        posterior_scores.append(posterior_score_fn(theta=theta, t=t, **kwargs))
        means_posterior_backward.append(
            mean_backward(theta, t, posterior_score_fn, nse, **kwargs)
        )

    means_posterior_backward_ana = torch.stack(means_posterior_backward_ana).permute(1, 0, 2)
    sigma_posterior_backward_ana = torch.stack(sigmas_posterior_backward_ana).permute(
        1, 0, 2, 3
    )
    means_posterior_backward = torch.stack(means_posterior_backward).permute(1, 0, 2)
    sigma_posterior_backward = torch.stack(sigmas_posterior_backward).permute(
        1, 0, 2, 3
    )

    mean_prior_backward = mean_backward(theta, t, prior_score_fn, nse)
    sigma_prior_backward = sigma_backward(t, prior.covariance_matrix, nse).repeat(
        theta.shape[0], 1, 1
    )

    logL = log_L(
        means_posterior_backward,
        sigma_posterior_backward,
        mean_prior_backward,
        sigma_prior_backward,
    )
    logL.sum().backward()

    gradlogL = theta.grad
    posterior_scores = torch.stack(posterior_scores).sum(axis=0)
    posterior_scores_ana = torch.stack(posterior_scores_ana).sum(axis=0)
    prior_score = prior_score_fn(theta, t)

    if debug:
        return (
            (1 - n_obs) * prior_score + posterior_scores + gradlogL,
            gradlogL.detach().cpu(),
            posterior_scores.detach().cpu(),
            means_posterior_backward.detach().cpu(),
            sigma_posterior_backward.detach().cpu(),
            posterior_scores_ana.detach().cpu(),
            means_posterior_backward_ana.detach().cpu(),
            sigma_posterior_backward_ana.detach().cpu(),
        )
    else:
        return (1 - n_obs) * prior_score + posterior_scores + gradlogL


def euler_sde_sampler(score_fn, nsamples, beta, device="cpu", debug=False):
    theta_t = torch.randn((nsamples, 2)).to(device)  # (nsamples, 2)
    time_pts = torch.linspace(1, 0, 1000).to(device)  # (ntime_pts,)
    theta_list = [theta_t]
    gradlogL_list = []
    posterior_scores_list = []
    means_posterior_backward_list = []
    sigma_posterior_backward_list = []
    posterior_scores_list_ana = []
    means_posterior_backward_list_ana = []
    sigma_posterior_backward_list_ana = []
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
                posterior_scores,
                means_posterior_backward,
                sigma_posterior_backward,
                posterior_scores_ana,
                means_posterior_backward_ana,
                sigma_posterior_backward_ana,
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
        theta_list.append(theta_t.detach().cpu())
        if debug:
            gradlogL_list.append(gradlogL)
            posterior_scores_list.append(posterior_scores)
            means_posterior_backward_list.append(means_posterior_backward)
            sigma_posterior_backward_list.append(sigma_posterior_backward)
            posterior_scores_list_ana.append(posterior_scores_ana)
            means_posterior_backward_list_ana.append(means_posterior_backward_ana)
            sigma_posterior_backward_list_ana.append(sigma_posterior_backward_ana)
    if debug:
        return (
            theta_t,
            theta_list,
            gradlogL_list,
            posterior_scores_list,
            means_posterior_backward_list,
            sigma_posterior_backward_list,
            posterior_scores_list_ana,
            means_posterior_backward_list_ana,
            sigma_posterior_backward_list_ana,
        )
    else:
        return theta_t, theta_list


if __name__ == "__main__":
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from tasks.toy_examples.prior import GaussianPrior
    from nse import NSE, NSELoss
    from sm_utils import train

    from tqdm import tqdm

    torch.manual_seed(1)

    N_TRAIN = 10_000
    N_SAMPLES = 4096
    N_OBS = 2

    # Task
    task = SBIGaussian2d(prior_type="gaussian")
    # Prior and Simulator
    prior = task.prior
    simulator = task.simulator

    # Observation
    theta_true = torch.FloatTensor([-5, 150])  # true parameters
    x_obs = simulator(theta_true)  # x_0 ~ simulator(theta_true)

    # True posterior: p(theta|x_0)
    true_posterior = task.true_posterior(x_obs)

    # True posterior p(theta|mean(x_1,...,x_10)), x_i ~ simulator(theta_true)
    x_obs_100 = torch.cat(
        [simulator(theta_true).reshape(1, -1) for i in range(100)], dim=0
    )
    true_posterior = task.true_posterior(torch.mean(x_obs_100[:N_OBS], axis=0))

    # Train data
    theta_train = task.prior.sample((N_TRAIN,))
    x_train = simulator(theta_train)

    # normalize theta
    theta_train_ = (theta_train - theta_train.mean(axis=0)) / theta_train.std(axis=0)

    # normalize x
    x_train_ = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_obs_ = (x_obs - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_obs_100_ = (x_obs_100 - x_train.mean(axis=0)) / x_train.std(axis=0)

    # score_network
    #dataset = torch.utils.data.TensorDataset(theta_train_.cuda(), x_train_.cuda())
    #score_net = NSE(theta_dim=2, x_dim=2, hidden_features=[128, 256, 128]).cuda()

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
    score_net = torch.load("score_net.pkl").cuda()

    loc_ = (prior.prior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    cov_ = (
            torch.diag(1 / theta_train.std(axis=0))
            @ prior.prior.covariance_matrix
            @ torch.diag(1 / theta_train.std(axis=0))
    )
    prior_ = torch.distributions.MultivariateNormal(loc=loc_.cuda(),
                                                    covariance_matrix=cov_.cuda())

    from functools import partial
    # task.true_posterior = torch.distributions.MultivariateNormal(loc=task.true_posterior.loc.cuda(),
    #                                                             covariance_matrix=task.true_posterior.covariance_matrix.cuda())
    score_fn = partial(
        diffused_tall_posterior_score,
        prior=prior_,
        posterior_fn_ana=task.true_posterior,
        x_obs=x_obs_100[:N_OBS].cuda(),
        x_obs_=x_obs_100_[:N_OBS].cuda(),
        nse=score_net,
        theta_mean=theta_train.mean(axis=0),
        theta_std=theta_train.std(axis=0),
    )

    # (
    # theta_learned,
    # theta_list_learned,
    # ) = euler_sde_sampler(
    #     score_fn_1, N_SAMPLES, beta=score_net.beta, device="cuda:0",
    # )
    # (
    #     theta_ana,
    #     theta_list_ana,
    # ) = euler_sde_sampler(
    #     score_fn_2, N_SAMPLES, beta=score_net.beta, device="cpu", 
    # )


    (
        theta_learned,
        theta_list_learned,
        gradlogL_list,
        posterior_scores_list,
        means_posterior_backward_list,
        sigma_posterior_backward_list,
        posterior_scores_list_ana,
        means_posterior_backward_list_ana,
        sigma_posterior_backward_list_ana,
    ) = euler_sde_sampler(
        score_fn, N_SAMPLES, beta=score_net.beta, device="cuda:0", debug=True
    )
    posterior_scores = torch.stack(posterior_scores_list)
    posterior_scores_ana = torch.stack(posterior_scores_list_ana)

    posterior_scores_diff = torch.linalg.norm(posterior_scores_ana - posterior_scores, dim=-1)

    theta_learned = theta_learned.detach().cpu()
    theta_list_learned[0] = theta_list_learned[0].cpu().detach()
    all_theta = torch.stack(theta_list_learned)
    # # unnormalize samples
    theta_learned = theta_learned * theta_train.std(axis=0) + theta_train.mean(axis=0)
    s = posterior_scores_diff.mean(dim=0)
    #s = posterior_scores_diff.max(dim=0).values
    s = s - s.min()
    plt.scatter(*theta_learned.T, label='DDGM', s=s.clip(0, 100))
    plt.scatter(*true_posterior.sample((N_SAMPLES,)).T, label='True', alpha=.1)
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.ylim(140, 170)
    plt.xlim(10, -10)
    plt.legend()
    plt.show()

    #Erreur des score en fonction de la distance a theta true
    plt.scatter(posterior_scores_diff.max(dim=0).values, torch.linalg.norm(theta_learned - theta_true[None,], axis=-1))
    #plt.yscale('log')
    plt.ylabel('Distance to true theta')
    plt.xlabel('Max score error through sampling')
    plt.show()

    plt.scatter(posterior_scores_diff.mean(dim=0), torch.linalg.norm(theta_learned - theta_true[None,], axis=-1))
    #plt.yscale('log')
    plt.ylabel('Distance to true theta')
    plt.xlabel('Mean score error through sampling')
    plt.show()

    #Erreur de score a la premiere iteration
    plt.scatter(posterior_scores_diff[0], torch.linalg.norm(theta_list_learned[0].detach().cpu(), dim=-1))
    plt.show()

    #Tout erreur vs tout theta
    plt.scatter(torch.linalg.norm(all_theta[:-1], dim=-1).flatten(),
                posterior_scores_diff.flatten())
    plt.show()

    # theta_list_learned = [
    #     theta * theta_train.std(axis=0) + theta_train.mean(axis=0)
    #     for theta in theta_list_learned
    # ]
    #
    # diff_theta_list = [
    #     torch.linalg.norm((theta_list_learned[i] - theta_list_ana[i]), dim=-1).mean()
    #     for i in range(len(theta_list_learned))
    # ]
    #t = torch.linspace(1, 0, len(diff_theta_list) + 1)[:-1]
    #
    # plt.plot(t, diff_theta_list, label="diff theta", marker="o", alpha=0.5, linewidth=3)
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig(f"diff_theta_n_obs_{N_OBS}.png")
    # plt.clf()

    # plt.scatter(
    #     theta_learned[:, 0], theta_learned[:, 1], label="learned", marker="o", alpha=0.5
    # )
    # plt.scatter(
    #     theta_ana[:, 0], theta_ana[:, 1], label="analytical", marker="o", alpha=0.5
    # )
    # plt.xlabel("theta_1")
    # plt.ylabel("theta_2")
    # plt.legend()
    # plt.savefig(f"samples_n_obs_{N_OBS}.png")
    # plt.clf()

    #
    # diff_posterior_scores_list = [
    #     torch.linalg.norm((posterior_scores_list[i] - posterior_scores_list_ana[i]), dim=-1).mean()
    #     for i in range(len(posterior_scores_list))
    # ]
    # plt.plot(
    #     t,
    #     diff_posterior_scores_list,
    #     label="diff posterior scores",
    #     marker="o",
    #     alpha=0.5,
    #     linewidth=3,
    # )
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig(f"diff_posterior_scores_n_obs_{N_OBS}.png")
    # plt.clf()
    #
    # for i in range(N_OBS):
    #     diff_mean_posterior_backward_list = [
    #         torch.linalg.norm((m[:,i,:] - m_ana[:,i,:]), dim=-1).mean() for m, m_ana in zip(means_posterior_backward_list, means_posterior_backward_list_ana)
    #     ]
    #     plt.plot(
    #         t,
    #         diff_mean_posterior_backward_list,
    #         label=f"diff posterior means {i}",
    #         marker="o",
    #         alpha=0.5,
    #         linewidth=3,
    #     )
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig(f"diff_posterior_means_backward_n_obs_{N_OBS}.png")
    # plt.clf()
    #
    # for i in range(N_OBS):
    #     diff_sigma_posterior_backward_list = [
    #         torch.linalg.norm((s[:,i,:,:] - s_ana[:,i,:,:]), dim=(-2,-1)).mean() for s, s_ana in zip(sigma_posterior_backward_list, sigma_posterior_backward_list_ana)
    #     ]
    #     plt.plot(
    #         t,
    #         diff_sigma_posterior_backward_list,
    #         label=f"diff posterior sigma {i}",
    #         marker="o",
    #         alpha=0.5,
    #         linewidth=3,
    #     )
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig(f"diff_posterior_sigma_backward_n_obs_{N_OBS}.png")
    # plt.clf()
    #
    #
