import matplotlib.pyplot as plt
import torch

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


def eta_lda(
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

    final_gaussian_etas = (1 - n_observations) * eta_prior + etas_posterior.sum(axis=1)
    final_gaussian_ldas = (1 - n_observations) * lda_prior + lambdas_posterior.sum(
        axis=1
    )

    return final_gaussian_etas, final_gaussian_ldas


def sigma_backward_autodiff(theta, x, t, score_fn, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    def mean_to_jac(theta, x):
        score = score_fn(theta=theta, t=t, x=x)
        mu = mean_backward(theta, t, score_fn, nse, x=x)
        return mu, (mu, score)

    grad_mean, _ = vmap(jacrev(mean_to_jac, has_aux=True))(theta, x)
    return (sigma_t**2 / (alpha_t**0.5)) * grad_mean


def comparison(theta, t, x_obs, task, theta_mean, theta_std, x_mean, x_std, nse):
    def compute_results(posterior_fn):
        n_obs = len(x_obs)
        # input normalization for the score network
        x_obs_ = (x_obs - x_mean) / x_std

        theta.requires_grad = True
        theta.grad = None

        means_posterior_backward = []
        sigmas_posterior_backward = []
        posterior_scores = []
        for i in range(n_obs):
            if posterior_fn is not None:
                posterior = posterior_fn(x_obs[i])
                # for comparison purposes we rescale the mean and covariance of the posterior
                loc = (posterior.loc - theta_mean) / theta_std
                cov = (
                    torch.diag(1 / theta_std)
                    @ posterior.covariance_matrix
                    @ torch.diag(1 / theta_std)
                )
                posterior_score = get_vpdiff_gaussian_score(loc, cov, nse)
                sigmas_posterior_backward.append(
                    sigma_backward(t, cov, nse).repeat(theta.shape[0], 1, 1)
                )
                kwargs = {}

            else:
                posterior_score = nse.score_net
                x = x_obs_[i].to(theta.device).repeat(theta.shape[0], 1)
                kwargs = {"x": x_obs_[i].to(theta.device)}
                sigmas_posterior_backward.append(
                    sigma_backward_autodiff(theta, x, t, posterior_score, nse)
                )
            posterior_scores.append(posterior_score(theta=theta, t=t, **kwargs))
            means_posterior_backward.append(
                mean_backward(theta, t, posterior_score, nse, **kwargs)
            )

        means_posterior_backward = torch.stack(means_posterior_backward).permute(
            1, 0, 2
        )
        sigma_posterior_backward = torch.stack(sigmas_posterior_backward).permute(
            1, 0, 2, 3
        )

        return (
            torch.stack(posterior_scores).sum(axis=0),
            means_posterior_backward,
            sigma_posterior_backward,
        )

    (
        posterior_scores_1,
        means_posterior_backward_1,
        sigma_posterior_backward_1,
    ) = compute_results(posterior_fn=None)
    (
        posterior_scores_2,
        means_posterior_backward_2,
        sigma_posterior_backward_2,
    ) = compute_results(posterior_fn=task.true_posterior)

    diff_posterior_scores = (posterior_scores_1 - posterior_scores_2).square().mean()
    diff_posterior_means_backward = (
        (means_posterior_backward_1 - means_posterior_backward_2).square().mean()
    )
    diff_posterior_sigma_backward = (
        (sigma_posterior_backward_1 - sigma_posterior_backward_2).square().mean()
    )

    return (
        diff_posterior_scores,
        diff_posterior_means_backward,
        diff_posterior_sigma_backward,
    )


def diffused_tall_posterior_score(
    theta, t, prior, posterior_fn, x_obs, nse, theta_mean=None, theta_std=None
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
    for i in range(n_obs):
        if posterior_fn is not None:
            posterior = posterior_fn(x_obs[i])
            # rescale the mean and covariance of the posterior
            if theta_mean is not None and theta_std is not None:
                loc = (posterior.loc - theta_mean) / theta_std
                cov = (
                    torch.diag(1 / theta_std)
                    @ posterior.covariance_matrix
                    @ torch.diag(1 / theta_std)
                )
                posterior = torch.distributions.MultivariateNormal(
                    loc=loc, covariance_matrix=cov
                )
            posterior_score_fn = get_vpdiff_gaussian_score(
                posterior.loc, posterior.covariance_matrix, nse
            )
            sigmas_posterior_backward.append(
                sigma_backward(t, posterior.covariance_matrix, nse).repeat(
                    theta.shape[0], 1, 1
                )
            )
            kwargs = {}

        else:
            posterior_score_fn = nse.score
            x = x_obs[i].to(theta.device).repeat(theta.shape[0], 1)
            kwargs = {"x": x_obs[i].to(theta.device)}
            sigmas_posterior_backward.append(
                sigma_backward_autodiff(theta, x, t, posterior_score_fn, nse)
            )

        posterior_scores.append(posterior_score_fn(theta=theta, t=t, **kwargs))
        means_posterior_backward.append(
            mean_backward(theta, t, posterior_score_fn, nse, **kwargs)
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
    prior_score = prior_score_fn(theta, t)

    return (1 - n_obs) * prior_score + posterior_scores + gradlogL


def euler_sde_sampler(score_fn, nsamples, beta, device="cpu"):
    theta_t = torch.randn((nsamples, 2)).to(device)  # (nsamples, 2)
    time_pts = torch.linspace(1, 0, 1000).to(device)  # (ntime_pts,)
    theta_list = []
    for i in tqdm(range(len(time_pts) - 1)):
        t = time_pts[i]
        dt = time_pts[i + 1] - t

        # calculate the drift and diffusion terms
        f = -0.5 * beta(t) * theta_t
        g = beta(t) ** 0.5
        score = score_fn(theta_t, t).detach()
        drift = f - g * g * score
        diffusion = g

        # euler-maruyama step
        theta_t = (
            theta_t.detach()
            + drift * dt
            + diffusion * torch.randn_like(theta_t) * torch.abs(dt) ** 0.5
        )
        theta_list.append(theta_t.detach().cpu())
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
    N_OBS = 5

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
    true_posterior_100 = task.true_posterior(torch.mean(x_obs_100, axis=0))

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
    dataset = torch.utils.data.TensorDataset(theta_train_.cuda(), x_train_.cuda())
    score_net = NSE(theta_dim=2, x_dim=2, hidden_features=[128, 256, 128]).cuda()

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
    score_net = torch.load("score_net.pkl")

    loc_ = (prior.prior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    cov_ = (
        torch.diag(1 / theta_train.std(axis=0))
        @ prior.prior.covariance_matrix
        @ torch.diag(1 / theta_train.std(axis=0))
    )
    prior_ = torch.distributions.MultivariateNormal(loc=loc_, covariance_matrix=cov_)

    from functools import partial

    score_fn_1 = partial(
        diffused_tall_posterior_score,
        prior=prior_,
        posterior_fn=None,
        x_obs=x_obs_100_[:N_OBS],
        nse=score_net,
    )
    score_fn_2 = partial(
        diffused_tall_posterior_score,
        prior=prior.prior,
        posterior_fn=task.true_posterior,
        x_obs=x_obs_100[:N_OBS],
        nse=score_net,
    )
    # score_fn_2 = partial(diffused_tall_posterior_score, prior=prior_, posterior_fn=task.true_posterior, x_obs=x_obs_100[:N_OBS], nse=score_net, theta_mean=theta_train.mean(axis=0), theta_std=theta_train.std(axis=0)

    theta_learned, theta_list_learned = euler_sde_sampler(
        score_fn_1, N_SAMPLES, beta=score_net.beta, device="cuda:0"
    )
    theta_ana, theta_list_ana = euler_sde_sampler(
        score_fn_2, N_SAMPLES, beta=score_net.beta, device="cpu"
    )

    theta_learned = theta_learned.detach().cpu()

    # unnormalize samples
    theta_learned = theta_learned * theta_train.std(axis=0) + theta_train.mean(axis=0)
    theta_list_learned = [
        theta * theta_train.std(axis=0) + theta_train.mean(axis=0)
        for theta in theta_list_learned
    ]

    diff_theta_list = [
        (theta_list_learned[i] - theta_list_ana[i]).square().mean()
        for i in range(len(theta_list_learned))
    ]
    t = torch.linspace(1, 0, len(diff_theta_list) + 1)[:-1]

    plt.plot(t, diff_theta_list, label="diff theta", marker="o", alpha=0.5, linewidth=3)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(f"diff_theta_n_obs_{N_OBS}.png")
    plt.clf()

    plt.scatter(
        theta_learned[:, 0], theta_learned[:, 1], label="learned", marker="o", alpha=0.5
    )
    plt.scatter(
        theta_ana[:, 0], theta_ana[:, 1], label="analytical", marker="o", alpha=0.5
    )
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.legend()
    plt.savefig(f"samples_n_obs_{N_OBS}.png")
    plt.clf()

    # theta = torch.randn((N_SAMPLES, 2))
    # t = torch.linspace(1, 0, 20)[:-1]

    # diff_posterior_scores_list  = []
    # diff_posterior_means_backward_list = []
    # diff_posterior_sigma_backward_list = []

    # for t_ in tqdm(t):
    #     diff_posterior_scores, diff_posterior_means_backward, diff_posterior_sigma_backward = comparison(theta, t_, x_obs_100[:N_OBS], task, theta_train.mean(axis=0), theta_train.std(axis=0), x_train.mean(axis=0), x_train.std(axis=0), score_net)
    #     diff_posterior_scores_list.append(diff_posterior_scores.detach().numpy())
    #     diff_posterior_means_backward_list.append(diff_posterior_means_backward.detach().numpy())
    #     diff_posterior_sigma_backward_list.append(diff_posterior_sigma_backward.detach().numpy())

    # plt.plot(t, diff_posterior_scores_list, label="diff posterior scores", marker="o", alpha=0.5, linewidth=3)
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig("diff_posterior_scores.png")
    # plt.clf()

    # plt.plot(t, diff_posterior_means_backward_list, label="diff posterior means backward", marker="o", alpha=0.5, linewidth=3)
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig("diff_posterior_means_backward.png")
    # plt.clf()

    # plt.plot(t, diff_posterior_sigma_backward_list, label="diff posterior sigma backward", marker="o", alpha=0.5, linewidth=3)
    # plt.xlabel("t")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.savefig("diff_posterior_sigma_backward.png")
    # plt.clf()
