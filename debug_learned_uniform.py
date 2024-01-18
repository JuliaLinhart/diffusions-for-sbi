import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_uniform_score
from nse import assure_positive_definitness
from tqdm import tqdm


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
        prior_score_fn,
        x_obs,
        nse,
        debug=False,
):
    n_obs = x_obs.shape[0]

    theta.requires_grad = True
    theta.grad = None

    sigma_0_t, mean_0_t, _ = tweedies_approximation(x=x_obs,
                                                         theta=theta,
                                                         nse=nse,
                                                         t=t,
                                                         score_fn=nse.score)

    scores = nse.score(theta[:, None], x_obs[None], t=t)
    sigma_prior_0_t, mean_prior_0_t, _ = tweedies_approximation_prior(theta=theta,
                                                                        t=t,
                                                                        score_fn=prior_score_fn,
                                                                        nse=nse)
    prior_score = prior_score_fn(theta, t)

    #sigma_prior_0_t = assure_positive_definitness(sigma_prior_0_t.detach())

    total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)

    # print(sigma_0_t.shape, mean_0_t.shape, sigma_prior_0_t.shape, mean_prior_0_t.shape)


    # This works but it's strange!
    #smallest_eig = torch.linalg.eigvals(lda).real.min(dim=-1).values
    if (nse.alpha(t)**.5 > .5) and (n_obs > 1): #(nse.sigma(t) / (nse.alpha(t)**.5)) < .05: #smallest_eig > 0).all():
        sigma_prior_eigvals = torch.linalg.eigvals(sigma_prior_0_t).real[0]
        lim_sup_sigma = (n_obs / (n_obs - 1)) * sigma_prior_eigvals.min().item()
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


def euler_sde_sampler(score_fn, nsamples, beta, dim_theta, device="cpu", ana=False, debug=False):
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



if __name__ == "__main__":
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from nse import NSE, NSELoss
    from sm_utils import train

    from functools import partial

    torch.manual_seed(1)

    N_TRAIN = 10_000
    N_SAMPLES = 4096
    N_OBS = 100

    # Task
    task = SBIGaussian2d(prior_type="uniform")
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
    # torch.save(score_net, "score_net_uniform.pkl")

    # load score network
    score_net = torch.load("score_net_uniform.pkl").cuda()

    # normalize prior
    low_ = (prior.low - theta_train.mean(axis=0)) / theta_train.std(axis=0) * 2
    high_ = (prior.high - theta_train.mean(axis=0)) / theta_train.std(axis=0) * 2
    
    prior_score_fn_ = get_vpdiff_uniform_score(low_.cuda(), high_.cuda(), score_net)


    # score function for tall posterior (learned and analytical)
    score_fn = partial(
        diffused_tall_posterior_score,
        prior_score_fn=prior_score_fn_, # normalized prior# analytical posterior
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
        score_fn, N_SAMPLES, dim_theta=2, beta=score_net.beta, device="cuda:0", debug=True
    )
    

    results_dict = {
        "all_theta_learned": all_theta_learned,
        "gradlogL": gradlogL,
        "lda": lda,
        "posterior_scores": posterior_scores,
        "means_posterior_backward": means_posterior_backward,
        "sigma_posterior_backward": sigma_posterior_backward,
    }
    torch.save(results_dict, f"results/uniform/results_dict_n_obs_{N_OBS}_lda_criteria.pkl")


    theta_learned = theta_learned.detach().cpu()
    all_theta_learned = all_theta_learned * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]
    # unnormalize sample s
    theta_learned = theta_learned * theta_train.std(axis=0) + theta_train.mean(axis=0)
    plt.scatter(*true_posterior.sample((N_SAMPLES,)).T, label="True (n=1)", alpha=0.1)
    plt.scatter(*all_theta_learned[-1].T, label=f"DDGM (n={N_OBS})",)#=s.clip(0, 100))
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.ylim(140, 170)
    plt.xlim(10, -10)
    plt.legend()
    plt.savefig(f"results/uniform/samples_n_obs_{N_OBS}.png")
    plt.clf()
    # plt.show()

