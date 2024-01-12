import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score


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
            return lda, eta, -.5 * (-torch.linalg.slogdet(lda).logabsdet + (mean[..., None].mT @ lda @ mean[..., None])[...,0, 0])

        n_observations = means_posterior_backward.shape[-2]
        lambdas_posterior, etas_posterior, zetas_posterior = from_canonical_to_sufficient(means_posterior_backward, covar_posteriors_backward)
        lda_prior, eta_prior, zeta_prior = from_canonical_to_sufficient(mean_prior_backward, covar_prior_backward)

        sum_zetas = zetas_posterior.sum(axis=1) + (1 - n_observations)*zeta_prior

        final_gaussian_etas = (1 - n_observations)*eta_prior + etas_posterior.sum(axis=1)
        final_gaussian_ldas = (1 - n_observations)*lda_prior + lambdas_posterior.sum(axis=1)
        final_gaussian_zeta = -.5 * (-torch.linalg.slogdet(final_gaussian_ldas).logabsdet
                                     + (final_gaussian_etas[..., None].mT @ torch.linalg.inv(final_gaussian_ldas) @final_gaussian_etas[..., None])[..., 0, 0])
        return sum_zetas - final_gaussian_zeta

def mean_backward(theta, t, score_fn, nse, **kwargs):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    return 1 / (alpha_t**.5) * (theta + sigma_t**2 * score_fn(theta=theta, t=t, **kwargs))

def sigma_backward(t, dist_cov, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    return (sigma_t**2) / alpha_t * (torch.eye(2).to(alpha_t.device) + sigma_t**2 * (-1) * torch.linalg.inv((nse.alpha(t) * dist_cov.to(alpha_t.device) + sigma_t**2 * torch.eye(2).to(alpha_t.device))))

def eta_lda(means_posterior_backward,
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

    return final_gaussian_etas, final_gaussian_ldas

def sigma_backward_autodiff(theta, x, t, score_fn, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    def mean_to_jac(theta, x):
        score = score_fn(theta=theta, t=t, x=x)
        mu = mean_backward(theta, t, score_fn, nse, x=x)
        return mu, (mu, score)
    
    grad_mean, _ = vmap(jacrev(mean_to_jac, has_aux=True))(theta, x)
    return (sigma_t**2 / (alpha_t ** .5))*grad_mean

def comparison(theta, t, x_obs, prior, prior_score_fn, post_score_fn_1, post_score_fn_2, theta_mean, theta_std, x_mean, x_std, nse):
    def compute_results(prior, prior_score_fn, posterior_score_fn):
        n_obs = len(x_obs)
        # input normalization for the score network
        x_obs_ = (x_obs - x_mean) / x_std

        theta.requires_grad = True
        theta.grad = None

        means_posterior_backward = []
        posterior_scores = []
        for i in range(n_obs):
            if posterior_score_fn is None:
                posterior = task.true_posterior(x_obs[i])
                # for comparison purposes we rescale the mean and covariance of the posterior
                posterior_score = get_vpdiff_gaussian_score(posterior.loc - theta_mean, posterior.covariance_matrix / theta_std**2, nse)
                sigma_posterior_backward = sigma_backward(t, posterior.covariance_matrix / theta_std**2, nse).repeat(theta.shape[0], n_obs, 1,1)
                kwargs = {}

            else:
                posterior_score = posterior_score_fn
                x = x_obs_[i].to(theta.device).repeat(theta.shape[0], 1)
                kwargs = {"x": x_obs_[i].to(theta.device)}
                sigma_posterior_backward = sigma_backward_autodiff(theta, x, t, posterior_score, nse).repeat(n_obs, 1,1,1).permute(1,0,2,3)
            posterior_scores.append(posterior_score(theta=theta,t=t, **kwargs))
            means_posterior_backward.append(mean_backward(theta, t, posterior_score, nse, **kwargs))
        
        means_posterior_backward = torch.stack(means_posterior_backward).permute(1,0,2)

        mean_prior_backward = mean_backward(theta, t, prior_score_fn, nse)
        sigma_prior_backward = sigma_backward(t, prior.prior.covariance_matrix, nse).repeat(theta.shape[0], 1,1)

        return torch.stack(posterior_scores).sum(axis=0), means_posterior_backward, sigma_posterior_backward, mean_prior_backward, sigma_prior_backward

    posterior_scores_1, means_posterior_backward_1, sigma_posterior_backward_1, mean_prior_backward_1, sigma_prior_backward_1 = compute_results(prior, prior_score_fn, post_score_fn_1)
    posterior_scores_2, means_posterior_backward_2, sigma_posterior_backward_2, mean_prior_backward_2, sigma_prior_backward_2 = compute_results(prior, prior_score_fn, post_score_fn_2)

    diff_posterior_scores = (posterior_scores_1 - posterior_scores_2).square().mean()
    diff_posterior_means_backward = (means_posterior_backward_1 - means_posterior_backward_2).square().mean()
    diff_posterior_sigma_backward = (sigma_posterior_backward_1 - sigma_posterior_backward_2).square().mean()
    diff_prior_means_backward = (mean_prior_backward_1 - mean_prior_backward_2).square().mean()
    diff_prior_sigma_backward = (sigma_prior_backward_1 - sigma_prior_backward_2).square().mean()

    return diff_posterior_scores, diff_posterior_means_backward, diff_posterior_sigma_backward, diff_prior_means_backward, diff_prior_sigma_backward


if __name__ == '__main__':
    
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from tasks.toy_examples.prior import GaussianPrior
    from nse import NSE, NSELoss
    from sm_utils import train

    from tqdm import tqdm

    torch.manual_seed(1)


    N_TRAIN = 10_000
    N_SAMPLES = 4096

    # Task
    task = SBIGaussian2d(prior_type="gaussian")
    # Prior and Simulator
    prior = task.prior
    simulator = task.simulator

    # Observation
    theta_true = torch.FloatTensor([-5, 150])  # true parameters
    x_obs = simulator(theta_true)  # x_0 ~ simulator(theta_true)
    observation = {"theta_true": theta_true, "x_obs": x_obs}

    # True posterior: p(theta|x_0)
    true_posterior = task.true_posterior(x_obs)

    # True posterior p(theta|mean(x_1,...,x_10)), x_i ~ simulator(theta_true)
    x_obs_100 = torch.cat([simulator(theta_true).reshape(1, -1) for i in range(100)], dim=0)
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
    dataset = torch.utils.data.TensorDataset(theta_train_, x_train_)
    score_net = NSE(theta_dim=2, x_dim=2, hidden_features=[128, 256, 128])

    avg_score_net = train(
        model=score_net,
        dataset=dataset,
        loss_fn=NSELoss(score_net),
        n_epochs=200,
        lr=1e-3,
        batch_size=256,
        prior_score=False, # learn the prior score via the classifier-free guidance approach
    )
    score_net = avg_score_net.module

    loc_ = prior.prior.loc - theta_train.mean(axis=0)
    cov_ = prior.prior.covariance_matrix / theta_train.std(axis=0) ** 2

    means = {
        0: loc_[0],
        1: loc_[1],
    }

    stds = {
        0: cov_[0][0],
        1: cov_[1][1],
    }

    prior_rescaled = GaussianPrior(means=means, stds=stds)

    prior_score_fn = get_vpdiff_gaussian_score(loc_, cov_, nse=score_net)

    theta = torch.randn((N_SAMPLES, 2))
    t = torch.linspace(1, 0, 10)

    diff_posterior_scores_list  = []
    diff_posterior_means_backward_list = []
    diff_posterior_sigma_backward_list = []

    for t_ in tqdm(t):
        diff_posterior_scores, diff_posterior_means_backward, diff_posterior_sigma_backward, _, _ = comparison(theta, t_, x_obs_100, prior_rescaled, prior_score_fn, score_net.score, None, theta_train.mean(axis=0), theta_train.std(axis=0), x_train.mean(axis=0), x_train.std(axis=0), score_net)
        diff_posterior_scores_list.append(diff_posterior_scores.detach().numpy())
        diff_posterior_means_backward_list.append(diff_posterior_means_backward.detach().numpy())
        diff_posterior_sigma_backward_list.append(diff_posterior_sigma_backward.detach().numpy())


    plt.plot(t, diff_posterior_scores_list, label="diff posterior scores", marker="o", alpha=0.5, linewidth=3)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("diff_posterior_scores.png")
    plt.clf()

    plt.plot(t, diff_posterior_means_backward_list, label="diff posterior means backward", marker="o", alpha=0.5, linewidth=3)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("diff_posterior_means_backward.png")
    plt.clf()
    
    plt.plot(t, diff_posterior_sigma_backward_list, label="diff posterior sigma backward", marker="o", alpha=0.5, linewidth=3)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("diff_posterior_sigma_backward.png")
    plt.clf()