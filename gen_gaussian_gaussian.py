import os

import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score
from tqdm import tqdm
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
import time


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
        score = score_fn(theta[:, None], x[None], t[None])
        # score = vmap(lambda theta: vmap(partial(score_fn, t=t),
        #                                 in_dims=(None, 0),
        #                                 randomness='different')(theta, x),
        #              randomness='different')(theta)
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


def euler_sde_sampler(score_fn, nsamples, dim, beta, device="cpu", theta_clipping_range=(None, None)):
    theta_t = torch.randn((nsamples, dim)).to(device)  # (nsamples, 2)
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

class EpsilonNet(torch.nn.Module):

    def __init__(self, DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2*DIM + 1, 5*DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(5*DIM, DIM),
            torch.nn.Tanh()
        )

    def forward(self, theta, x, t):
        return self.net(torch.cat((theta, x, t), dim=-1))

class FakeFNet(torch.nn.Module):

    def __init__(self, real_eps_fun, eps_net, eps_net_max):
        super().__init__()
        self.real_eps_fun = real_eps_fun
        self.eps_net = eps_net
        self.eps_net_max = eps_net_max

    def forward(self, theta, x, t):
        if len(t.shape) == 0:
            t = t[None, None].repeat(theta.shape[0], 1)
        real_eps = self.real_eps_fun(theta, x, t)
        perturb = self.eps_net(theta, x, t)
        return real_eps + self.eps_net_max * perturb


if __name__ == "__main__":
    import sys
    from nse import NSE

    path_to_save = sys.argv[1]

    torch.manual_seed(1)
    N_TRAIN = 10_000
    N_SAMPLES = 4096
    all_exps = []
    for DIM in [2, 4, 8, 10, 16, 32, 64]:
        for eps in [0, 1e-3, 1e-2, 1e-1]:
            for seed in tqdm(range(5), desc=f'Dim {DIM} eps {eps}'):

                # Observations
                torch.manual_seed(seed)
                means = torch.rand(DIM) * 20 - 10  # between -10 and 10
                stds = torch.rand(DIM) * 25 + 0.1  # between 0.1 and 25.1
                task = Gaussian_Gaussian_mD(dim=DIM, means=means, stds=stds)
                prior = task.prior
                simulator = task.simulator
                theta_true = prior.sample(sample_shape=(1,))  # true parameters

                x_obs = simulator(theta_true)  # x_obs ~ simulator(theta_true)
                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
                )

                # True posterior: p(theta|x_obs)
                true_posterior = task.true_posterior(x_obs)

                # Train data
                theta_train = task.prior.sample((N_TRAIN,))
                x_train = simulator(theta_train)

                score_net = NSE(theta_dim=DIM, x_dim=DIM)

                t = torch.linspace(0, 1, 1000).cuda()
                idm = torch.eye(DIM).cuda()
                inv_lik = torch.linalg.inv(task.simulator_cov).cuda()
                inv_prior = torch.linalg.inv(prior.covariance_matrix).cuda()
                posterior_cov = torch.linalg.inv(inv_lik + inv_prior)

                inv_prior_prior = inv_prior @ prior.loc.cuda()
                def real_eps(theta, x, t):
                    posterior_cov_diff = (posterior_cov[None] * score_net.alpha(t)[..., None] + (1 - score_net.alpha(t))[..., None] * idm[None])
                    posterior_mean_0 = (posterior_cov @ (inv_prior_prior[:, None] + inv_lik @ x.mT)).mT
                    posterior_mean_diff = (score_net.alpha(t)**.5) * posterior_mean_0
                    score = - (torch.linalg.inv(posterior_cov_diff) @ (theta - posterior_mean_diff)[..., None])[..., 0]
                    return - score_net.sigma(t) * score

                score_net.net = FakeFNet(real_eps_fun=real_eps,
                                         eps_net=EpsilonNet(DIM),
                                         eps_net_max=eps)
                score_net.cuda()

                t = torch.linspace(0, 1, 1000)
                prior_score_fn = get_vpdiff_gaussian_score(prior.loc.cuda(), prior.covariance_matrix.cuda(), score_net)

                def prior_score(theta, t):
                    mean_prior_0_t = mean_backward(theta, t, prior_score_fn, score_net)
                    prec_prior_0_t = prec_matrix_backward(t, prior.covariance_matrix.cuda(), score_net).repeat(
                        theta.shape[0], 1, 1
                    )
                    prior_score = prior_score_fn(theta, t)
                    return prior_score, mean_prior_0_t, prec_prior_0_t,


                for N_OBS in [2, 4, 8, 16, 32, 64, 90]:
                    true_posterior_cov = torch.linalg.inv(inv_lik*N_OBS + inv_prior)
                    true_posterior_mean = (true_posterior_cov @ (inv_prior_prior + inv_lik @ x_obs_100[:N_OBS].sum(dim=0).cuda()))
                    infos = {"true_posterior_mean": true_posterior_mean,
                             "true_posterior_cov": true_posterior_cov,
                             "true_theta": theta_true,
                             "N_OBS": N_OBS,
                             "seed": seed,
                             "dim": DIM,
                             "eps": eps,
                             "exps":{"Langevin": [],
                                     "GAUSS": [],
                                     "JAC": []}
                             }
                    for sampling_steps, eta in zip([50, 150, 400, 1000], [.2, .5, .8, 1]):
                        tstart_gauss = time.time()
                        samples_ddim = score_net.ddim(shape=(1000*N_OBS,),
                                                      x=x_obs_100[None, :N_OBS].repeat(1000, 1, 1).reshape(1000*N_OBS, -1).cuda(),
                                                      steps=100,
                                                      eta=.5).detach().reshape(1000, N_OBS, -1).cpu()
                        #normalize posterior
                        cov_est = vmap(lambda x:torch.cov(x.mT))(samples_ddim.permute(1, 0, 2))

                        samples_gauss = score_net.ddim(shape=(1000,),
                                                       x=x_obs_100[:N_OBS].cuda(),
                                                       eta=eta,
                                                       steps=sampling_steps,
                                                       prior_score_fn=prior_score,
                                                       dist_cov_est=cov_est.cuda(),
                                                       cov_mode='GAUSS').cpu()
                        tstart_jac = time.time()
                        samples_jac = score_net.ddim(shape=(1000,),
                                                     x=x_obs_100[:N_OBS].cuda(),
                                                     eta=eta,
                                                     steps=sampling_steps,
                                                     prior_score_fn=prior_score,
                                                     cov_mode='JAC').cpu()
                        tstart_lang = time.time()
                        with torch.no_grad():
                            lang_samples = score_net.annealed_langevin_geffner(shape=(1000,),
                                                                               x=x_obs_100[:N_OBS].cuda(),
                                                                               prior_score_fn=prior_score,
                                                                               lsteps=5,
                                                                               steps=sampling_steps)
                        t_end_lang = time.time()
                        dt_gauss = tstart_jac - tstart_gauss
                        dt_jac = tstart_lang - tstart_jac
                        dt_lang = t_end_lang - tstart_lang
                        true_posterior_cov = torch.linalg.inv(inv_lik*N_OBS + inv_prior)
                        true_posterior_mean = (true_posterior_cov @ (inv_prior_prior + inv_lik @ x_obs_100[:N_OBS].sum(dim=0).cuda()))
                        ref_samples = torch.distributions.MultivariateNormal(loc=true_posterior_mean, covariance_matrix=true_posterior_cov).sample((1000,)).cpu()
                        infos["exps"]["Langevin"].append({"dt": dt_lang, "samples": lang_samples, "n_steps": sampling_steps})
                        infos["exps"]["GAUSS"].append({"dt": dt_gauss, "samples": samples_gauss, "n_steps": sampling_steps})
                        infos["exps"]["JAC"].append({"dt": dt_jac, "samples": samples_jac, "n_steps": sampling_steps})
                        if 'DDIM' not in infos:
                            infos['DDIM'] = {"samples": samples_ddim.cpu(), "steps": 100, "eta": .5}
                    all_exps.append(infos)
                    print(N_OBS, eps)
                    torch.save(all_exps,
                               os.path.join(path_to_save, 'gaussian_exp.pt'))

