import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev
from vp_diffused_priors import get_vpdiff_gaussian_score
from debug_learned_gaussian import diffused_tall_posterior_score, euler_sde_sampler
from ot.sliced import max_sliced_wasserstein_distance
from tqdm import tqdm
from time import time


def langevin_once(theta, t, score_fun, n_steps, r, nse, **kwargs):
    for i in range(n_steps):
        z = torch.randn_like(theta)
        g = score_fun(theta, t)
        eps = (
                r
                * (nse.alpha(t) ** .5)
                * min(nse.sigma(t) ** 2, 1 / g.square().mean())
        )
        # eps = 2*alpha_t*(r*torch.linalg.norm(z, axis=-1).mean(axis=0)/torch.linalg.norm(g, axis=-1).mean(axis=0))**2

        theta = (theta + eps * g + ((2 * eps) ** .5) * z).detach()
    return theta


def annealed_langevin(score_fn, t, n_samples, n_steps, nsr, dim, device, nse):
    samples = torch.randn((n_samples, dim)).to(device)
    timesteps = torch.linspace(0, 1, t).to(device)
    all_samples = [samples.clone().cpu()]
    for t in tqdm(torch.flip(timesteps, dims=(0,))):
        samples = langevin_once(samples, t, score_fun=score_fn, n_steps=n_steps, r=nsr, nse=nse)
        all_samples.append(samples.clone().cpu())
    return torch.stack(all_samples)


if __name__ == "__main__":
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from functools import partial

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
    ana_score = lambda theta, x, t, **kwargs: get_vpdiff_gaussian_score(mean=x,
                                                                        cov=cov_.cuda(),
                                                                        nse=score_net)(theta, t)

    data = {}
    for N_OBS in [10, 20, 30, 40, 60, 80, 100]:
        data[N_OBS] = {}
        score_fn_ana = partial(
            diffused_tall_posterior_score,
            prior=prior_,  # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
            dist_cov_est=cov_.cuda().repeat(N_OBS, 1, 1),  # cov_est,
            cov_mode='GAUSS',
            warmup_alpha=0,
            psd_clipping=False,
            score_fn=ana_score
        )
        ref_samples_ = euler_sde_sampler(
            score_fn_ana, N_SAMPLES, beta=score_net.beta, device="cuda:0", debug=True, theta_clipping_range=(-3, 3),
        )[1]
        ref_samples = ref_samples_ * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]
        data[N_OBS]["ref_samples"] = ref_samples
        score_fn = partial(
            diffused_tall_posterior_score,
            prior=prior_,  # normalized prior# analytical posterior
            x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
            nse=score_net,  # trained score network
            cov_mode='JAC',
            warmup_alpha=1,
            psd_clipping=False,
            scale_gradlogL=False
        )
        tstart = time()
        samples_ = annealed_langevin(score_fn=score_fn,
                                     n_samples=N_SAMPLES,
                                     n_steps=5,
                                     t=1000,
                                     nsr=.5,
                                     device='cuda:0',
                                     dim=2,
                                     nse=score_net).cpu()
        tend = time()
        samples = samples_ * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]
        data[N_OBS]['GAFNER2023'] = {
            "samples": samples,
            "dt": tend - tstart
        }
        data[N_OBS]['experiments'] = []
        for warmup_alpha in [0, 0.25, 0.5, 0.75][::-1]:
            for scale in [True, False]:
                for cov in ['GAUSS', 'JAC']:
                    if cov == 'GAUSS':
                        cov_est = vmap(lambda x: score_net.ddim(shape=(1000,), x=x, steps=100, eta=.5),
                                       randomness='different')(x_obs_100_[:N_OBS].cuda())
                        # normalize posterior

                        cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
                    else:
                        cov_est = None
                    score_fn = partial(
                        diffused_tall_posterior_score,
                        prior=prior_,  # normalized prior# analytical posterior
                        x_obs=x_obs_100_[:N_OBS].cuda(),  # observations
                        nse=score_net,  # trained score network
                        cov_mode=cov,
                        warmup_alpha=warmup_alpha,
                        psd_clipping=True if cov == 'JAC' else False,
                        scale_gradlogL=scale,
                        dist_cov_est=cov_est,
                    )

                    tstart = time()
                    samples_ = euler_sde_sampler(
                        score_fn, N_SAMPLES, beta=score_net.beta, device="cuda:0", debug=True, theta_clipping_range=(-3, 3),
                    )[1]
                    tend = time()
                    samples = samples_ * theta_train.std(axis=0)[None, None] + theta_train.mean(axis=0)[None, None]
                    data[N_OBS]['experiments'].append(
                        {
                            "scale": scale,
                            "cov_mode": cov,
                            "samples": samples,
                            "warmup_alpha": warmup_alpha,
                            "dt": tend - tstart
                        }
                    )
                    torch.save(data, f"gaussian_comparison.pt")
