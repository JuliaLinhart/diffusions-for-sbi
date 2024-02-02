import os
#from pyro.infer import MCMC, HMC
import matplotlib.pyplot as plt
import torch

from torch.func import vmap, jacrev, grad
from vp_diffused_priors import get_vpdiff_gaussian_score
from ot.sliced import max_sliced_wasserstein_distance
from tqdm import tqdm
from functools import partial
from tasks.toy_examples.data_generators import Gaussian_MixtGaussian_mD
import time
from nse import mean_backward, prec_matrix_backward



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


def MALA(x,
         lr,
         logpdf_fun,
         n_iter):
    pbar = tqdm(range(n_iter))
    for i in pbar:
        x.requires_grad_(True)
        logpdf_x = logpdf_fun(x)
        logpdf_x.sum().backward()
        eps = torch.randn_like(x)
        candidate = x.detach() + lr * x.grad.detach() + ((2*lr)**.5)*eps
        candidate.requires_grad_(True)
        logpdf_candidate = logpdf_fun(candidate)
        logpdf_candidate.sum().backward()
        backward_eps = (x - candidate - lr * candidate.grad) / ((2 * lr)**.5)

        log_ratio = logpdf_candidate - logpdf_x - .5*torch.linalg.norm(backward_eps, dim=-1)**2 + .5*torch.linalg.norm(eps, dim=-1)**2
        #log_ratio = logpdf_candidate - logpdf_x - torch.linalg.norm(backward_eps, dim=-1)**2 + torch.linalg.norm(eps, dim=-1)**2

        u = torch.log(torch.rand(size=(x.shape[0],))).to(x.device)
        is_accepted = u <= log_ratio
        x = x.detach()
        x[is_accepted] = candidate[is_accepted].detach()
        accept_rate = is_accepted.float().mean().item()
        pbar.set_description(f'Acceptance rate: {accept_rate:.2f}, Lr: {lr:.2e}')
        #print(x[0])
        if i < n_iter // 2:
            if accept_rate > .55:
                lr = lr * 1.01
            if accept_rate < .45:
                lr = lr*.99

    return x


def get_hmc_samples(task, prior, x_obs_100, N_OBS):
    posteriors = [task.true_posterior(x) for x in x_obs_100[:N_OBS]]
    def posterior_fun(theta):
        lk = 0
        for p in posteriors:
            lk += p.log_prob(theta)
        return lk
    def potential(theta_dic):
        theta = theta_dic["theta"]
        potential = vmap(lambda theta: (1 - N_OBS) * prior.log_prob(theta) + posterior_fun(theta).sum(dim=0))(
            theta
        ).sum()
        return potential

    from pyro.infer import MCMC, HMC
    kern = HMC(
        potential_fn=potential,
        step_size=1e-4 / N_OBS,
        num_steps=100
    )
    hmc = MCMC(
        kernel=kern,
        num_samples=100,
        warmup_steps=500,
        initial_params=
        {"theta": torch.randn((10_000, DIM)).cuda() + x_obs[:N_OBS].mean()}
    )
    hmc.run()
    ref_samples = hmc.get_samples()["theta"][-1].detach()
    return ref_samples.cpu()


if __name__ == "__main__":
    #from tasks.toy_examples.data_generators import SBIGaussian2d
    from nse import NSE, NSELoss
    from sm_utils import train

    torch.manual_seed(1)
    N_TRAIN = 10_000
    N_SAMPLES = 4096
    all_exps = []
    for DIM in [10, 50, 100]:
        for eps in [0, 1e-3, 1e-2, 1e-1]:
            for seed in range(5):
                # Observations
                torch.manual_seed(seed)# between 0.1 and 25.1
                task = Gaussian_MixtGaussian_mD(dim=DIM)
                prior = task.prior
                simulator = task.simulator
                theta_true = prior.sample(sample_shape=(1,))[0]  # true parameters

                x_obs = simulator(theta_true)  # x_obs ~ simulator(theta_true)
                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
                )

                # True posterior: p(theta|x_obs)
                true_posterior = task.true_posterior(x_obs)
                # Train data
                theta_train = task.prior.sample((N_TRAIN,))
                x_train = vmap(simulator, randomness='same')(theta_train)

                score_net = NSE(theta_dim=DIM, x_dim=DIM)

                def real_eps(theta, x, t):
                    score = vmap(grad(lambda theta, x, t: task.diffused_posterior(x, score_net.alpha(t)).log_prob(theta)))(theta, x, t)
                    return - score_net.sigma(t) *score

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
                    return prior_score, mean_prior_0_t, prec_prior_0_t
                #samples = score_net.ddim(shape=(1000,), x=x_obs_100.mean(dim=0)[None].repeat(1000, 1), steps=1000)
                #sanity check
                # fig, ax = plt.subplots(1, 1)
                # ax.scatter(*task.true_posterior(x_obs_100.mean(dim=0)).sample((1000,))[:, :2].cpu().T)
                # ax.scatter(*samples.cpu()[:, :2].T)
                # ax.scatter(*theta_true[:2].cpu())
                # fig.show()

                for N_OBS in [2, 4, 8, 16, 32, 64, 90][::-1]:
                    ref_samples = MALA(
                        x=torch.randn((10_000, DIM)).cuda()*(1 / N_OBS) + x_obs[:N_OBS].mean(),
                        lr=1e-3 / N_OBS,
                        logpdf_fun=vmap(lambda theta: (1-N_OBS)*prior.log_prob(theta) + vmap(lambda x: task.true_posterior(x).log_prob(theta))(x_obs_100[:N_OBS]).sum(dim=0)),
                        n_iter=1_000).cpu()

                    infos = {"ref_samples": ref_samples,
                             "N_OBS": N_OBS,
                             "seed": seed,
                             "dim": DIM,
                             "eps": eps,
                             "exps":
                                 {
                                 "Langevin": [],
                                 "GAUSS": [],
                                 "JAC": []}
                             }

                    for sampling_steps, eta in zip(
                            [50, 150, 400, 1000][::-1],
                            [.2, .5, .8, 1][::-1]):
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
                                                                               steps=sampling_steps).cpu()
                        t_end_lang = time.time()
                        dt_gauss = tstart_jac - tstart_gauss
                        dt_jac = tstart_lang - tstart_jac
                        dt_lang = t_end_lang - tstart_lang
                        #     true_posterior_cov = torch.linalg.inv(inv_lik*N_OBS + inv_prior)
                        #     true_posterior_mean = (true_posterior_cov @ (inv_prior_prior + inv_lik @ x_obs_100[:N_OBS].sum(dim=0).cuda()))
                        #     ref_samples = torch.distributions.MultivariateNormal(loc=true_posterior_mean, covariance_matrix=true_posterior_cov).sample((1000,)).cpu()
                        infos["exps"]["Langevin"].append({"dt": dt_lang, "samples": lang_samples, "n_steps": sampling_steps})
                        infos["exps"]["GAUSS"].append({"dt": dt_gauss, "samples": samples_gauss, "n_steps": sampling_steps})
                        infos["exps"]["JAC"].append({"dt": dt_jac, "samples": samples_jac, "n_steps": sampling_steps})
                        all_exps.append(infos)
                        print(N_OBS, eps)
                        torch.save(all_exps,
                                   '/mnt/data/gabriel/sbi/gaussian_mixture_exp.pt')
                        # fig, axes = plt.subplots(1, 1, figsize=(12, 8))
                        # fig.suptitle(f'N={N_OBS} eps={eps}')
                        # axes.scatter(*samples_gauss[..., :2].T, label='Gauss', alpha=.2)#, label='Ground truth')
                        # axes.scatter(*samples_jac[..., :2].T, label='Jac', alpha=.2)#, label='Ground truth')
                        # axes.scatter(*ref_samples[..., :2].T, label='Ref', alpha=.2)
                        # axes.scatter(*lang_samples[..., :2].T, label='Lang', alpha=.2)#, label='Ground truth')
                        #
                        # axes.legend()
                        # #axes[0].scatter(*ref_samples[..., :2].T)
                        # # for ax in axes[-1]:
                        # #     ax.set_xlabel("theta_1")
                        # # for ax in axes[:, 0]:
                        # #     ax.set_ylabel("theta_2")
                        # fig.show()
