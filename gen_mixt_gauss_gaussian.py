# Script for the Gaissian Mixture toy example. Reproduces experiemnts from section 4.1 of the paper.

import os
import torch

from torch.func import vmap, grad
from embedding_nets import EpsilonNet, FakeFNet
from vp_diffused_priors import get_vpdiff_gaussian_score
from tqdm import tqdm
from tasks.toy_examples.data_generators import Gaussian_MixtGaussian_mD
import time
from nse_toy import mean_backward, prec_matrix_backward

# MALA sampler for true posterior
def MALA(x, lr, logpdf_fun, n_iter):
    pbar = tqdm(range(n_iter))
    for i in pbar:
        x.requires_grad_(True)
        logpdf_x = logpdf_fun(x)
        logpdf_x.sum().backward()
        eps = torch.randn_like(x)
        candidate = x.detach() + lr * x.grad.detach() + ((2 * lr) ** 0.5) * eps
        candidate.requires_grad_(True)
        logpdf_candidate = logpdf_fun(candidate)
        logpdf_candidate.sum().backward()
        backward_eps = (x - candidate - lr * candidate.grad) / ((2 * lr) ** 0.5)

        log_ratio = (
            logpdf_candidate
            - logpdf_x
            - 0.5 * torch.linalg.norm(backward_eps, dim=-1) ** 2
            + 0.5 * torch.linalg.norm(eps, dim=-1) ** 2
        )
        # log_ratio = logpdf_candidate - logpdf_x - torch.linalg.norm(backward_eps, dim=-1)**2 + torch.linalg.norm(eps, dim=-1)**2

        u = torch.log(torch.rand(size=(x.shape[0],))).to(x.device)
        is_accepted = u <= log_ratio
        x = x.detach()
        x[is_accepted] = candidate[is_accepted].detach()
        accept_rate = is_accepted.float().mean().item()
        pbar.set_description(f"Acceptance rate: {accept_rate:.2f}, Lr: {lr:.2e}")
        # print(x[0])
        if i < n_iter // 2:
            if accept_rate > 0.55:
                lr = lr * 1.01
            if accept_rate < 0.45:
                lr = lr * 0.99

    return x


if __name__ == "__main__":
    from nse_toy import NSE
    import sys
    path_to_save = sys.argv[1]
    os.makedirs(path_to_save, exist_ok=True)

    torch.manual_seed(1)

    all_exps = []
    for DIM in [10]: #, 50, 100]:
        for eps in [0, 1e-3, 1e-2, 1e-1]:
            for seed in range(5):
                # Simulator and observations
                torch.manual_seed(seed)  # between 0.1 and 25.1
                task = Gaussian_MixtGaussian_mD(dim=DIM)
                prior = task.prior
                simulator = task.simulator
                theta_true = prior.sample(sample_shape=(1,))[0]  # true parameters

                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
                )

                # True posterior score / epsilon network
                score_net = NSE(theta_dim=DIM, x_dim=DIM) #, net_type="fnet")
                # beta_min = 0.1
                # beta_max = 40
                # beta_d = beta_max - beta_min
                # score_net.alpha = lambda t: torch.exp(
                #     -0.5 * 0.5 * beta_d * (t**2) + beta_min * t
                # )

                def real_eps(theta, x, t):
                    score = vmap(
                        grad(
                            lambda theta, x, t: task.diffused_posterior(
                                x, score_net.alpha(t)
                            ).log_prob(theta)
                        )
                    )(theta, x, t)
                    return -score_net.sigma(t) * score

                # Perturbed score network
                score_net.net = FakeFNet(
                    real_eps_fun=real_eps, eps_net=EpsilonNet(DIM), eps_net_max=eps
                )
                score_net.cuda()

                # Prior score
                t = torch.linspace(0, 1, 1000)
                prior_score_fn = get_vpdiff_gaussian_score(prior.loc.cuda(), prior.covariance_matrix.cuda(), score_net)

                def prior_score(theta, t):
                    mean_prior_0_t = mean_backward(theta, t, prior_score_fn, score_net)
                    prec_prior_0_t = prec_matrix_backward(t, prior.covariance_matrix.cuda(), score_net).repeat(
                        theta.shape[0], 1, 1
                    )
                    prior_score = prior_score_fn(theta, t)
                    return prior_score, mean_prior_0_t, prec_prior_0_t

                # Sampling
                for N_OBS in [2, 4, 8, 16, 32, 64, 90][::-1]:
                    # True posterior samples
                    ref_samples = MALA(
                        x=torch.randn((10_000, DIM)).cuda() * (1 / N_OBS)
                        + x_obs_100[:N_OBS].mean(),  ## does x_obs change something??
                        lr=1e-3 / N_OBS,
                        logpdf_fun=vmap(
                            lambda theta: (1 - N_OBS) * prior.log_prob(theta)
                            + vmap(lambda x: task.true_posterior(x).log_prob(theta))(
                                x_obs_100[:N_OBS]
                            ).sum(dim=0)
                        ),
                        n_iter=1_000,
                    ).cpu()

                    infos = {
                        "ref_samples": ref_samples,
                        "N_OBS": N_OBS,
                        "seed": seed,
                        "dim": DIM,
                        "eps": eps,
                        "exps": {"Langevin": [], "GAUSS": [], "JAC": []},
                    }

                    for tau, lsteps in zip(
                        [0.5, 0.1, 0.05, 0.01], [5, 10, 50, 100]
                    ):
                        infos["exps"][f"Langevin_L_{lsteps}_tau_{tau}"] = []


                    # Approximate posterior samples
                    for sampling_steps, eta in zip(
                        [50, 150, 400, 1000][::-1], [0.2, 0.5, 0.8, 1][::-1]
                    ):
                        tstart_gauss = time.time()
                        # Estimate the Gaussian covariance
                        samples_ddim = (
                            score_net.ddim(
                                shape=(1000 * N_OBS,),
                                x=x_obs_100[None, :N_OBS]
                                .repeat(1000, 1, 1)
                                .reshape(1000 * N_OBS, -1)
                                .cuda(),
                                steps=100,
                                eta=0.5,
                            )
                            .detach()
                            .reshape(1000, N_OBS, -1)
                            .cpu()
                        )
                        cov_est = vmap(lambda x: torch.cov(x.mT))(
                            samples_ddim.permute(1, 0, 2)
                        )
                        # Sample with "GAUSS"
                        print()
                        print("GAUSS sampling...")
                        samples_gauss = score_net.ddim(
                            shape=(1000,),
                            x=x_obs_100[:N_OBS].cuda(),
                            eta=eta,
                            steps=sampling_steps,
                            prior_score_fn=prior_score,
                            # prior=prior,
                            dist_cov_est=cov_est.cuda(),
                            cov_mode="GAUSS",
                        ).cpu()
                        # print(samples_gauss.isnan().sum())

                        tstart_jac = time.time()
                        # Sample with JAC
                        print()
                        print("JAC sampling...")
                        samples_jac = score_net.ddim(
                            shape=(1000,),
                            x=x_obs_100[:N_OBS].cuda(),
                            eta=eta,
                            steps=sampling_steps,
                            prior_score_fn=prior_score,
                            # prior=prior,
                            cov_mode="JAC",
                        ).cpu()
                        # print(samples_jac.isnan().sum())

                        # Sample with Langevin
                        for tau, lsteps in zip(
                            [0.5, 0.1, 0.05, 0.01], [5, 10, 50, 100]
                        ):
                            tstart_lang = time.time()
                            with torch.no_grad():
                                print()
                                print(f"Langevin sampling... Lsteps: {lsteps}, tau: {tau}")
                                lang_samples = score_net.annealed_langevin_geffner(
                                    shape=(1000,),
                                    x=x_obs_100[:N_OBS].cuda(),
                                    prior_score_fn=prior_score,
                                    lsteps=lsteps,
                                    tau=tau,
                                    steps=sampling_steps,
                                ).cpu()
                            t_end_lang = time.time()
                            dt_lang = t_end_lang - tstart_lang

                            infos["exps"][f"Langevin_L_{lsteps}_tau_{tau}"].append(
                            {
                                "dt": dt_lang,
                                "samples": lang_samples,
                                "n_steps": sampling_steps,
                            }
                        )

                        dt_gauss = tstart_jac - tstart_gauss
                        dt_jac = tstart_lang - tstart_jac

                        infos["exps"]["GAUSS"].append(
                            {
                                "dt": dt_gauss,
                                "samples": samples_gauss,
                                "n_steps": sampling_steps,
                            }
                        )
                        infos["exps"]["JAC"].append(
                            {
                                "dt": dt_jac,
                                "samples": samples_jac,
                                "n_steps": sampling_steps,
                            }
                        )
                        all_exps.append(infos)
                        print(f"N_OBS: {N_OBS}, eps: {eps}")
                        torch.save(
                            all_exps,
                            os.path.join(path_to_save, "gaussian_mixture_exp_langevin.pt"),
                        )
