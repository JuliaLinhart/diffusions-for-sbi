from jax import random
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
import torch
import numpy as np
import os
import jax.lax as lax

from tasks.sbibm.task_mcmc import MCMCTask, get_mcmc_samples, get_predictive_sample

def summary_statistics(y, stimulus_I):

    if not isinstance(y, jnp.ndarray):
        y = jnp.array(y.numpy())
    if not isinstance(stimulus_I, jnp.ndarray):
        stimulus_I = jnp.array(stimulus_I.numpy())

    num_spikes = jnp.sum(y).reshape(1)

    sta = lax.conv_general_dilated(
        y.reshape(1, 1, -1).astype(jnp.float32),
        stimulus_I.reshape(1, 1, -1).astype(jnp.float32),
        window_strides=(1,),
        padding=[(8, 8)],
        feature_group_count=1,
    ).squeeze()[-9:]

    return jnp.concatenate([num_spikes, sta], axis=0)


def model(
    design_matrix,
    prior_params,
    x_obs=None,
    n_obs=1,
):
    prior_params_ = prior_params.copy()
    for k, v in prior_params_.items():
        prior_params_[k] = jnp.array(v.numpy())
    theta = numpyro.sample(
        "theta",
        dist.MultivariateNormal(**prior_params_),
    )

    design_matrix = jnp.array(design_matrix.numpy())

    if len(theta.shape) == 1:
        theta = theta[None, :]
        
    psi = jnp.matmul(design_matrix, theta[0,:])
    z = 1 / (1 + jnp.exp(-psi))

    for i in range(n_obs):
        if x_obs is None:
            numpyro.sample(f"obs_{i}", dist.Bernoulli(probs=z))
        else:
            numpyro.sample(f"obs_{i}", dist.Bernoulli(probs=z), obs=x_obs[i])
    

class BernoulliGLM(MCMCTask):
    def __init__(
            self, 
            prior_params=None,
            summary="sufficient", # or "raw"
            dim=10,
            **kwargs):
        
        self.summary = summary
        self.dim_theta = dim
        if summary == "sufficient":
            name = "bernoulli_glm"
            self.dim_x = 10
        elif summary == "raw":
            name = "bernoulli_glm_raw"
            self.dim_x = 100
        else:
            raise NotImplementedError

        super().__init__(
            name=name, prior_params=prior_params, model=model, **kwargs
        )
    
        self.stimulus = {
            "dt": 1,  # timestep
            "duration": 100,  # duration of input stimulus
            "seed": 42,  # seperate seed to freeze noise on input current
        }

        # Prior on offset and filter
        # Smoothness in filter encouraged by penalyzing 2nd order differences
        M = self.dim_theta - 1
        D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
        F = torch.matmul(D, D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
        Binv = torch.zeros(size=(M + 1, M + 1))
        Binv[0, 0] = 0.5  # offset
        Binv[1:, 1:] = torch.matmul(F.T, F)  # filter

        self.prior_params = {"loc": torch.zeros((M + 1,)), "precision_matrix": Binv}

    def prior(self):
        return torch.distributions.MultivariateNormal(**self.prior_params)
    
    def _simulate_one(self, rng_key, theta, n_obs, raw=False):
        stimulus_I = torch.load(f"{self.save_path}files/stimulus_I.pt")
        design_matrix = torch.load(f"{self.save_path}files/design_matrix.pt")

        x = get_predictive_sample(
            rng_key=rng_key,
            model=self.model,
            cond={"theta": theta},
            n_obs=n_obs,
            # model kwargs
            prior_params=self.prior_params,
            design_matrix=design_matrix,
        )

        if self.summary == "sufficient" and not raw:
            sta = []
            for i in range(n_obs):
                sta.append(summary_statistics(x[i], stimulus_I))
            sta = jnp.stack(sta)
            return sta
        return x # shape(n_obs, dim_x)

    def _posterior_sampler(self, rng_key, x_star, theta_star, n_obs, num_samples):
        design_matrix = torch.load(f"{self.save_path}files/design_matrix.pt")

        samples = get_mcmc_samples(
            rng_key=rng_key,
            model=self.model,
            init_value={"theta": theta_star},
            data=x_star,
            n_obs=n_obs,
            num_samples=num_samples,
            # model kwargs
            prior_params=self.prior_params,
            design_matrix=design_matrix,
        )["theta"]
        return samples
    
    
    def generate_reference_data(self, nb_obs=25, n_repeat=100, save=True, load_theta=False, **kwargs):
        kwargs["raw"] = True
        return super().generate_reference_data(nb_obs, n_repeat, save, load_theta, **kwargs)
    

    
    def generate_stimulus(self):
        path = f"{self.save_path}files/"
        path_2 = f"{self.save_path[:-1]}_raw/files/"
        os.makedirs(path, exist_ok=True)
        os.makedirs(path_2, exist_ok=True)
        stimulus_t = torch.arange(
            0, self.stimulus["duration"], self.stimulus["dt"], dtype=torch.float32
        )
        torch.save(stimulus_t, f"{path}stimulus_t.pt")
        torch.save(stimulus_t, f"{path_2}stimulus_t.pt")
        stimulus_I = torch.from_numpy(
            np.random.RandomState(self.stimulus["seed"])
            .randn(len(stimulus_t))
            .reshape(-1)
            .astype(np.float32)
        )
        torch.save(stimulus_I, f"{path}stimulus_I.pt")
        torch.save(stimulus_I, f"{path_2}stimulus_I.pt")

        # Build design matrix X, such that X * h returns convolution of x with filter h
        # Including linear offset by first element
        design_matrix = torch.zeros(size=(len(stimulus_t), self.dim_theta - 1))
        for j in range(self.dim_theta - 1):
            design_matrix[j:, j] = stimulus_I[0 : len(stimulus_t) - j]
        design_matrix = torch.cat(
            (torch.ones(size=(len(stimulus_t), 1)), design_matrix), axis=1
        )
        torch.save(design_matrix, f"{path}design_matrix.pt")
        torch.save(design_matrix, f"{path_2}design_matrix.pt")

        print(f"Saved stimulus and design matrix at {path} and {path_2}")

        return stimulus_I, design_matrix

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate data for the Lotka-Volterra example"
    )
    parser.add_argument(
        "--train_data", action="store_true", help="Generate training data"
    )
    parser.add_argument(
        "--reference_data", action="store_true", help="Generate reference data"
    )
    parser.add_argument(
        "--reference_posterior",
        action="store_true",
        help="Generate reference posterior samples",
    )
    parser.add_argument(
        "--save_path", type=str, default="data/", help="Path to save the data"
    )
    parser.add_argument(
        "--summary", type=str, default="sufficient", help="Whether to use sufficient summary statistics"
    )
    parser.add_argument(
        "--generate_stimulus", action="store_true", help="Generate stimulus"
    )

    args = parser.parse_args()

    rng_key = random.PRNGKey(1)

    bglm = BernoulliGLM(save_path=args.save_path, summary=args.summary)
    os.makedirs(bglm.save_path, exist_ok=True)

    if args.generate_stimulus:
        stimulus_I, design_matrix = bglm.generate_stimulus()
        print(stimulus_I.shape, design_matrix.shape)

    if args.train_data:
        data = bglm.generate_training_data(rng_key=rng_key, n_simulations=50)
        print(data["theta"].shape, data["x"].shape)

    if args.reference_data:
        ref_data = bglm.generate_reference_data(rng_key=rng_key)
        print(ref_data[0].shape, ref_data[1][1].shape)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                samples = bglm.generate_reference_posterior_samples(
                    rng_key=rng_key, num_obs=num_obs, n_obs=n_obs, num_samples=1_000
                )
                print(samples.shape)

    # # simulate one check
    # theta = bglm.prior().sample((1,))
    # x = bglm.simulator(rng_key, theta, n_obs=1)
    # print(x.shape, theta.shape)
                
    # # simulator distribution check
    # import sbibm
    # name = "bernoulli_glm" if args.summary == "sufficient" else "bernoulli_glm_raw"
    # bglm_sbibm = sbibm.get_task(name)
    # theta = bglm.prior().sample((1,))
    # x_sbibm = [bglm_sbibm.get_simulator()(theta) for _ in range(1000)]
    # x_sbibm = torch.concatenate(x_sbibm, axis=0)
    # x_jl = bglm.simulator(rng_key, theta, n_obs=1000)
    # print(x_sbibm.shape, x_jl.shape)

    # import matplotlib.pyplot as plt
    # plt.scatter(x_sbibm[:,0], x_sbibm[:,1], label='sbibm')
    # plt.scatter(x_jl[:,0], x_jl[:,1], label='jl')
    # plt.legend()
    # plt.savefig(f'{name}_sim_check.png')
    # plt.clf()

    # # reference posterior check
    # import sbibm
    # name = "bernoulli_glm_raw" if args.summary == "raw" else "bernoulli_glm"
    # bglm_sbibm = sbibm.get_task("bernoulli_glm_raw")
    # x_star = bglm_sbibm.get_observation(1)
    # theta_star = bglm_sbibm.get_true_parameters(1)
    # samples_sbibm = bglm_sbibm.get_reference_posterior_samples(1)[:1000]

    # if x_star.ndim == 1:
    #     x_star = x_star[None, :]
    # if theta_star.ndim > 1:
    #     theta_star = theta_star[0]
    
    # samples_jl = bglm.sample_reference_posterior(rng_key=rng_key, x_star=x_star, theta_star=theta_star, n_obs=1, num_samples=1000)

    # print(samples_sbibm.shape, samples_jl.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(samples_sbibm[:,0], samples_sbibm[:,1], label='sbibm')
    # plt.scatter(samples_jl[:,0], samples_jl[:,1], label='jl')
    # plt.scatter(theta_star[0], theta_star[1], label='theta_star')
    # plt.legend()
    # plt.savefig(f'{name}_post_check.png')
    # plt.clf()