from jax import random
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
import warnings
import torch
import numpy as np

from tasks.sbibm.task_mcmc import MCMCTask, get_mcmc_samples, get_predictive_sample


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def model(
    x_obs=None,
    n_obs=1,
    prior_params={"low": -3.0, "high": 3.0},
):
    theta_1 = numpyro.sample(
        "theta_1", dist.Uniform(low=prior_params["low"], high=prior_params["high"])
    )
    theta_2 = numpyro.sample(
        "theta_2", dist.Uniform(low=prior_params["low"], high=prior_params["high"])
    )
    theta_3 = numpyro.sample(
        "theta_3", dist.Uniform(low=prior_params["low"], high=prior_params["high"])
    )
    theta_4 = numpyro.sample(
        "theta_4", dist.Uniform(low=prior_params["low"], high=prior_params["high"])
    )
    theta_5 = numpyro.sample(
        "theta_5", dist.Uniform(low=prior_params["low"], high=prior_params["high"])
    )

    m = jnp.array([theta_1, theta_2])
    s1 = theta_3**2
    s2 = theta_4**2
    rho = jnp.tanh(theta_5)
    eps = 0.000001  # add eps to diagonal to ensure PSD
    S = jnp.array([[s1**2 + eps, rho * s1 * s2], [rho * s1 * s2, s2**2 + eps]])

    if x_obs is None:
        for i in range(n_obs):
            numpyro.sample(
                name=f"obs_{i}", fn=dist.MultivariateNormal(m, S), sample_shape=(4,)
            )
    else:
        for i in range(n_obs):
            numpyro.sample(
                f"obs_{i}",
                dist.MultivariateNormal(m, S),
                sample_shape=(4,),
                obs=x_obs[f"obs_{i}"],
            )


class SLCP(MCMCTask):
    def __init__(
        self,
        prior_params={"low": -3.0, "high": 3.0},
        num_samples_per_case=2500,
        **kwargs,
    ):
        super().__init__(name="slcp", prior_params=prior_params, model=model, **kwargs)

        self.num_samples_per_case = num_samples_per_case

    def prior(self):
        return torch.distributions.Uniform(
            low=torch.tensor([self.prior_params["low"] for _ in range(5)]).float(),
            high=torch.tensor([self.prior_params["high"] for _ in range(5)]).float(),
        )

    def _simulate_one(self, rng_key, theta, n_obs):
        # theta shape must be (5,) to iterate over the conditions
        if len(theta.shape) > 1:
            theta = theta[0]
        assert theta.shape[0] == 5

        # define the condition
        cond = {}
        for j in range(1, 5 + 1):
            cond[f"theta_{j}"] = theta[j - 1]

        # simulate
        x = get_predictive_sample(
            rng_key=rng_key, model=self.model, cond=cond, n_obs=n_obs
        )

        return x  # shape (n_obs, dim_x=8)

    def _posterior_sampler(self, rng_key, x_star, theta_star, n_obs, num_samples):
        # prepare the data object for numpyro
        data = {}
        for i in range(n_obs):
            data[f"obs_{i}"] = x_star[i][None, :].reshape(1, 4, 2)
        for i in range(1, 5 + 1):
            data[f"theta_{i}"] = theta_star[i - 1]

        # sample from the posterior
        init_value = {}
        for i in range(1, 5 + 1):
            init_value[f"theta_{i}"] = theta_star[i - 1]

        init_values_per_case = {
            "case_1": init_value,
            "case_2": init_value,
            "case_3": init_value,
            "case_4": init_value,
        }
        init_values_per_case["case_2"]["theta_3"] = -init_value["theta_3"]
        init_values_per_case["case_3"]["theta_4"] = -init_value["theta_4"]
        init_values_per_case["case_4"]["theta_3"] = -init_value["theta_3"]
        init_values_per_case["case_4"]["theta_4"] = -init_value["theta_4"]

        samples_mcmc = {}
        for case, init_value in init_values_per_case.items():
            samples_ = get_mcmc_samples(
                rng_key=rng_key,
                model=self.model,
                init_value=init_value,
                data=data,
                num_samples=self.num_samples_per_case,
                n_obs=n_obs,
            )
            samples_mcmc[case] = jnp.stack(
                [samples_[f"theta_{i+1}"] for i in range(5)]
            ).T

        samples = jnp.concatenate(
            [
                samples_mcmc["case_1"],
                samples_mcmc["case_2"],
                samples_mcmc["case_3"],
                samples_mcmc["case_4"],
            ]
        )

        # shuffle the samples to get samples fro meach chain
        samples = samples[np.random.permutation(samples.shape[0])]
        # keep only num_samples
        samples = samples[:num_samples]

        return samples


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

    args = parser.parse_args()

    rng_key = random.PRNGKey(1)

    slcp = SLCP(save_path=args.save_path, num_samples_per_case=2500)
    os.makedirs(slcp.save_path, exist_ok=True)

    if args.train_data:
        data = slcp.generate_training_data(rng_key=rng_key, n_simulations=50_000)
        print(data["x"].shape, data["theta"].shape)

    if args.reference_data:
        ref_data = slcp.generate_reference_data(rng_key=rng_key)
        print(ref_data[0].shape, ref_data[1][1].shape)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                samples = slcp.generate_reference_posterior_samples(
                    rng_key=rng_key, num_obs=num_obs, n_obs=n_obs, num_samples=1_000
                )
                print(samples.shape)

    # # simulate one check
    # theta = slcp.prior().sample((1,))
    # x = slcp.simulator(rng_key, theta, n_obs=1)
    # print(x.shape, theta.shape)
