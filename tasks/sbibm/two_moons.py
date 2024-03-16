import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch
import math
import numpy as np
from jax import random
from tasks.sbibm.task_mcmc import MCMCTask, get_mcmc_samples, get_predictive_sample


def model(
        prior_params={"bound": 1.},
        x_obs=None,
        n_obs=1,
        predictive=False,
):  
    theta = numpyro.sample(
        "theta",
        dist.Uniform(
            low=jnp.ones(2) * -prior_params["bound"],
            high=jnp.ones(2) * prior_params["bound"],
        ),
    )

    a_dist = dist.Uniform(
        low=-math.pi / 2.0,
        high=+math.pi / 2.0,
    )
    r_dist = dist.Normal(
        loc=0.1,
        scale=0.01,
    )

    if x_obs is None:
        for i in range(n_obs):
            a = numpyro.sample(f"a_{i}", a_dist)
            r = numpyro.sample(f"r_{i}", r_dist)
            p = jnp.array(
                [
                    jnp.cos(a) * r + 0.25,
                    jnp.sin(a) * r,
                ]
            )

            ang = jnp.array([-math.pi / 4.0])
            c = jnp.cos(ang)
            s = jnp.sin(ang)
            z0 = c * theta[0] - s * theta[1]
            z1 = s * theta[0] + c * theta[1]

            transformed_value = p + jnp.array([-jnp.abs(z0), z1]).reshape(1,-1)
            mvn = dist.MultivariateNormal(
                loc=transformed_value,
                covariance_matrix=jnp.eye(2) * 0.0001,
            )
            delta = dist.Delta(transformed_value)
            sample_dist = delta if predictive else mvn
            numpyro.sample(f"obs_{i}", sample_dist)
    else:
        for i in range(n_obs):
            a = numpyro.sample(f"a_{i}", a_dist)
            r = numpyro.sample(f"r_{i}", r_dist)
            p = jnp.array(
                [
                    jnp.cos(a) * r + 0.25,
                    jnp.sin(a) * r,
                ]
            )

            ang = jnp.array([-math.pi / 4.0])
            c = jnp.cos(ang)
            s = jnp.sin(ang)
            z0 = c * theta[0] - s * theta[1]
            z1 = s * theta[0] + c * theta[1]

            transformed_value = p + jnp.array([-jnp.abs(z0), z1]).reshape(1,-1)
            mvn = dist.MultivariateNormal(
                loc=transformed_value,
                covariance_matrix=jnp.eye(2) * 1e-6,
            )
            delta = dist.Delta(transformed_value)
            sample_dist = delta if predictive else mvn
            numpyro.sample(f"obs_{i}", sample_dist, obs=x_obs[f"obs_{i}"])
    

class TwoMoons(MCMCTask):
    def __init__(
        self,
        prior_params={"bound":1.},
        num_samples_per_case=500,
        **kwargs,
    ):
        super().__init__(name="two_moons", prior_params=prior_params, model=model, **kwargs)

        self.prior_params["low"] = torch.tensor([-self.prior_params["bound"] for _ in range(2)]).float()
        self.prior_params["high"] = torch.tensor([self.prior_params["bound"] for _ in range(2)]).float()

        self.num_samples_per_case = num_samples_per_case

        self.dim_theta = 2
        self.dim_x = 2

    def prior(self):
        return torch.distributions.Uniform(
            low=self.prior_params["low"],
            high=self.prior_params["high"],
        )
    
    def _simulate_one(self, rng_key, theta, n_obs):
        if len(theta.shape) > 1:
            theta = theta[0]
        assert theta.shape[0] == 2

        x = get_predictive_sample(
            rng_key=rng_key,
            model=model,
            cond={"theta": theta},
            n_obs=n_obs,
            predictive=True,
        )
        return x  # shape (n_obs, dim_x=2)
    
    def _posterior_sampler(self, rng_key, x_star, theta_star, n_obs, num_samples, **kwargs):
        # prepare the data object for numpyro
        data = {}
        for i in range(n_obs):
            data[f"obs_{i}"] = x_star[i][None, :]
        for i in range(1, 2 + 1):
            data[f"theta_{i}"] = theta_star[i - 1]

        # init values
        ang = jnp.array([-math.pi / 4.0])
        c = jnp.cos(-ang)
        s = jnp.sin(-ang)

        # p = self.simulator(rng_key, torch.zeros(1,2))
        # p = jnp.array(p.numpy())
        q_0 = - x_star[0, 0]
        q_1 = x_star[0, 1]

        init_1 = jnp.array([c * q_0 - s * q_1, s * q_0 + c * q_1])[:,0]
        init_2 = jnp.array([-c * q_0 - s * q_1, -s * q_0 + c * q_1])[:,0]

        init_values_per_case = {
            "case_1": {"theta": init_1},
            "case_2": {"theta": init_2},
        }

        # sample from the posterior
        samples_mcmc = {}
        for case, init_value in init_values_per_case.items():
            samples_mcmc[case] = get_mcmc_samples(
                rng_key=rng_key,
                model=self.model,
                init_value=init_value,
                data=data,
                num_samples=self.num_samples_per_case,
                n_obs=n_obs,
            )["theta"]

        samples = jnp.concatenate(
            [
                samples_mcmc["case_1"],
                samples_mcmc["case_2"],
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
    parser.add_argument(
        "--check_sim", action="store_true", help="Check the simulator"
    )
    parser.add_argument(
        "--check_post", action="store_true", help="Check the reference posterior"
    )

    args = parser.parse_args()

    rng_key = random.PRNGKey(1)

    two_moons = TwoMoons(save_path=args.save_path)
    os.makedirs(two_moons.save_path, exist_ok=True)

    if args.train_data:
        data = two_moons.generate_training_data(rng_key=rng_key, n_simulations=50)
        print("Training data:", data["x"].shape, data["theta"].shape)

    if args.reference_data:
        ref_data = two_moons.generate_reference_data(rng_key=rng_key)
        print(ref_data[0].shape, ref_data[1][1].shape)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                samples = two_moons.generate_reference_posterior_samples(
                    rng_key=rng_key, num_obs=num_obs, n_obs=n_obs, num_samples=5
                )
                print(samples.shape)

    if args.check_sim:
        # # simulate one check
        # theta = two_moons.prior().sample((1,))
        # x = two_moons.simulator(rng_key, theta, n_obs=1)
        # print(x.shape, theta.shape)

        # simulator distribution check
        import sbibm
        tm_sbibm = sbibm.get_task("two_moons")
        theta = two_moons.prior().sample((1,))
        x_sbibm = [tm_sbibm.get_simulator()(theta) for _ in range(1000)]
        x_sbibm = torch.concatenate(x_sbibm, axis=0)
        x_jl = two_moons.simulator(rng_key, theta, n_obs=1000)
        print(x_sbibm.shape, x_jl.shape)

        import matplotlib.pyplot as plt
        plt.scatter(x_sbibm[:,0], x_sbibm[:,1], label='sbibm')
        plt.scatter(x_jl[:,0], x_jl[:,1], label='jl')
        plt.legend()
        plt.savefig('_checks/two_moons_sim_check.png')
        plt.clf()

    if args.check_post:
        # reference posterior check
        import sbibm
        tm_sbibm = sbibm.get_task("two_moons")

        x_star = two_moons.get_reference_observation(num_obs=2)
        theta_star = two_moons.get_reference_parameters()[1]
        samples_sbibm = tm_sbibm._sample_reference_posterior(num_samples=1000, observation=x_star, num_observation=2)
        samples_jl = two_moons.sample_reference_posterior(rng_key=rng_key, x_star=x_star, theta_star=theta_star, n_obs=1, num_samples=1000)
        # samples_jl_30 = two_moons.sample_reference_posterior(rng_key=rng_key, x_star=x_star, theta_star=theta_star, n_obs=30, num_samples=1000)

        print(samples_sbibm.shape, samples_jl.shape)
        import matplotlib.pyplot as plt
        plt.scatter(samples_sbibm[:,0], samples_sbibm[:,1], label='sbibm')
        plt.scatter(samples_jl[:,0], samples_jl[:,1], label='jl')
        # plt.scatter(samples_jl_30[:,0], samples_jl_30[:,1], label='jl_30')
        plt.scatter(theta_star[0], theta_star[1], label='theta_star')
        plt.scatter(x_star[0,0], x_star[0,1], label='x_star')
        plt.legend()
        plt.savefig('_checks/two_moons_post_check.png')
        plt.clf()

    import sbibm
    import matplotlib.pyplot as plt
    tm_sbibm = sbibm.get_task("two_moons")
    theta = tm_sbibm.get_prior()(1000)
    x = [tm_sbibm.get_simulator()(theta_) for theta_ in theta]
    x = torch.cat(x, axis=0)
    
    data = two_moons.generate_training_data(n_simulations=1000, save=False)
    x_new = data["x"]
    theta_new = data["theta"]

    plt.scatter(x[:,0], x[:,1], label='sbibm')
    plt.scatter(x_new[:,0], x_new[:,1], label='jl')
    plt.legend()
    plt.savefig('_checks/two_moons_train_x_check.png')
    plt.clf()

    plt.scatter(theta[:,0], theta[:,1], label='sbibm')
    plt.scatter(theta_new[:,0], theta_new[:,1], label='jl')
    plt.legend()
    plt.savefig('_checks/two_moons_train_theta_check.png')
    plt.clf()