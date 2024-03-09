import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import torch

from jax import random
from tasks.sbibm.task_mcmc import MCMCTask, get_mcmc_samples, get_predictive_sample


def model(
    prior_params={"loc": 0.0, "scale": 1.0},
    x_obs=None,
    n_obs=1,
    dim=10,
):
    theta = numpyro.sample(
        "theta",
        dist.MultivariateNormal(
            loc=jnp.ones(dim) * prior_params["loc"],
            covariance_matrix=jnp.eye(dim) * prior_params["scale"],
        ),
    )
    mixing_dist = dist.Categorical(probs=jnp.ones(2) / 2.0)
    component_dists = [
        dist.MultivariateNormal(loc=theta, covariance_matrix=jnp.eye(10) * 2.25),
        dist.MultivariateNormal(loc=theta, covariance_matrix=jnp.eye(10) / 9.0),
    ]
    mixture = dist.MixtureGeneral(mixing_dist, component_dists)
    if x_obs is None:
        for i in range(n_obs):
            numpyro.sample(f"obs_{i}", mixture)
    else:
        for i in range(n_obs):
            numpyro.sample(f"obs_{i}", mixture, obs=x_obs[i])  # , 0]


class GaussianMixture(MCMCTask):
    def __init__(self, dim=10, prior_params={"loc": 0.0, "scale": 1.0}, **kwargs):
        super().__init__(
            name="gaussian_mixture", prior_params=prior_params, model=model, **kwargs
        )

        self.dim = dim

        self.simulator_params = {
            "mixture_locs_factor": torch.tensor([1.0, 1.0]),
            "mixture_scales": torch.tensor([2.25, 1 / 9]),
            "mixture_weights": torch.tensor([0.5, 0.5]),
        }

    def prior(self):
        return torch.distributions.MultivariateNormal(
            loc=torch.tensor(
                [self.prior_params["loc"] for _ in range(self.dim)]
            ).float(),
            covariance_matrix=torch.eye(self.dim) * self.prior_params["scale"],
        )

    def _simulate_one(self, rng_key, theta, n_obs):
        x = get_predictive_sample(
            rng_key=rng_key,
            model=self.model,
            cond={"theta": theta},
            n_obs=n_obs,
            dim=self.dim,
        )
        return x  # shape (n_obs, dim_x=10)

    def _posterior_sampler(self, rng_key, x_star, theta_star, n_obs, num_samples):
        samples = get_mcmc_samples(
            rng_key=rng_key,
            model=self.model,
            init_value={"theta": theta_star},
            data=x_star,
            num_samples=num_samples,
            n_obs=n_obs,
        )["theta"]
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

    gmm = GaussianMixture(save_path=args.save_path)
    os.makedirs(gmm.save_path, exist_ok=True)

    if args.train_data:
        data = gmm.generate_training_data(rng_key=rng_key, n_simulations=50)
        print("Training data:", data["x"].shape, data["theta"].shape)

    if args.reference_data:
        ref_data = gmm.generate_reference_data(rng_key=rng_key)
        print(ref_data[0].shape, ref_data[1][1].shape)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                samples = gmm.generate_reference_posterior_samples(
                    rng_key=rng_key, num_obs=num_obs, n_obs=n_obs, num_samples=5
                )
                print(samples.shape)

    # # simulate one check
    theta = gmm.prior().sample((1,))
    # x = gmm.simulator(rng_key, theta, n_obs=1)
    # print(x.shape, theta.shape)
                
    # simulator distribution check
    from sbibm.tasks.gaussian_mixture.task import GaussianMixture as GaussianMixtureSBIBM
    gmm_sbibm = GaussianMixtureSBIBM(dim=10)
    x_sbibm = [gmm_sbibm.get_simulator()(theta) for _ in range(1000)]
    x_sbibm = torch.concatenate(x_sbibm, axis=0)
    x_jl = gmm.simulator(rng_key, theta, n_obs=1000)
    print(x_sbibm.shape, x_jl.shape)

    import matplotlib.pyplot as plt
    plt.scatter(x_sbibm[:,0], x_sbibm[:,1], label='sbibm')
    plt.scatter(x_jl[:,0], x_jl[:,1], label='jl')
    plt.legend()
    plt.savefig('gaussian_mixture_comparison.png')
    plt.clf()

    # x_star = torch.load('/data/parietal/store3/work/jlinhart/git_repos/diffusions-for-sbi/results/sbibm/gaussian_mixture/x_obs_100_num_1_new.pkl')
    # theta_star = torch.load('/data/parietal/store3/work/jlinhart/git_repos/diffusions-for-sbi/results/sbibm/gaussian_mixture/theta_true_list.pkl')[0][0]
    # samples_sbibm = torch.load('/data/parietal/store3/work/jlinhart/git_repos/diffusions-for-sbi/results/sbibm/gaussian_mixture/reference_posterior_samples/true_posterior_samples_num_1_n_obs_8.pkl')
    # samples_jl = gmm.sample_reference_posterior(rng_key=rng_key, x_star=x_star, theta_star=theta_star, n_obs=8, num_samples=1000)

    # print(samples_sbibm.shape, samples_jl.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(samples_sbibm[:,1], samples_sbibm[:,2], label='sbibm')
    # plt.scatter(samples_jl[:,1], samples_jl[:,2], label='jl')
    # plt.savefig('gaussian_mixture_comparison.png')
    # plt.clf()
