import torch
import jax.numpy as jnp
import numpy as np
import os
from numpyro.infer import NUTS, MCMC, Predictive, init_to_value
from numpyro.handlers import condition

from functools import partial
from jax import random
from tqdm import tqdm


class Task:
    # general task class for SBIBM
    def __init__(
        self, name, prior_params, model, torch_version=True, save_path="data/"
    ):
        self.name = name
        self.prior_params = prior_params
        self.model = partial(model, prior_params=prior_params)
        self.torch_version = torch_version
        self.save_path = save_path + name + "/"

    def _prior_dist(self, torch_version):
        raise NotImplementedError

    def _simulate_one(self, rng_key, theta, n_obs, torch_version):
        raise NotImplementedError

    def _mcmc_sampler(self, rng_key, x_star, theta_star, n_obs, num_samples):
        raise NotImplementedError

    def prior(self, torch_version=None):
        if torch_version is None:
            torch_version = self.torch_version

        return self._prior_dist(torch_version=torch_version)

    def simulator(self, rng_key, theta, n_obs=1, torch_version=None):
        if torch_version is None:
            torch_version = self.torch_version

        # ensure theta is a jax array
        if isinstance(theta, torch.Tensor):
            theta = theta.numpy()
        theta = jnp.array(theta)

        x = self._simulate_one(
            rng_key=rng_key, theta=theta, n_obs=n_obs, torch_version=torch_version
        )
        return x

    def sample_reference_posterior(
        self, rng_key, x_star, theta_star, n_obs=1, num_samples=1000
    ):
        # ensure theta_star is a jax array
        if isinstance(theta_star, torch.Tensor):
            theta_star = theta_star.numpy()
        theta_star = jnp.array(theta_star)

        # ensure x_star is a jax array
        if isinstance(x_star, torch.Tensor):
            x_star = x_star.numpy()
        x_star = jnp.array(x_star)

        # ensure x_star contains n_obs observations
        if x_star.shape[0] > n_obs:
            x_star = x_star[:n_obs]
        assert x_star.shape[0] == n_obs

        samples = self._mcmc_sampler(
            rng_key=rng_key,
            x_star=x_star,
            theta_star=theta_star,
            n_obs=n_obs,
            num_samples=num_samples,
        )
        return samples

    def generate_training_data(self, rng_key, n_simulations, save=True):
        print("Generating training data")
        # prior samples
        prior = self.prior(torch_version=False)
        rng_key, subkey = random.split(rng_key)
        theta_train = prior.sample(subkey, (n_simulations,))

        # simulator samples
        x_train = []
        for theta_i in tqdm(theta_train):
            x_i = self.simulator(
                rng_key=rng_key, theta=theta_i, n_obs=1, torch_version=False
            )
            x_train.append(x_i)
        x_train = jnp.stack(x_train)[:, 0, :]

        if self.torch_version:
            x_train = torch.from_numpy(np.asarray(x_train)).float()
            theta_train = torch.tensor(np.asarray(theta_train)).float()

        dataset_train = {"theta": theta_train, "x": x_train}
        if save:
            filename = f"{self.save_path}dataset_n_train_{n_simulations}.pkl"
            print("Saving at", filename)
            torch.save(dataset_train, filename)

        return dataset_train

    def generate_reference_data(
        self, rng_key, nb_obs=25, n_repeat=100, save=True, load_theta=False
    ):
        version = "torch" if self.torch_version else "jax"
        print(
            f"Generating reference data ({version} version) for {nb_obs} observations and {n_repeat} repetitions."
        )

        # reference parameters
        filename = f"{self.save_path}theta_true_list.pkl"
        if not load_theta:
            rng_key, subkey = random.split(rng_key)
            prior = self.prior(torch_version=False)
            theta_star = prior.sample(subkey, (nb_obs,))
            if self.torch_version:
                theta_star = torch.tensor(np.asarray(theta_star)).float()
            if save:
                print("Saving at", filename)
                torch.save(theta_star, filename)
        else:
            theta_star = torch.load(filename)

        # reference observations
        x_star = {}
        for num_obs in range(1, nb_obs + 1):
            theta_true = theta_star[num_obs - 1]
            x_star[num_obs] = self.simulator(
                rng_key=rng_key, theta=theta_true, n_obs=n_repeat
            )
            if save:
                path = f"{self.save_path}reference_observations/"
                os.makedirs(path, exist_ok=True)
                filename = f"{path}x_obs_{n_repeat}_num_{num_obs}.pkl"
                print("Saving at", filename)
                torch.save(x_star[num_obs], filename)

        return theta_star, x_star

    def get_reference_parameters(self):
        filename = f"{self.save_path}theta_true_list.pkl"
        theta_star = torch.load(filename)
        if self.torch_version and not isinstance(theta_star, torch.Tensor):
            theta_star = torch.tensor(np.asarray(theta_star)).float()
        return theta_star

    def get_reference_observation(self, num_obs, n_repeat=100):
        filename = (
            f"{self.save_path}reference_observations/x_obs_{n_repeat}_num_{num_obs}.pkl"
        )
        x_star = torch.load(filename)
        if self.torch_version and not isinstance(x_star, torch.Tensor):
            x_star = torch.tensor(np.asarray(x_star)).float()
        return x_star

    def generate_reference_posterior_samples(
        self, rng_key, num_obs, n_obs, num_samples=1000, save=True
    ):
        print(
            "Generating reference posterior samples for num_obs =",
            num_obs,
            "and n_obs =",
            n_obs,
        )

        # reference data for num_obs
        theta_star = self.get_reference_parameters()[num_obs - 1]
        x_star = self.get_reference_observation(num_obs=num_obs)

        # sample from the posterior
        samples = self.sample_reference_posterior(
            rng_key=rng_key,
            x_star=x_star,
            theta_star=theta_star,
            n_obs=n_obs,
            num_samples=num_samples,
        )
        if save:
            path = f"{self.save_path}reference_posterior_samples/"
            os.makedirs(path, exist_ok=True)
            filename = f"{path}true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
            print("Saving at", filename)
            torch.save(samples, filename)
        return samples

    def get_reference_posterior_samples(self, num_obs, n_obs):
        filename = f"{self.save_path}reference_posterior_samples/true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
        samples = torch.load(filename)

        if self.torch_version and not isinstance(samples, torch.Tensor):
            samples = torch.tensor(np.asarray(samples)).float()
        return samples


def get_predictive_sample(rng_key, model, cond, n_obs, **model_kwargs):
    predictive = Predictive(condition(model, cond), num_samples=1)
    rng_key, subkey = random.split(rng_key)
    data = predictive(subkey, n_obs=n_obs, **model_kwargs)
    # concatenate the observations
    x = jnp.stack([data[f"obs_{i}"].reshape(-1) for i in range(n_obs)])

    return x


def get_mcmc_samples(
    rng_key, model, init_value, data, num_samples, n_obs, **model_kwargs
):
    kernel = NUTS(model, init_strategy=init_to_value(site=None, values=init_value))
    mcmc = MCMC(kernel, num_warmup=100, num_samples=num_samples, num_chains=1)
    rng_key, subkey = random.split(rng_key)
    mcmc.run(subkey, x_obs=data, n_obs=n_obs, **model_kwargs)

    return mcmc.get_samples()
