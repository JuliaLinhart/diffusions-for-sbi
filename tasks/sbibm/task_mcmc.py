import torch
import jax.numpy as jnp
import numpy as np
from numpyro.infer import NUTS, MCMC, Predictive, init_to_value
from numpyro.handlers import condition

from functools import partial
from jax import random

from tasks.sbibm.task import Task


class MCMCTask(Task):
    # task class for SBIBM examples with MCMC posterior sampling
    def __init__(
        self,
        model,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        if "prior_params" in kwargs:
            if kwargs["prior_params"] is not None:
                self.model = partial(model, prior_params=self.prior_params)

    def _simulate_one(self, rng_key, theta, n_obs, **kwargs):
        raise NotImplementedError

    def simulator(self, theta, n_obs=1, rng_key=None, **kwargs):
        # ensure theta is a jax array
        if isinstance(theta, torch.Tensor):
            theta = theta.numpy()
        theta = jnp.array(theta)

        if rng_key is None:
            rng_key = random.PRNGKey(np.random.randint(0, 2**32))

        # simulate with numpyro model
        x = self._simulate_one(rng_key=rng_key, theta=theta, n_obs=n_obs, **kwargs)

        # convert x to torch
        x = torch.from_numpy(np.asarray(x)).float()
        return x

    def sample_reference_posterior(
        self, rng_key, x_star, theta_star, n_obs=1, num_samples=1000, **kwargs
    ):
        # ensure x_star contains n_obs observations
        if x_star.shape[0] > n_obs:
            x_star = x_star[:n_obs]
        assert x_star.shape[0] == n_obs

        # ensure theta_star is a jax array
        if isinstance(theta_star, torch.Tensor):
            theta_star = theta_star.numpy()
        theta_star = jnp.array(theta_star)

        # ensure x_star is a jax array
        if isinstance(x_star, torch.Tensor):
            x_star = x_star.numpy()
        x_star = jnp.array(x_star)

        samples = self._posterior_sampler(
            rng_key=rng_key,
            x_star=x_star,
            theta_star=theta_star,
            n_obs=n_obs,
            num_samples=num_samples,
            **kwargs,
        )

        # convert samples to torch
        samples = torch.from_numpy(np.asarray(samples)).float()
        return samples

    def generate_training_data(self, n_simulations, save=True, n_obs=1, **kwargs):
        kwargs["rng_key"] = None
        return super().generate_training_data(
            n_simulations=n_simulations, save=save, n_obs=n_obs, **kwargs
        )


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
