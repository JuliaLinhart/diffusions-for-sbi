import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpyro
import numpyro.distributions as dist
from jax import random
import torch

from tasks.sbibm.task_mcmc import MCMCTask, get_mcmc_samples, get_predictive_sample


def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u = z[0]
    v = z[1]
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def model(
    prior_params={
        "loc": [-0.125, -3.0, -0.125, -3.0],
        "scale": [0.50, 0.50, 0.50, 0.50],
    },
    x_obs=None,
    n_obs=1,
    summary=None,
):
    """
    :param int N: number of measurement times in the output
    :param numpy.ndarray x: measured populations with shape (N, 2)
    :param int factor: factor for the number of actual simulated time samples
    """
    # initial population
    z_init = np.array([30.0, 1.0])
    # measurement times
    ts = jnp.arange(0, 20 + 0.1, 0.1)
    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.LogNormal(
            loc=jnp.array(prior_params["loc"]), scale=jnp.array(prior_params["scale"])
        ),
    )

    # integrate dz/dt, the result will have shape N x 2
    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-8, atol=1e-8, mxstep=1000)

    if summary is None:
        z_sub = z
        sigma_i = 0.001
    elif summary == "subsample":
        z_sub = z[::21, :].T.reshape(-1)
        sigma_i = 0.1

    for i in range(n_obs):
        if x_obs is None:
            numpyro.sample(
                f"obs_{i}",
                dist.LogNormal(
                    jnp.log(jnp.clip(z_sub, 1e-10, 10000)), scale=sigma_i
                ),  # noqa
            )
        else:
            numpyro.sample(
                f"obs_{i}",
                dist.LogNormal(jnp.log(jnp.clip(z_sub, 1e-10, 10000)), scale=sigma_i),
                obs=x_obs[i],
            )


class LotkaVolterra(MCMCTask):
    def __init__(
        self,
        prior_params={
            "loc": [-0.125, -3.0, -0.125, -3.0],
            "scale": [0.50, 0.50, 0.50, 0.50],
        },
        summary="subsample",
        **kwargs,
    ):
        super().__init__(
            name="lotka_volterra", prior_params=prior_params, model=model, **kwargs
        )

        self.summary = summary

    def prior(self):
        return torch.distributions.LogNormal(
            loc=torch.tensor(self.prior_params["loc"]),
            scale=torch.tensor(self.prior_params["scale"]),
        )

    def _simulate_one(self, rng_key, theta, n_obs):
        x = get_predictive_sample(
            rng_key=rng_key,
            model=self.model,
            cond={"theta": theta},
            n_obs=n_obs,
            summary=self.summary,
        )
        return x  # shape (n_obs, dim_x=20)

    def _posterior_sampler(self, rng_key, x_star, theta_star, n_obs, num_samples):
        samples = get_mcmc_samples(
            rng_key=rng_key,
            model=self.model,
            init_value={"theta": theta_star},
            data=x_star,
            n_obs=n_obs,
            num_samples=num_samples,
            summary=self.summary,
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

    lv = LotkaVolterra(save_path=args.save_path)
    os.makedirs(lv.save_path, exist_ok=True)

    if args.train_data:
        data = lv.generate_training_data(rng_key=rng_key, n_simulations=50_000)
        print(data["theta"].shape, data["x"].shape)

    if args.reference_data:
        ref_data = lv.generate_reference_data(rng_key=rng_key)
        print(ref_data[0].shape, ref_data[1][1].shape)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                samples = lv.generate_reference_posterior_samples(
                    rng_key=rng_key, num_obs=num_obs, n_obs=n_obs, num_samples=1_000
                )
                print(samples.shape)

    # # simulate one check
    # theta = lv.prior().sample((1,))
    # x = lv.simulator(rng_key, theta, n_obs=8)
    # print(x.shape, theta.shape)
