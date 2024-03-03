import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from jax import random
from jax.experimental.ode import odeint
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from sbibm.tasks.sir.task import SIR
from tqdm import tqdm

rng_key = random.PRNGKey(1)


def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """

    N = 1e6
    S = z[0]
    I = z[1]
    R = z[2]

    beta, gamma = (theta[..., 0], theta[..., 1])

    dS_dt = -beta * S * I * 1.0 / N
    dI_dt = beta * S * I * 1.0 / N - gamma * I
    dR_dt = gamma * I

    return jnp.stack([dS_dt, dI_dt, dR_dt])


def model(x_obs=None, n_obs=1):
    """
    :param int N: number of measurement times in the output
    :param numpy.ndarray x: measured populations with shape (N, 2)
    :param int factor: factor for the number of actual simulated time samples
    """
    # population size
    N = 1e6
    # initial values
    z_init = np.array([N - 1, 1.0, 0.0])
    # measurement times
    ts = jnp.arange(float(160))
    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.LogNormal(
            loc=jnp.array([jnp.log(0.4), jnp.log(0.8)]), scale=jnp.array([0.50, 0.20])
        ),
    )

    # integrate dz/dt, the result will have shape N x 2
    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-8, atol=1e-8, mxstep=1000)
    z = z[::16, :]

    for i in range(n_obs):
        if x_obs is None:
            numpyro.sample(
                f"obs_{i}", dist.Binomial(total_count=1000, probs=z[:, 1] / N)
            )
        else:
            numpyro.sample(
                f"obs_{i}",
                dist.Binomial(total_count=1000, probs=z[:, 1] / N),
                obs=x_obs[i],
            )


class SIR_multiobs(SIR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    def get_simulator(self):
        return self.simulator

    def simulator(self, theta, verbose=True):
        num_samples = theta.shape[0]
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        theta = jnp.array(theta)
        x = []
        for theta in tqdm(theta, disable=not verbose):
            predictive = Predictive(
                condition(self.model, {"theta": theta}), num_samples=1
            )
            data = predictive(rng_key, n_obs=1)
            x.append(data[f"obs_0"])
        x = jnp.array(x)
        x = torch.tensor(np.array(x)).reshape(num_samples, -1)
        return x

    def _sample_reference_posterior_multiobs(self, num_samples, x_obs, theta_true):
        n_obs = x_obs.shape[0]
        x_obs = jnp.array(x_obs)
        theta_true = jnp.array(theta_true)

        kernel = NUTS(
            self.model, init_strategy=init_to_value(None, values={"theta": theta_true})
        )
        mcmc = MCMC(
            kernel,
            num_warmup=1000,
            num_samples=num_samples,
        )
        mcmc.run(rng_key=rng_key, x_obs=x_obs, n_obs=n_obs)

        samples = mcmc.get_samples()["theta"]

        return samples


if __name__ == "__main__":
    theta_true = torch.cat(torch.load("./data-plcr/theta_true_list.pkl")).numpy()
    theta_true = jnp.array(theta_true)

    for num_obs in range(1, 10 + 1):
        theta_star = theta_true[num_obs - 1]

        # generating and saving new observations
        # predictive = Predictive(
        #     condition(model, {"theta": theta_star}),
        #     num_samples=1)
        # data = predictive(rng_key, n_obs=100)
        # x_star = jnp.stack([data[f'obs_{i}'].reshape(-1) for i in range(100)])
        # x_star = torch.from_numpy(np.asarray(x_star)).float()
        # torch.save(x_star, f'./data/x_obs_100_num_{num_obs}_plcr.pkl')

        x_star = torch.load(f"./data/x_obs_100_num_{num_obs}_plcr.pkl")
        x_star = jnp.array(x_star.numpy().astype(np.int32))

        samples_mcmc = {}
        for n_obs in [1, 8, 14, 22, 30]:
            print(f"Generating samples for num_obs={num_obs} and n_obs={n_obs}")

            data = {}
            for i in range(n_obs):
                data[f"obs_{i}"] = x_star[i, :]
            data["theta"] = theta_star

            x_obs = jnp.array([data[f"obs_{i}"] for i in range(n_obs)]).reshape(
                n_obs, -1
            )

            kernel = NUTS(
                model, init_strategy=init_to_value(None, values={"theta": theta_star})
            )
            mcmc = MCMC(
                kernel,
                num_warmup=1000,
                num_samples=10_000,
            )
            mcmc.run(rng_key=rng_key, x_obs=x_obs, n_obs=n_obs)

            samples_mcmc[n_obs] = mcmc.get_samples()["theta"]

            filename = (
                f"./samples-plcr/true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
            )
            samples = np.asarray(samples_mcmc[n_obs])
            samples = torch.from_numpy(samples).float()
            torch.save(samples, filename)

    # idx_theta = 0
    # colors = {1: 'C0', 10: 'C1', 50: 'C2'}
    # fig, ax = plt.subplots(figsize=(8, 6))
    # for n_obs in [1]:
    #     sns.kdeplot(
    #         samples_mcmc[n_obs][:, idx_theta], ax=ax, c=colors[n_obs], label=n_obs)
    # ax.axvline(x=theta_star[idx_theta], c='k', ls='--')
    # ax.legend()
    # fig.show()

    prior = dist.LogNormal(
        loc=jnp.array([jnp.log(0.4), jnp.log(0.125)]), scale=jnp.array([0.50, 0.20])
    )
    theta_df = prior.sample(rng_key, (30_000,))
    x_df = []
    for theta_i in tqdm(theta_df):
        predictive = Predictive(condition(model, {"theta": theta_i}), num_samples=1)
        data = predictive(rng_key, n_obs=1)
        xi = data["obs_0"].reshape(-1)
        x_df.append(xi)
    x_df = jnp.stack(x_df)
    x_df = np.asarray(x_df)
    x_df = torch.from_numpy(x_df)
    theta_df = np.asarray(theta_df)
    theta_df = torch.from_numpy(theta_df)
    data = {}
    data["theta"] = theta_df
    data["x"] = x_df
    torch.save(data, "./data-plcr/training_data_30_000.pkl")
