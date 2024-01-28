import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns


rng_key = random.PRNGKey(1)


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


def model(x_obs=None, n_obs=1, n_time_samples=20, factor=10):
    """
    :param int N: number of measurement times in the output
    :param numpy.ndarray x: measured populations with shape (N, 2)
    :param int factor: factor for the number of actual simulated time samples
    """
    # initial population
    z_init = np.array([30.0, 1.0])
    # measurement times
    ts = jnp.arange(factor*float(n_time_samples))
    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.LogNormal(loc=jnp.array([-0.125, -3.0, -0.125, -3.0]),
                       scale=jnp.array([0.50, 0.50, 0.50, 0.50]))
    )

    # integrate dz/dt, the result will have shape N x 2
    z = odeint(
        dz_dt,
        z_init,
        ts,
        theta,
        rtol=1e-8,
        atol=1e-8,
        mxstep=1000
    )
    z = z[::factor, :]

    for i in range(n_obs):
        # sigma = 0.10
        sigma_i = numpyro.sample(f"sigma_{i}", dist.LogNormal(-1, 1).expand([2]))
        loc = jnp.log(z)
        if x_obs is None:
            numpyro.sample(
                f"obs_{i}",
                dist.LogNormal(loc, scale=sigma_i)
            )
        else:
            numpyro.sample(
                f"obs_{i}",
                dist.LogNormal(loc, scale=sigma_i), obs=x_obs[i]
            )


# the output of the simulator has always 20 dimensions, but the time series are
# actually generated for 20*factor time samples (factor=10 in sbibm)
factor = 3

theta = jnp.array([1.14, 0.03, 1.00, 0.035])
predictive = Predictive(
    condition(model, {"theta": theta}),
    num_samples=1)

samples_mcmc = {}
for n_obs in [1, 5]:

    data = predictive(rng_key, n_obs=n_obs)

    x_obs = jnp.array(
        [data[f'obs_{i}'] for i in range(n_obs)]).reshape(n_obs, -1, 2)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=100,
        num_samples=1000
    )
    mcmc.run(
        rng_key=rng_key,
        x_obs=x_obs,
        n_obs=n_obs,
        factor=factor)

    samples_mcmc[n_obs] = mcmc.get_samples()['theta']

colors = {1: 'C0', 10: 'C1', 50: 'C2'}
fig, ax = plt.subplots(figsize=(8, 6))
for n_obs in [1, 10, 50]:
    sns.kdeplot(samples_mcmc[n_obs][:, 0], ax=ax, c=colors[n_obs])
fig.show()
