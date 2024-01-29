import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_uniform


rng_key = random.PRNGKey(1)


def model(x_obs=None, n_obs=1):
    theta_1 = numpyro.sample("theta_1", dist.Uniform(low=-3.0, high=+3.0))
    theta_2 = numpyro.sample("theta_2", dist.Uniform(low=-3.0, high=+3.0))
    theta_3 = numpyro.sample("theta_3", dist.Uniform(low=-3.0, high=+3.0))
    theta_4 = numpyro.sample("theta_4", dist.Uniform(low=-3.0, high=+3.0))
    theta_5 = numpyro.sample("theta_5", dist.Uniform(low=-3.0, high=+3.0))

    m = jnp.array([theta_1, theta_2])
    s1 = theta_3**2
    s2 = theta_4**2
    rho = jnp.tanh(theta_5)
    S = jnp.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])

    if x_obs is None:
        for i in range(n_obs):
            numpyro.sample(
                name=f"obs_{i}",
                fn=dist.MultivariateNormal(m, S),
                sample_shape=(4,)
            )
    else:
        for i in range(n_obs):
            numpyro.sample(
                f"obs_{i}",
                dist.MultivariateNormal(m, S),
                sample_shape=(4,),
                obs=x_obs[f"obs_{i}"]
            )


if __name__ == "__main__":
    import torch
    import sbibm

    n_obs_list = [1, 10, 50]
    theta_1 = -1.0
    theta_2 = -2.0
    theta_3 = -1.0
    theta_4 = -1.0
    theta_5 = +1.0

    predictive = Predictive(
        condition(
            model,
            {
                "theta_1": theta_1,
                "theta_2": theta_2,
                "theta_3": theta_3,
                "theta_4": theta_4,
                "theta_5": theta_5}
            ),
        num_samples=1)
    data = predictive(rng_key, n_obs=1)

    task = sbibm.get_task('slcp')
    simulator = task.get_simulator()
    theta = torch.tensor([theta_1, theta_2, theta_3, theta_4, theta_5]).reshape(1, -1)
    print(theta.shape)
    # data = simulator(theta)
    x_obs_100 = torch.cat([task.unflatten_data(simulator(theta)) for _ in range(100)])
    print(x_obs_100.shape)

    samples_mcmc = {}
    for n_obs in n_obs_list:
        data = predictive(rng_key, n_obs=n_obs)
        print(data[f'obs_0'].shape)
        data = {f'obs_{i}': jnp.array(x_obs_100[i,:].unsqueeze(0)) for i in range(n_obs)}
        print(data[f'obs_0'].shape)
        kernel = NUTS(
            model,
            init_strategy=init_to_uniform(None, radius=3))
        mcmc = MCMC(kernel, num_warmup=100, num_samples=2500, num_chains=4)
        mcmc.run(
            rng_key,
            x_obs=data,
            n_obs=n_obs
        )
        samples_mcmc[n_obs] = jnp.stack(
            [mcmc.get_samples()[f'theta_{i+1}'] for i in range(5)])


    colors = {1: 'C0', 10: 'C1', 50: 'C2'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for n_obs in n_obs_list:
        sns.kdeplot(samples_mcmc[n_obs][3, :],
                    bw_method=0.5,
                    lw=3.0,
                    ax=ax,
                    label='mcmc',
                    color=colors[n_obs])
    ax.axvline(x=theta_4, ls='--', c='k')
    ax.set_xlim(-3.5, 3.5)
    ax.legend()
    # fig.show()
    fig.savefig('slcp-pyro.png')
