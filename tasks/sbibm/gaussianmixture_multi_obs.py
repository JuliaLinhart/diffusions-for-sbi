from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import condition
import seaborn as sns
import matplotlib.pyplot as plt
import torch

rng_key = random.PRNGKey(1)


def model(x_obs=None, n_obs=1):
    # theta = numpyro.sample("theta", dist.Uniform(
    #     low=-10,
    #     high=+10))
    theta = numpyro.sample("theta", dist.MultivariateNormal(
        loc=jnp.zeros(10),
        covariance_matrix=jnp.eye(10)
    ))
    mixing_dist = dist.Categorical(probs=jnp.ones(2) / 2.)
    component_dists = [dist.MultivariateNormal
                       (loc=theta, covariance_matrix=jnp.eye(10)*2.25),
                       dist.MultivariateNormal
                       (loc=theta, covariance_matrix=jnp.eye(10)/9.)]
    mixture = dist.MixtureGeneral(mixing_dist, component_dists)
    if x_obs is None:
        for i in range(n_obs):
            numpyro.sample(
                f"obs_{i}",
                mixture
            )
    else:
        for i in range(n_obs):
            numpyro.sample(
                f"obs_{i}",
                mixture,
                obs=x_obs[i] #, 0] 
            )

if __name__ == "__main__":
    
    # theta = jnp.array([0.0])
    # predictive = Predictive(
    #     condition(model, {"theta": theta}),
    #     num_samples=1)
    
    from tasks.sbibm.data_generators import get_task
    task = get_task("gaussian_mixture")
    theta = task.get_true_parameters(1)
    simulator = task.get_simulator()
    x_obs_100 = torch.cat([simulator(theta) for _ in range(100)])
    print(x_obs_100.shape)

    samples_mcmc = {}
    for n_obs in [1, 10, 50]:

        # data = predictive(rng_key, n_obs=n_obs)

        # x_obs = jnp.array(
        #     [data[f'obs_{i}'] for i in range(n_obs)]).reshape(n_obs, -1)
        # print(x_obs.shape)
        x_obs = jnp.array(x_obs_100)[:n_obs]

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=10_000)
        mcmc.run(
            rng_key,
            x_obs=x_obs,
            n_obs=n_obs
        )
        samples_mcmc[n_obs] = mcmc.get_samples()['theta']
        print(samples_mcmc[n_obs].shape)

    colors = {1: 'C0', 10: 'C1', 50: 'C2'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for n_obs in [1, 10, 50]:
        sns.kdeplot(
            samples_mcmc[n_obs], lw=3.0, c=colors[n_obs], ax=ax, label=n_obs)
    ax.legend()
    ax.axvline(x=theta, ls='--', c='k')
    # fig.show()
    fig.savefig('gaussianmixture-multi-obs.png')