# change tasks to match [Geffner et al. 2023]

import torch
import pyro
import sbibm

# from jax import random
# import jax.numpy as jnp
# from numpyro.infer import MCMC, NUTS

from tasks.sbibm.gaussianlinear_multi_obs import GausianLinear_multiobs
# from tasks.sbibm.gaussianmixture_multi_obs import model

def get_task(task_name):
    if task_name == "gaussian_linear":
        task = sbibm.get_task("gaussian_linear", prior_scale=1.0)
        for i,s in enumerate(task.observation_seeds):
            # if i == 5:
            #     s = 1000020
            #     task.observation_seeds[i] = s
            torch.manual_seed(s)
            true_parameters = task.get_prior()(1)
            task._save_true_parameters(i+1, true_parameters)
        # Σ is a diagonal matrix with elements increasing linearly from 0.6 to 1.4
        cov = torch.diag(torch.linspace(0.6, 1.4, 10))
        task.simulator_params = {
            "precision_matrix": torch.inverse(cov),
        }
        return task
    elif task_name == "gaussian_mixture":
        task = sbibm.get_task("gaussian_mixture", dim=10)
        task.prior_dist = pyro.distributions.MultivariateNormal(
            torch.zeros(10), torch.eye(10)
        )
        for i,s in enumerate(task.observation_seeds):
            torch.manual_seed(s)
            true_parameters = task.get_prior()(1)
            task._save_true_parameters(i+1, true_parameters)
        task.simulator_params = {
            "mixture_locs_factor": torch.tensor([1.0, 1.0]),
            "mixture_scales": torch.tensor([2.25, 1/9]),
            "mixture_weights": torch.tensor([0.5, 0.5]),
        }
        return task
    elif task_name in ["sir", "lotka_volterra"]:
        # no changes for lotka_volterra and sir
        return sbibm.get_task(task_name)
    else:
        raise ValueError(f"Unknown task {task_name}")
    
def get_tall_posterior_samples(task_name, x_obs, num_samples=1000, save_path='/'):
    if task_name == "gaussian_linear":
        task = GausianLinear_multiobs(prior_scale=1., simulator_scale=torch.linspace(0.6, 1.4, 10))
        samples = task._sample_reference_posterior_multiobs(num_samples, x_obs)
    # elif task_name == "gaussian_mixture":
    #     x_obs = jnp.array(x_obs)
    #     kernel = NUTS(model)
    #     mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    #     mcmc.run(
    #         rng_key=random.PRNGKey(1),
    #         x_obs=x_obs,
    #         n_obs=x_obs.shape[0]
    #     )
    #     samples = mcmc.get_samples()["theta"]
    else:
        raise NotImplementedError
    
    return samples
        
