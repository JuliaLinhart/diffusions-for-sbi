# change tasks to match [Geffner et al. 2023]

import torch
import pyro
import sbibm

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