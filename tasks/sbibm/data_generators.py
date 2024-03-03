# change tasks to match [Geffner et al. 2023]

import pyro
import sbibm
import torch

from tasks.sbibm.gaussianlinear_multi_obs import GaussianLinear_multiobs

# from tasks.sbibm.gaussianmixture_multi_obs import GaussianMixture_multiobs
# from tasks.sbibm.slcp_multi_obs import SLCP_multiobs
# from tasks.sbibm.lotkavolterra_multi_obs import LotkaVolterra_multiobs
# from tasks.sbibm.sir_multi_obs import SIR_multiobs


def get_task(task_name):
    if task_name == "gaussian_linear":
        task = sbibm.get_task("gaussian_linear", prior_scale=1.0)
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
        task.simulator_params = {
            "mixture_locs_factor": torch.tensor([1.0, 1.0]),
            "mixture_scales": torch.tensor([2.25, 1 / 9]),
            "mixture_weights": torch.tensor([0.5, 0.5]),
        }
        return task
    elif task_name in ["lotka_volterra", "sir", "slcp"]:
        # no changes for lotka_volterra and sir
        return sbibm.get_task(task_name)
    else:
        raise ValueError(f"Unknown task {task_name}")


# def get_multiobs_task(task_name):
#     if task_name == "gaussian_linear":
#         return GaussianLinear_multiobs()
#     elif task_name == "gaussian_mixture":
#         return GaussianMixture_multiobs()
#     elif task_name == "lotka_volterra":
#         return LotkaVolterra_multiobs()
#     elif task_name == "slcp":
#         return SLCP_multiobs()
#     elif task_name == "sir":
#         return SIR_multiobs()
#     else:
#         raise ValueError(f"Unknown task {task_name}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    task = get_task("slcp")
    prior = task.get_prior()
    simulator = task.get_simulator()

    print(task.prior_dist.base_dist)
    print(task.prior_dist.base_dist.low)
    print(task.prior_dist.base_dist.high)

    theta = prior(1)
    print(theta.shape)

    x_obs = simulator(theta)
    print(x_obs.shape)

    theta_train = prior(10000)
    low_norm = (
        (task.prior_dist.base_dist.low - theta_train.mean(0)) / theta_train.std(0) * 2
    )
    high_norm = (
        (task.prior_dist.base_dist.high - theta_train.mean(0)) / theta_train.std(0) * 2
    )
    print(low_norm)
    print(high_norm)

    print()
    print(((theta_train - theta_train.mean(0)) / theta_train.std(0)).mean(0))
    print(((theta_train - theta_train.mean(0)) / theta_train.std(0)).std(0))

    # ref_samples = task._sample_reference_posterior(1000, observation=x_obs)
    # plt.scatter(ref_samples[:, 0], ref_samples[:, 1], alpha=0.1)
    # plt.savefig("ref_samples.png")
    # plt.clf()
