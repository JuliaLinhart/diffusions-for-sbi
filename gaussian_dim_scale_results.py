import torch
import matplotlib.pyplot as plt

from experiment_utils import gaussien_wasserstein, count_outliers, remove_outliers
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD

PATH_EXPERIMENT = "results/gaussian/"
DIM_LIST = [2,4,8,16,32]
N_OBS = [1,8,14,22,30]

def load_samples(dim, n_obs, cov_mode="GAUSS", random_prior=False, langevin=False):
    path = PATH_EXPERIMENT + f"{dim}d"
    if random_prior:
        path += "random_prior"
    theta_true = torch.load(path + "/theta_true.pkl")
    path += f"/n_train_30000_n_epochs_1000_lr_{1e-3}/"
    if langevin:
        path += "langevin_steps_400_5/"
    else:
        path += "euler_steps_1000/"
    path += f"posterior_samples_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    return torch.load(path), theta_true



fig, axs = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True)
for j, threshold in enumerate([100, 1000]):
    for i, n_obs in enumerate(N_OBS):
        w_gauss, w_langevin = [], []
        for dim in DIM_LIST:
            task = Gaussian_Gaussian_mD(dim=dim)
            x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")[:n_obs]
            samples_analytic = task.true_tall_posterior(x_obs).sample((1000,))
            samples_langevin, theta_true = load_samples(dim, n_obs, langevin=True)
            samples_gauss, _ = load_samples(dim, n_obs, cov_mode="GAUSS")

            # # outliers
            # outliers_langevin = count_outliers(samples_langevin, theta_true)
            # outliers_gauss = count_outliers(samples_gauss, theta_true)
            # print(f"outliers langevin: {outliers_langevin}")
            # print(f"outliers gauss: {outliers_gauss}")

            # remove outliers
            samples_langevin = remove_outliers(samples_langevin, theta_true, threshold=threshold)
            samples_gauss = remove_outliers(samples_gauss, theta_true, threshold=threshold)

            w_gauss.append(gaussien_wasserstein(samples_analytic.unsqueeze(0), samples_gauss.unsqueeze(0)))
            w_langevin.append(gaussien_wasserstein(samples_analytic.unsqueeze(0), samples_langevin.unsqueeze(0)))
        axs[j,i].plot(DIM_LIST, w_gauss, label="gauss")
        axs[j,i].plot(DIM_LIST, w_langevin, label="langevin")
        axs[j,i].set_title(f"{n_obs} observations, threshold={threshold}")
        axs[j,i].set_xlabel("dim")
        axs[j,i].set_xticks(DIM_LIST)
        axs[j,i].set_ylabel("W2")
        axs[j,i].legend()

plt.savefig(PATH_EXPERIMENT + f"gaussian_wasserstein.png")
plt.savefig(PATH_EXPERIMENT + f"gaussian_wasserstein.pdf")
plt.clf()

# plt.scatter(samples_analytic[:,0], samples_analytic[:,1], label="analytic")
# plt.scatter(samples[:,0], samples[:,1], label="langevin")
# plt.scatter(samples_gauss[:,0], samples_gauss[:,1], label="gauss")
# plt.legend()
# plt.savefig(PATH_EXPERIMENT + f"{dim}d/samples.png")
# plt.clf()