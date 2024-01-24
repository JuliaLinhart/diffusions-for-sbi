import torch
import matplotlib.pyplot as plt

from experiment_utils import gaussien_wasserstein, count_outliers, remove_outliers, load_losses, dist_to_dirac
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
from plot_utils import set_plotting_style

PATH_EXPERIMENT = "results/gaussian/"
DIM_LIST = [2,4,8,16,32]
N_OBS = [1,8,14,22,30]
N_TRAIN = 10000
LR = 1e-4

def load_samples(dim, n_obs, cov_mode="GAUSS", random_prior=False, langevin=False):
    path = PATH_EXPERIMENT + f"{dim}d"
    if random_prior:
        path += "random_prior"
    theta_true = torch.load(path + "/theta_true.pkl")
    path += f"/n_train_{N_TRAIN}_n_epochs_1000_lr_{LR}/"
    if langevin:
        path += "langevin_steps_400_5/"
    else:
        path += "euler_steps_1000/"
    path += f"posterior_samples_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    return torch.load(path), theta_true

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--w_dist", action="store_true")

    args = parser.parse_args()

    set_plotting_style()

    if args.losses:
        # plot losses
        fig, axs = plt.subplots(5, 5, figsize=(25, 25), constrained_layout=True)
        for i, dim in enumerate(DIM_LIST):
            for j, n_obs in enumerate(N_OBS):
                train_losses, val_losses = load_losses(f"{dim}d", n_train=N_TRAIN, lr=LR, path=PATH_EXPERIMENT)
                axs[i,j].plot(train_losses, label="train")
                axs[i,j].plot(val_losses, label="val")
                axs[i,j].set_title(f"dim {dim}, {n_obs} observations")
                axs[i,j].set_xlabel("epochs")
                axs[i,j].set_ylabel("loss")
                axs[i,j].legend()
        plt.savefig(PATH_EXPERIMENT + f"losses_n_train_{N_TRAIN}_lr_{LR}.png")
        plt.clf()

    if args.w_dist:
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        # for j, percentage in enumerate([0, 0.01, 0.05, 0.1]):
        for i, n_obs in enumerate(N_OBS):
            w_gauss, w_langevin = [], []
            for dim in DIM_LIST:
                task = Gaussian_Gaussian_mD(dim=dim)
                x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")[:n_obs]
                samples_analytic = task.true_tall_posterior(x_obs).sample((1000,))
                samples_langevin, theta_true = load_samples(dim, n_obs, langevin=True)
                samples_gauss, _ = load_samples(dim, n_obs, cov_mode="GAUSS")

                # # remove outliers
                # samples_langevin = remove_outliers(samples_langevin, theta_true, percentage=0)
                # samples_gauss = remove_outliers(samples_gauss, theta_true, percentage=0)

                # # outliers
                # outliers_langevin = count_outliers(samples_langevin, theta_true)
                # outliers_gauss = count_outliers(samples_gauss, theta_true)
                # print(f"outliers langevin: {outliers_langevin}")
                # print(f"outliers gauss: {outliers_gauss}")

                w_gauss.append(gaussien_wasserstein(samples_analytic.unsqueeze(0), samples_gauss.unsqueeze(0)))
                w_langevin.append(gaussien_wasserstein(samples_analytic.unsqueeze(0), samples_langevin.unsqueeze(0)))
                # w_gauss.append(dist_to_dirac(samples_gauss, theta_true, percentage=percentage)["mmd"])
                # w_langevin.append(dist_to_dirac(samples_langevin, theta_true, percentage=percentage)["mmd"])
            axs[i].plot(DIM_LIST, w_gauss, label="GAUSS")
            axs[i].plot(DIM_LIST, w_langevin, label="LANGEVIN")
            axs[i].set_title(f"{n_obs} observations") #, {percentage}% outliers rm")
            axs[i].set_xlabel("dim")
            axs[i].set_xticks(DIM_LIST)
            axs[i].set_ylabel("W2")
            axs[i].set_ylim(0, 2)
            axs[i].legend()

        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_train_{N_TRAIN}_lr_{LR}.png")
        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_train_{N_TRAIN}_lr_{LR}.pdf")
        plt.clf()

        # dim = 2
        # task = Gaussian_Gaussian_mD(dim=dim)

        # x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")
        # fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        # for i, n_obs in enumerate(N_OBS):
        #     x_obs_ = x_obs[:n_obs]
        #     samples_analytic = task.true_tall_posterior(x_obs_).sample((1000,))
        #     samples, _ = load_samples(dim, n_obs, langevin=True)
        #     samples_gauss, theta_true = load_samples(dim, n_obs, cov_mode="GAUSS")
        #     axs[i].scatter(samples_analytic[:,0], samples_analytic[:,1], label="analytic")
        #     axs[i].scatter(samples[:,0], samples[:,1], label="langevin")
        #     axs[i].scatter(samples_gauss[:,0], samples_gauss[:,1], label="gauss")
        #     axs[i].scatter(theta_true[0], theta_true[1], label="theta_true", color="black")
        #     axs[i].set_title(f"{n_obs} observations")
        #     axs[i].legend()
        # plt.savefig(PATH_EXPERIMENT + f"{dim}d/samples_new.png")
        # plt.clf()