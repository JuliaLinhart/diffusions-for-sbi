import torch
import matplotlib.pyplot as plt

from experiment_utils import (
    gaussien_wasserstein,
    count_outliers,
    remove_outliers,
    load_losses,
    dist_to_dirac,
)
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
from plot_utils import set_plotting_style
from ot import sliced_wasserstein_distance

PATH_EXPERIMENT = "results/gaussian/"
DIM_LIST = [2, 4, 8, 16, 32]
N_OBS = [1, 8, 14, 22, 30]
N_TRAIN = 10000
LR = 1e-4
N_EPOCHS = 10000


def load_samples(dim, n_obs, cov_mode="GAUSS", random_prior=False, langevin=False, clip=False):
    path = PATH_EXPERIMENT + f"{dim}d"
    if random_prior:
        path += "random_prior"
    theta_true = torch.load(path + "/theta_true.pkl")
    path += f"/n_train_{N_TRAIN}_n_epochs_{N_EPOCHS}_lr_{LR}/"
    if langevin:
        path += "langevin_steps_400_5/"
    else:
        path += "euler_steps_1000/"
    path += f"posterior_samples_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        path = path[:-4] + "_clip.pkl"
    return torch.load(path), theta_true


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

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
                train_losses, val_losses = load_losses(
                    f"{dim}d", n_train=N_TRAIN, lr=LR, path=PATH_EXPERIMENT
                )
                axs[i, j].plot(train_losses, label="train")
                axs[i, j].plot(val_losses, label="val")
                axs[i, j].set_title(f"dim {dim}, {n_obs} observations")
                axs[i, j].set_xlabel("epochs")
                axs[i, j].set_ylabel("loss")
                axs[i, j].legend()
        plt.savefig(PATH_EXPERIMENT + f"losses_n_train_{N_TRAIN}_lr_{LR}.png")
        plt.clf()

    if args.w_dist:
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        # for j, percentage in enumerate([0, 0.01, 0.05, 0.1]):
        for i, n_obs in enumerate(N_OBS):
            w_gauss, w_langevin, w_jac, w_jac_clip, w_gauss_clip, w_langevin_clip = [], [], [], [], [], []
            for dim in DIM_LIST:
                task = Gaussian_Gaussian_mD(dim=dim)
                x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")[:n_obs]
                true_posterior = task.true_tall_posterior(x_obs)
                samples_analytic = true_posterior.sample((1000,))
                samples_gauss, theta_true = load_samples(dim, n_obs, cov_mode="GAUSS")
                samples_jac, _ = load_samples(dim, n_obs, cov_mode="JAC")
                samples_jac_clip, _ = load_samples(dim, n_obs, cov_mode="JAC", clip=True)
                samples_gauss_clip, _ = load_samples(dim, n_obs, cov_mode="GAUSS", clip=True)
                samples_langevin, _ = load_samples(dim, n_obs, langevin=True)
                samples_langevin_clip, _ = load_samples(dim, n_obs, langevin=True, clip=True)

                w_gauss.append(sliced_wasserstein_distance(samples_analytic, samples_gauss))
                w_jac.append(sliced_wasserstein_distance(samples_analytic, samples_jac))
                w_jac_clip.append(sliced_wasserstein_distance(samples_analytic, samples_jac_clip))
                w_gauss_clip.append(sliced_wasserstein_distance(samples_analytic, samples_gauss_clip))
                w_langevin.append(sliced_wasserstein_distance(samples_analytic, samples_langevin))
                w_langevin_clip.append(sliced_wasserstein_distance(samples_analytic, samples_langevin_clip))

            axs[i].plot(DIM_LIST, w_gauss, label="GAUSS", linewidth=3, marker="o", alpha=0.7, color="blue")
            axs[i].plot(DIM_LIST, w_jac, label="JAC", linewidth=3, marker="o", alpha=0.7, color="orange")
            axs[i].plot(DIM_LIST, w_langevin, label="LANGEVIN", linewidth=3, marker="o", alpha=0.7, color="green")
            axs[i].plot(DIM_LIST, w_gauss_clip, "--", label="GAUSS (clip)", linewidth=3, marker="o", alpha=0.7, color="blue")
            axs[i].plot(DIM_LIST, w_jac_clip, "--", label="JAC (clip)", linewidth=3, marker="o", alpha=0.7, color="orange")
            axs[i].plot(DIM_LIST, w_langevin_clip, "--", label="LANGEVIN (clip)", linewidth=3, marker="o", alpha=0.7, color="green")
            axs[i].set_title(f"{n_obs} observations")  # , {percentage}% outliers rm")
            axs[i].set_xlabel("dim")
            axs[i].set_xticks(DIM_LIST)
            axs[i].set_ylabel("sliced W2")
            axs[i].set_ylim(0, 10)
        axs[0].legend()

        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_train_{N_TRAIN}_lr_{LR}.png")
        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_train_{N_TRAIN}_lr_{LR}.pdf")
        plt.clf()


        # same but with one subfigure per dim and plots as function of n_obs
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        for i, dim in tqdm(enumerate(DIM_LIST)):
            w_gauss, w_langevin, w_jac, w_jac_clip, w_gauss_clip, w_langevin_clip = [], [], [], [], [], []
            for n_obs in N_OBS:
                task = Gaussian_Gaussian_mD(dim=dim)
                x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")[:n_obs]
                true_posterior = task.true_tall_posterior(x_obs)
                samples_analytic = true_posterior.sample((1000,))
                samples_gauss, theta_true = load_samples(dim, n_obs, cov_mode="GAUSS")
                samples_jac, _ = load_samples(dim, n_obs, cov_mode="JAC")
                samples_jac_clip, _ = load_samples(dim, n_obs, cov_mode="JAC", clip=True)
                samples_gauss_clip, _ = load_samples(dim, n_obs, cov_mode="GAUSS", clip=True)
                samples_langevin, _ = load_samples(dim, n_obs, langevin=True)
                samples_langevin_clip, _ = load_samples(dim, n_obs, langevin=True, clip=True)

                w_gauss.append(sliced_wasserstein_distance(samples_analytic, samples_gauss))
                w_jac.append(sliced_wasserstein_distance(samples_analytic, samples_jac))
                w_jac_clip.append(sliced_wasserstein_distance(samples_analytic, samples_jac_clip))
                w_gauss_clip.append(sliced_wasserstein_distance(samples_analytic, samples_gauss_clip))
                w_langevin.append(sliced_wasserstein_distance(samples_analytic, samples_langevin))
                w_langevin_clip.append(sliced_wasserstein_distance(samples_analytic, samples_langevin_clip))
            
            axs[i].plot(N_OBS, w_gauss, label="GAUSS", linewidth=3, marker="o", alpha=0.7, color="blue")
            axs[i].plot(N_OBS, w_jac, label="JAC", linewidth=3, marker="o", alpha=0.7, color="orange")
            axs[i].plot(N_OBS, w_langevin, label="LANGEVIN", linewidth=3, marker="o", alpha=0.7, color="green")
            axs[i].plot(N_OBS, w_gauss_clip, "--", label="GAUSS (clip)", linewidth=3, marker="o", alpha=0.7, color="blue")
            axs[i].plot(N_OBS, w_jac_clip, "--", label="JAC (clip)", linewidth=3, marker="o", alpha=0.7, color="orange")
            axs[i].plot(N_OBS, w_langevin_clip, "--", label="LANGEVIN (clip)", linewidth=3, marker="o", alpha=0.7, color="green")
            axs[i].set_title(f"dim {dim}")
            axs[i].set_xlabel("n_obs")
            axs[i].set_xticks(N_OBS)
            axs[i].set_ylabel("sliced W2")
            axs[i].set_ylim(0, 10)
        axs[0].legend()
        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_train_{N_TRAIN}_lr_{LR}_per_dim.png")
        plt.clf()
            


    dim = 16
    task = Gaussian_Gaussian_mD(dim=dim)

    x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")
    fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    for i, n_obs in enumerate(N_OBS):
        x_obs_ = x_obs[:n_obs]
        true_posterior = task.true_tall_posterior(x_obs_)
        samples_analytic = true_posterior.sample((1000,))
        # samples_jac, _ = load_samples(dim, n_obs, cov_mode="JAC")
        # samples_jac_clip, _ = load_samples(dim, n_obs, cov_mode="JAC_clip")
        # samples, _ = load_samples(dim, n_obs, langevin=True)
        samples_gauss, theta_true = load_samples(dim, n_obs, cov_mode="GAUSS")

        axs[i].scatter(samples_analytic[:, 0], samples_analytic[:, 1], label="Analytic")
        axs[i].scatter(samples_gauss[:, 0], samples_gauss[:, 1], label="GAUSS")
        # axs[i].scatter(samples_jac[:, 0], samples_jac[:, 1], label="JAC")
        # axs[i].scatter(samples_jac_clip[:, 0], samples_jac_clip[:, 1], label="JAC (clip)")
        # axs[i].scatter(samples[:,0], samples[:,1], label="langevin")
        axs[i].scatter(theta_true[0], theta_true[1], label="theta_true", color="black")
        axs[i].set_title(f"{n_obs} observations")
        axs[i].legend()
    plt.savefig(PATH_EXPERIMENT + f"{dim}d/samples.png")
    plt.clf()
