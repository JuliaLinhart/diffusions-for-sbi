import torch
import matplotlib.pyplot as plt

from experiment_utils import (
    gaussien_wasserstein,
    count_outliers,
    remove_outliers,
    dist_to_dirac,
)
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
from plot_utils import set_plotting_style, METHODS_STYLE, METRICS_STYLE
from ot import sliced_wasserstein_distance
from sbibm.metrics import mmd

PATH_EXPERIMENT = "results/gaussian/"
N_OBS = [1, 8, 14, 22, 30]
N_TRAIN = 10000
N_EPOCHS = 10000
DIM_LR_DICT = {
    2: 1e-4, 
    4: 1e-4, 
    8: 1e-4, 
    16: 1e-4, 
    32: 1e-3}

METRICS = ["swd", "mmd", "mmd_to_dirac"]

def load_losses(task_name, n_train, lr, path):
    losses = torch.load(path + f'{task_name}/n_train_{n_train}_n_epochs_{N_EPOCHS}_lr_{lr}/train_losses.pkl')
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    best_epoch = losses["best_epochs"]
    return train_losses, val_losses, best_epoch

def path_to_results(dim, n_obs, lr, cov_mode=None, random_prior=False, langevin=False, clip=False):
    path = PATH_EXPERIMENT + f"{dim}d"
    if random_prior:
        path += "random_prior"
    theta_true_path = path + "/theta_true.pkl"
    path += f"/n_train_{N_TRAIN}_n_epochs_{N_EPOCHS}_lr_{lr}/"
    path = path + "langevin_steps_400_5/" if langevin else path + "euler_steps_1000/"
    runtimes_path = path + f"time_n_obs_{n_obs}.pkl"
    samples_path = path + f"posterior_samples_n_obs_{n_obs}.pkl"
    if not langevin:
        runtimes_path = runtimes_path[:-4] + f"_{cov_mode}.pkl"
        samples_path = samples_path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        runtimes_path = runtimes_path[:-4] + "_clip.pkl"
        samples_path = samples_path[:-4] + "_clip.pkl"
    return samples_path, theta_true_path, runtimes_path

def load_runtimes(dim, n_train, lr, cov_mode=None, random_prior=False, langevin=False, clip=False):
    return torch.load(path_to_results(dim, n_train, lr, cov_mode, random_prior, langevin, clip)[-1])

def load_samples(dim, n_obs, lr, cov_mode=None, random_prior=False, langevin=False, clip=False):
    path = path_to_results(dim, n_obs, lr, cov_mode, random_prior, langevin, clip)
    return torch.load(path[0]), torch.load(path[1])


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--runtimes", action="store_true")
    parser.add_argument("--w_dist", action="store_true")

    args = parser.parse_args()

    set_plotting_style()

    if args.losses:
        # plot losses
        lr_list = [1e-4, 1e-3]
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True, sharey=True)
        for i, dim in enumerate(DIM_LR_DICT.keys()):
            best_val_loss = {}
            for lr_, c in zip(lr_list, ["blue", "orange"]):
                train_losses, val_losses, best_epoch = load_losses(
                    f"{dim}d", n_train=N_TRAIN, lr=lr_, path=PATH_EXPERIMENT
                )
                best_val_loss[lr_] = val_losses[best_epoch]

                # axs[i, j].plot(train_losses, label=f"train") #, lr={LR}")
                axs[i].plot(val_losses, label=f"val, lr={lr_}", color=c)
                axs[i].axvline(best_epoch, color=c, linestyle="--", linewidth=3)
            # get lr for min best val loss
            best_lr = min(best_val_loss, key=best_val_loss.get)
            axs[i].set_title(fr"${dim}$D" + f" \n best_lr={best_lr}")
            axs[i].set_xlabel("epochs")
            axs[i].set_ylabel("loss")
            axs[i].legend()
        plt.savefig(PATH_EXPERIMENT + f"losses_n_train_{N_TRAIN}.png")
        plt.savefig(PATH_EXPERIMENT + f"losses_n_train_{N_TRAIN}.pdf")
        plt.clf()

    if args.runtimes:
        #plot runtimes for every dim as a function of dim, then as a function of n_obs
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        for j, n_obs in enumerate(N_OBS):
            runtimes_dict = {method: [] for method in METHODS_STYLE.keys()}
            for dim, lr in DIM_LR_DICT.items():
                for method in METHODS_STYLE.keys():
                    clip = True if "clip" in method else False
                    langevin = True if "LANGEVIN" in method else False
                    runtimes_dict[method].append(load_runtimes(dim, n_obs, lr, cov_mode=method.split("_")[0], langevin=langevin, clip=clip))

            for k, v in runtimes_dict.items():
                axs[j].plot(list(DIM_LR_DICT.keys()), v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
                axs[j].set_title(rf"$n={n_obs}$")
                axs[j].set_xlabel(r"Dimenson ($m$)")
                axs[j].set_xscale("log", base=2)
                axs[j].set_xticks(list(DIM_LR_DICT.keys()))
                axs[j].set_ylabel("runtimes (s)")
                # axs[0, j].set_ylim(0, 1)
        
        # for j, (dim, lr) in enumerate(DIM_LR_DICT.items()):
        #     runtimes_dict = {method: [] for method in METHODS_STYLE.keys()}
        #     for n_obs in N_OBS:
        #         for method in METHODS_STYLE.keys():
        #             clip = True if "clip" in method else False
        #             langevin = True if "LANGEVIN" in method else False
        #             runtimes_dict[method].append(load_runtimes(dim, n_obs, lr, cov_mode=method.split("_")[0], langevin=langevin, clip=clip))

        #     for k, v in runtimes_dict.items():
        #         axs[1, j].plot(N_OBS, v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
        #         axs[1, j].set_title(fr"${dim}$D")
        #         axs[1, j].set_xlabel(r"$n$")
        #         axs[1, j].set_xticks(N_OBS)
        #         axs[1, j].set_ylabel("runtimes (s)")
        #         # axs[1, j].set_ylim(0, 1)
        # axs[1, 4].legend()
                

        plt.savefig(PATH_EXPERIMENT + f"runtimes.png")
        plt.savefig(PATH_EXPERIMENT + f"runtimes.pdf")
        plt.clf()


    if args.w_dist:
        fig, axs = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True)
        for j, n_obs in enumerate(N_OBS):
            wdist_dict = {method: [] for method in METHODS_STYLE.keys()}
            mmd_dict = {method: [] for method in METHODS_STYLE.keys()}
            # mmd_to_dirac_dict = {method: [] for method in METHODS_STYLE.keys()}
            for dim, lr in DIM_LR_DICT.items():
                task = Gaussian_Gaussian_mD(dim=dim)
                x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")[:n_obs]
                theta_true = torch.load(PATH_EXPERIMENT + f"{dim}d/theta_true.pkl")
                true_posterior = task.true_tall_posterior(x_obs)
                samples_analytic = true_posterior.sample((1000,))
                for method in METHODS_STYLE.keys():
                    clip = True if "clip" in method else False
                    langevin = True if "LANGEVIN" in method else False
                    samples, theta_true = load_samples(dim, n_obs, lr, cov_mode=method.split("_")[0], langevin=langevin, clip=clip)
                    wdist_dict[method].append(sliced_wasserstein_distance(samples_analytic, samples, n_projections=100))
                    mmd_dict[method].append(mmd(samples_analytic, samples))
                    # mmd_to_dirac_dict[method].append(dist_to_dirac(samples, theta_true, metrics=["mmd"])["mmd"])

            for k, v in wdist_dict.items():
                axs[0, j].plot(list(DIM_LR_DICT.keys()), v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
            for k, v in mmd_dict.items():
                axs[1, j].plot(list(DIM_LR_DICT.keys()), v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
            # for k, v in mmd_to_dirac_dict.items():
            #     axs[2, j].plot(list(DIM_LR_DICT.keys()), v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
            for i, metric in enumerate(["swd", "mmd"]):
                axs[i, j].set_title(rf"$n={n_obs}$")
                axs[i, j].set_xlabel(r"Dimenson ($m$)")
                axs[i, j].set_xscale("log", base=2)
                axs[i, j].set_xticks(list(DIM_LR_DICT.keys()))
                axs[i, j].set_ylabel(METRICS_STYLE[metric]["label"])
                axs[i, j].set_ylim(0, 0.8)
        axs[i, j].legend()
        

        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_obs.png")
        plt.savefig(PATH_EXPERIMENT + f"wasserstein_n_obs.pdf")
        plt.clf()


        # same but with one subfigure per dim and plots as function of n_obs
        fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
        for j, (dim, lr) in enumerate(DIM_LR_DICT.items()):
            wdist_dict = {method: [] for method in METHODS_STYLE.keys()}
            mmd_dict = {method: [] for method in METHODS_STYLE.keys()}
            mmd_to_dirac_dict = {method: [] for method in METHODS_STYLE.keys()}
            for _, n_obs in enumerate(N_OBS):
                task = Gaussian_Gaussian_mD(dim=dim)
                x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")[:n_obs]
                theta_true = torch.load(PATH_EXPERIMENT + f"{dim}d/theta_true.pkl")
                true_posterior = task.true_tall_posterior(x_obs)
                samples_analytic = true_posterior.sample((1000,))
                for method in METHODS_STYLE.keys():
                    clip = True if "clip" in method else False
                    langevin = True if "LANGEVIN" in method else False
                    samples, theta_true = load_samples(dim, n_obs, lr, cov_mode=method.split("_")[0], langevin=langevin, clip=clip)
                    # wdist_dict[method].append(sliced_wasserstein_distance(samples_analytic, samples, n_projections=100))
                    # mmd_dict[method].append(mmd(samples_analytic, samples))
                    mmd_to_dirac_dict[method].append(dist_to_dirac(samples, theta_true, metrics=["mmd"])["mmd"])

            # for k, v in wdist_dict.items():
            #     axs[0, j].plot(N_OBS, v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
            # for k, v in mmd_dict.items():
            #     axs[1, j].plot(N_OBS, v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
            for k, v in mmd_to_dirac_dict.items():
                axs[j].plot(N_OBS, v, linewidth=3, alpha=0.7, **METHODS_STYLE[k])
            # for i, metric in enumerate(METRICS):
            axs[j].set_title(fr"${dim}$D")
            axs[j].set_xlabel(r"$n$")
            axs[j].set_xticks(N_OBS)
            axs[j].set_ylabel(METRICS_STYLE["mmd_to_dirac"]["label"])
            axs[j].set_ylim(0, 0.8)
        axs[j].legend()

        plt.savefig(PATH_EXPERIMENT + f"wasserstein_per_dim.png")
        plt.savefig(PATH_EXPERIMENT + f"wasserstein_per_dim.pdf")
        plt.clf()
            


    # dim = 16
    # task = Gaussian_Gaussian_mD(dim=dim)

    # x_obs = torch.load(PATH_EXPERIMENT + f"{dim}d/x_obs_100.pkl")
    # fig, axs = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    # for i, n_obs in enumerate(N_OBS):
    #     x_obs_ = x_obs[:n_obs]
    #     true_posterior = task.true_tall_posterior(x_obs_)
    #     samples_analytic = true_posterior.sample((1000,))
    #     # samples_jac, _ = load_samples(dim, n_obs, cov_mode="JAC")
    #     # samples_jac_clip, _ = load_samples(dim, n_obs, cov_mode="JAC_clip")
    #     # samples, _ = load_samples(dim, n_obs, langevin=True)
    #     samples_gauss, theta_true = load_samples(dim, n_obs, cov_mode="GAUSS")

    #     axs[i].scatter(samples_analytic[:, 0], samples_analytic[:, 1], label="Analytic")
    #     axs[i].scatter(samples_gauss[:, 0], samples_gauss[:, 1], label="GAUSS")
    #     # axs[i].scatter(samples_jac[:, 0], samples_jac[:, 1], label="JAC")
    #     # axs[i].scatter(samples_jac_clip[:, 0], samples_jac_clip[:, 1], label="JAC (clip)")
    #     # axs[i].scatter(samples[:,0], samples[:,1], label="langevin")
    #     axs[i].scatter(theta_true[0], theta_true[1], label="theta_true", color="black")
    #     axs[i].set_title(f"{n_obs} observations")
    #     axs[i].legend()
    # plt.savefig(PATH_EXPERIMENT + f"{dim}d/samples.png")
    # plt.clf()
