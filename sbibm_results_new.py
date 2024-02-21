import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from tasks.sbibm.data_generators import get_task

from experiment_utils import dist_to_dirac
from ot import sliced_wasserstein_distance
from sbibm. metrics import mmd, c2st

from matplotlib import colormaps as cm
from plot_utils import set_plotting_style, METHODS_STYLE, METRICS_STYLE, pairplot_with_groundtruth_md

from tqdm import tqdm

PATH_EXPERIMENT = "results/sbibm/"
TASKS = {
    # "gaussian_linear": "Gaussian Linear",
    # "gaussian_mixture": "Gaussian Mixture",
    "slcp_good": "SLCP",
    "lotka_volterra_good": "Lotka-Volterra",
    "sir_good": "SIR",
}
N_TRAIN = [1000, 3000, 10000, 30000, 50000]
BATCH_SIZE = 256 # 64
N_EPOCHS = 5000 
LR = 1e-4

TASKS_DICT = {
    "slcp_good": {"lr": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256, 256]},
    "lotka_volterra_good": {"lr": [1e-3, 1e-3, 1e-3, 1e-4, 1e-4], "bs": [256, 256, 256, 256, 256]},
    "sir_good": {"lr": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4], "bs": [64, 64, 64, 64, 64]},
}

N_OBS = [1, 8, 14, 22, 30]
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #list(np.arange(1,26)) 

def load_losses(task_name, n_train, lr, path, n_epochs=N_EPOCHS):
    batch_size = BATCH_SIZE
    if task_name == "lotka_volterra_good":
        path = path + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}_new_log/"
    else:
        path = path + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}_new/"
    losses = torch.load(path + f'train_losses.pkl')
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    best_epoch = losses["best_epoch"]
    return train_losses, val_losses, best_epoch

def path_to_results(task_name, result_name, num_obs, n_train, lr, n_obs, cov_mode=None, langevin=False, clip=False):
    batch_size = TASKS_DICT[task_name]["bs"][N_TRAIN.index(n_train)]
    lr = TASKS_DICT[task_name]["lr"][N_TRAIN.index(n_train)]
    if task_name == "lotka_volterra_good":
        path = PATH_EXPERIMENT + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{N_EPOCHS}_lr_{lr}_new_log/"
    else:
        path = PATH_EXPERIMENT + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{N_EPOCHS}_lr_{lr}_new/"
    path = path + "langevin_steps_400_5/" if langevin else path + "euler_steps_1000/"
    path  = path + result_name + f"_{num_obs}_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        path = path[:-4] + "_clip.pkl"
    # path = path[:-4] + "_prior.pkl"
    return path

def load_runtimes(task_name, n_train, lr,  n_obs, cov_mode=None, langevin=False, clip=False):
    runtimes = {}
    for num_obs in NUM_OBSERVATION_LIST:
        path = path_to_results(task_name, "time", num_obs, n_train, lr, n_obs, cov_mode, langevin, clip)
        runtimes[num_obs] = torch.load(path)
    return runtimes

def load_samples(task_name, n_train, lr, n_obs, cov_mode=None, langevin=False, clip=False):
    samples = {}
    for num_obs in NUM_OBSERVATION_LIST:
        filename = path_to_results(task_name, "posterior_samples", num_obs, n_train, lr, n_obs, cov_mode, langevin, clip)
        samples[num_obs] = torch.load(filename)
    return samples

def load_reference_samples(task_name, n_obs):
    path = PATH_EXPERIMENT + f"{task_name}/reference_posterior_samples/" #_prior/"
    samples_ref = {}
    for num_obs in NUM_OBSERVATION_LIST:
        filename = path + f"true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
        samples_ref[num_obs] = torch.load(filename)
        if task_name == "slcp_good":
            samples_ref[num_obs] = samples_ref[num_obs].reshape(-1, 5)
        samples_ref[num_obs] = samples_ref[num_obs][:1000]
    return samples_ref

def ignore_num_obs(task_name, num_obs):
    ignore = False
    # if task_name == "sir_good" and num_obs == 4:
    #     ignore = True
    # if task_name == "lotka_volterra_good" and num_obs == 6:
    #     ignore = True
    return ignore


# compute mean distance to true theta over all observations
def compute_mean_distance(
    metric, task_name, n_train, n_obs, cov_mode=None, langevin=False, clip=False, percentage=0,
):
    samples = load_samples(
        task_name,
        n_train=n_train,
        lr=LR,
        n_obs=n_obs,
        cov_mode=cov_mode,
        langevin=langevin,
        clip=clip,
    )

    # # load results if already computed
    # filename = PATH_EXPERIMENT + f"{task_name}/metrics/cov_mode{cov_mode}_langevin{langevin}_clip{clip}/n_train{n_train}_n_obs{n_obs}_metric{metric}.pkl"
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # if os.path.exists(filename):
    #     dist_dict = torch.load(filename)
    #     return dist_dict
    # else:

    dist_list = []

    if metric in ["mmd", "swd", "c2st"]:
        samples_ref = load_reference_samples(task_name, n_obs)
        for num_obs in NUM_OBSERVATION_LIST:
            if not ignore_num_obs(task_name, num_obs):
                # mmd
                if metric == "mmd":
                    dist = mmd(
                        torch.tensor(np.array(samples_ref[num_obs])), samples[num_obs]
                    )
                # sliced wasserstein distance
                if metric == "swd":
                    dist = sliced_wasserstein_distance(
                        np.array(samples_ref[num_obs]), np.array(samples[num_obs]), n_projections=100
                    )
                if metric == "c2st":
                    dist = c2st(torch.tensor(np.array(samples_ref[num_obs])), samples[num_obs])

                dist_list.append(dist)
    if metric == "mmd_to_dirac":
        for num_obs in NUM_OBSERVATION_LIST:
            if not ignore_num_obs(task_name, num_obs):
                # mmd to dirac
                theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list.pkl")[num_obs-1]
                # theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl")[num_obs-1]
                dist = dist_to_dirac(
                    samples[num_obs], theta_true, percentage=percentage, scaled=True,
                )["mmd"]

                dist_list.append(dist)
        
    dist_dict = {
        "mean": torch.tensor(dist_list).mean(),
        "std": torch.tensor(dist_list).std(),
        }
    # torch.save(dist_dict, filename)

    return dist_dict

def ignore_method_n_obs(metric, method, task_name, n_obs):
    ignore = False
    return ignore

def ignore_method_n_train(metric, method, task_name, n_train):
    ignore = False
    return ignore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--w_dist", action="store_true")
    parser.add_argument("--mmd_dist", action="store_true")
    parser.add_argument("--c2st_dist", action="store_true")
    parser.add_argument("--dirac_dist", action="store_true")
    parser.add_argument("--plot_samples", action="store_true")

    args = parser.parse_args()

    alpha, alpha_fill = set_plotting_style()

    if args.losses:
        # plot losses to select lr
        lr_list = [1e-4, 1e-3]
        n_rows = len(TASKS)
        n_cols = len(N_TRAIN)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharex=True, sharey=True) #, constrained_layout=True)
        fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)

        for i, task_name in enumerate(TASKS.keys()):
            for j, n_train in enumerate(N_TRAIN):
                best_val_loss = {}
                for lr_, c in zip(lr_list, ["blue", "orange"]):
                    train_losses, val_losses, best_epoch = load_losses(
                        task_name, n_train, lr_, path=PATH_EXPERIMENT
                    )
                    best_val_loss[lr_] = val_losses[best_epoch]
                    axs[i, j].plot(train_losses, label=f"train, lr={lr_}", color=c, linewidth=3, alpha=0.3)
                    axs[i, j].plot(val_losses, label=f"val, lr={lr_}", color=c, linewidth=3, alpha=0.9)
                    axs[i, j].axvline(best_epoch, color=c, linestyle="--", linewidth=5, alpha=0.9, zorder=10000)
                # print(best_val_loss)
                # get lr for min best val loss
                best_lr = min(best_val_loss, key=best_val_loss.get)
                print(f"best lr for {task_name}, {n_train}: {best_lr}")
                # set ax title as N_train only at the top
                if i == 0:
                    axs[i, j].set_title(r"$N_\mathrm{train}$"+rf"$={n_train}$")
                # label xaxis only at the bottom
                if i == len(TASKS) - 1:
                    axs[i, j].set_xlabel("epochs")
                # label yaxis only at the left
                if j == 0:
                    axs[i, j].set_ylabel(TASKS[task_name])
                axs[i, j].set_ylim([0, 0.5])
        axs[i, j].legend()
        plt.savefig(PATH_EXPERIMENT + f"_plots_new/losses_bs_{BATCH_SIZE}.png")
        plt.savefig(PATH_EXPERIMENT + f"_plots_new/sbibm_losses_bs_{BATCH_SIZE}.pdf")
        plt.clf()

    if args.w_dist or args.mmd_dist or args.c2st_dist:
        metric = "swd" if args.w_dist else "mmd" if args.mmd_dist else "c2st"
        task_names = TASKS.keys()
        # plot mean distance to true theta as function of n_obs
        n_rows = len(task_names)
        n_cols = len(N_TRAIN)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharex=True) #, constrained_layout=True)
        fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
        for i, task_name in enumerate(task_names):
            for j, n_train in tqdm(
                enumerate(N_TRAIN), desc=f"{task_name}, {metric}"
            ):
                mean_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                std_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                for n_obs in N_OBS:
                    for method in METHODS_STYLE.keys():
                        if ignore_method_n_train(metric, method, task_name, n_train):
                            print(f"ignoring {method} for {task_name}, {n_train}")
                            dist_mean = torch.nan
                            dist_std = torch.nan
                        else:
                            dist = compute_mean_distance(
                                task_name=task_name,
                                n_train=n_train,
                                n_obs=n_obs,
                                cov_mode=method.split("_")[0],
                                langevin=True if "LANGEVIN" in method else False,
                                clip=True if "clip" in method else False,
                                metric=metric,
                            )
                            dist_mean = dist["mean"]
                            dist_std = dist["std"]

                        mean_dist_dict[method].append(dist_mean)
                        std_dist_dict[method].append(dist_std)

                for k, mean_, std_ in zip(mean_dist_dict.keys(), mean_dist_dict.values(), std_dist_dict.values()):
                    mean_, std_ = torch.FloatTensor(mean_), torch.FloatTensor(std_)
                    axs[i, j].fill_between(
                        N_OBS,
                        mean_ - std_,
                        mean_ + std_,
                        alpha=alpha_fill,
                        color=METHODS_STYLE[k]["color"],
                    )
                    axs[i, j].plot(
                        N_OBS,
                        mean_,
                        alpha=alpha,
                        **METHODS_STYLE[k],
                    )

                # set ax title as N_train only at the top
                if i == 0:
                    axs[i, j].set_title(r"$N_\mathrm{train}$"+rf"$={n_train}$")
                # label xaxis only at the bottom
                if i == len(task_names) - 1:
                    axs[i, j].set_xlabel(r"$n$")
                    axs[i, j].set_xticks(N_OBS)
                # label yaxis only at the left
                if j == 0:
                    axs[i, j].set_ylabel(TASKS[task_name] + "\n" + METRICS_STYLE[metric]["label"])
                else:
                    axs[i, j].set_yticklabels([])
                if metric == "swd":
                    if "lotka" in task_name:
                        axs[i, j].set_ylim([0, 0.05])
                    if "sir" in task_name:
                        axs[i, j].set_ylim([0, 0.05])
                elif metric == "mmd":
                    if "lotka" in task_name:
                        axs[i, j].set_ylim([0, 1])
                    if "sir" in task_name:
                        axs[i, j].set_ylim([0, 1])
                else:
                    axs[i, j].set_ylim([0, 1])
        handles, labels = axs[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc="lower right")

        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_obs_final.png")
        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_obs_final.pdf")
        plt.clf()

        # same but as function of n_train
        n_rows = len(task_names)
        n_cols = len(N_OBS)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharex=True)
        fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
        for i, task_name in enumerate(task_names):
            for j, n_obs in tqdm(
                enumerate(N_OBS), desc=f"{task_name}, {metric}"
            ):
                mean_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                std_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                for n_train in N_TRAIN:
                    for method in METHODS_STYLE.keys():
                        if ignore_method_n_obs(metric, method, task_name, n_obs):
                            print(f"ignoring {method} for {task_name}, {n_obs}")
                            dist_mean = torch.nan
                            dist_std = torch.nan
                        else:
                            dist = compute_mean_distance(
                                task_name=task_name,
                                n_train=n_train,
                                n_obs=n_obs,
                                cov_mode=method.split("_")[0],
                                langevin=True if "LANGEVIN" in method else False,
                                clip=True if "clip" in method else False,
                                metric=metric,
                            )
                            dist_mean = dist["mean"]
                            dist_std = dist["std"]

                        mean_dist_dict[method].append(dist_mean)
                        std_dist_dict[method].append(dist_std)

                for k, mean_, std_ in zip(mean_dist_dict.keys(), mean_dist_dict.values(), std_dist_dict.values()):
                    mean_, std_ = torch.FloatTensor(mean_), torch.FloatTensor(std_)
                    axs[i, j].fill_between(
                        N_TRAIN,
                        mean_ - std_,
                        mean_ + std_,
                        alpha=alpha_fill,
                        color=METHODS_STYLE[k]["color"],
                    )
                    axs[i, j].plot(
                        N_TRAIN,
                        mean_,
                        alpha=alpha,
                        **METHODS_STYLE[k],
                    )

                # set ax title as N_obs only at the top
                if i == 0:
                    axs[i, j].set_title(r"$n$"+rf"$={n_obs}$")
                # label xaxis only at the bottom
                if i == len(task_names) - 1:
                    axs[i, j].set_xlabel(r"$N_\mathrm{train}$")
                    axs[i, j].set_xscale("log")
                    axs[i, j].set_xticks(N_TRAIN)
                # label yaxis only at the left
                if j == 0:
                    axs[i, j].set_ylabel(TASKS[task_name] + "\n" + METRICS_STYLE[metric]["label"])
                else:
                    axs[i, j].set_yticklabels([])
                if metric == "mmd":
                    if "lotka" in task_name:
                        axs[i, j].set_ylim([0, 1])
                    if "sir" in task_name:
                        axs[i, j].set_ylim([0, 1])
                elif metric == "swd":
                    if "lotka" in task_name:
                        axs[i, j].set_ylim([0, 0.05])
                    if "sir" in task_name:
                        axs[i, j].set_ylim([0, 0.05])
                else:
                    axs[i, j].set_ylim([0, 1])

        handles, labels = axs[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc="lower right")

        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_train_final.png")
        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_train_final.pdf")
        plt.clf()

    if args.dirac_dist:

        # splot as function of n_train
        metric = "mmd_to_dirac"
        fig, axs = plt.subplots(len(TASKS), 5, figsize=(25, len(TASKS)*5), sharex=True) #, constrained_layout=True)
        fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)

        for i, task_name in enumerate(TASKS.keys()):
            for j, n_obs in tqdm(
                enumerate(N_OBS), desc=f"{task_name}, {metric}"
            ):
                mean_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                std_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                for n_train in N_TRAIN:
                    for method in METHODS_STYLE.keys():
                        if ignore_method_n_obs(metric, method, task_name, n_obs):
                            print(f"ignoring {method} for {task_name}, {n_obs}")
                            dist_mean = torch.nan
                            dist_std = torch.nan
                        else:
                            dist = compute_mean_distance(
                                task_name=task_name,
                                n_train=n_train,
                                n_obs=n_obs,
                                cov_mode=method.split("_")[0],
                                langevin=True if "LANGEVIN" in method else False,
                                clip=True if "clip" in method else False,
                                metric=metric,
                            )
                            dist_mean = dist["mean"]
                            dist_std = dist["std"]

                        mean_dist_dict[method].append(dist_mean)
                        std_dist_dict[method].append(dist_std)

                for k, mean_, std_ in zip(mean_dist_dict.keys(), mean_dist_dict.values(), std_dist_dict.values()):
                    mean_, std_ = torch.FloatTensor(mean_), torch.FloatTensor(std_)
                    axs[i, j].fill_between(
                        N_TRAIN,
                        mean_ - std_,
                        mean_ + std_,
                        alpha=alpha_fill,
                        color=METHODS_STYLE[k]["color"],
                    )
                    axs[i, j].plot(
                        N_TRAIN,
                        mean_,
                        alpha=alpha,
                        **METHODS_STYLE[k],
                    )

                # set ax title as N_obs only at the top
                if i == 0:
                    axs[i, j].set_title(r"$n$"+rf"$={n_obs}$")
                # label xaxis only at the bottom
                if i == len(TASKS) - 1:
                    axs[i, j].set_xlabel(r"$N_\mathrm{train}$")
                    axs[i, j].set_xscale("log")
                    axs[i, j].set_xticks(N_TRAIN)
                # label yaxis only at the left
                if j == 0:
                    axs[i, j].set_ylabel(TASKS[task_name] + "\n" + METRICS_STYLE[metric]["label"])
                else:
                    axs[i, j].set_yticklabels([])
                if "lotka" in task_name:
                    axs[i, j].set_ylim([0, 8])
                elif "sir" in task_name:
                    axs[i, j].set_ylim([0, 0.3])
                else:
                    axs[i, j].set_ylim([0, 1])

        handles, labels = axs[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc="lower right")

        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_train_final.png")
        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_train_final.pdf")
        plt.clf()

        # plot as function of n_obs
        metric = "mmd_to_dirac"
        fig, axs = plt.subplots(len(TASKS), 4, figsize=(20, len(TASKS)*5), sharex=True) #, constrained_layout=True)
        fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
        
        for i, task_name in enumerate(TASKS.keys()):
            for j, n_train in enumerate(N_TRAIN):
                mean_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                std_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                for n_obs in N_OBS:
                    for method in METHODS_STYLE.keys():
                        if ignore_method_n_train(metric, method, task_name, n_train):
                            print(f"ignoring {method} for {task_name}, {n_train}")
                            dist_mean = torch.nan
                            dist_std = torch.nan
                        else:
                            dist = compute_mean_distance(
                                task_name=task_name,
                                n_train=n_train,
                                n_obs=n_obs,
                                cov_mode=method.split("_")[0],
                                langevin=True if "LANGEVIN" in method else False,
                                clip=True if "clip" in method else False,
                                metric=metric,
                            )
                            dist_mean = dist["mean"]
                            dist_std = dist["std"]

                        mean_dist_dict[method].append(dist_mean)
                        std_dist_dict[method].append(dist_std)

                for k, mean_, std_ in zip(mean_dist_dict.keys(), mean_dist_dict.values(), std_dist_dict.values()):
                    mean_, std_ = torch.FloatTensor(mean_), torch.FloatTensor(std_)
                    axs[i, j].fill_between(
                        N_OBS,
                        mean_ - std_,
                        mean_ + std_,
                        alpha=alpha_fill,
                        color=METHODS_STYLE[k]["color"],
                    )
                    axs[i, j].plot(
                        N_OBS,
                        mean_,
                        alpha=alpha,
                        **METHODS_STYLE[k],
                    )

                # set ax title as N_train only at the top
                if i == 0:
                    axs[i, j].set_title(r"$N_\mathrm{train}$"+rf"$={n_train}$")
                # label xaxis only at the bottom
                if i == len(TASKS) - 1:
                    axs[i, j].set_xlabel(r"$n$")
                    axs[i, j].set_xticks(N_OBS)
                # label yaxis only at the left
                if j == 0:
                    axs[i, j].set_ylabel(TASKS[task_name] + "\n" + METRICS_STYLE[metric]["label"])
                else:
                    axs[i, j].set_yticklabels([])
                if "lotka" in task_name:
                    axs[i, j].set_ylim([0, 8])
                elif "sir" in task_name:
                    axs[i, j].set_ylim([0, 0.3])
                else:
                    axs[i, j].set_ylim([0, 1])

        handles, labels = axs[0, 0].get_legend_handles_labels()
        plt.legend(handles, labels, loc="lower right")

        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_obs_final.png")
        plt.savefig(PATH_EXPERIMENT + f"_plots_new/{metric}_n_obs_final.pdf")
        plt.clf()

    if args.plot_samples:

        n_train = 30000
        task_name = "slcp_good"
        save_path = PATH_EXPERIMENT + f"_samples/{task_name}/"
        os.makedirs(save_path, exist_ok=True)

        # num_obs_lists = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
        num_obs_lists = [[1,2,3,4,5,6,7,8,9,10]]
        for num_obs_list in num_obs_lists:
            fig, axs = plt.subplots(len(num_obs_list), 5, figsize=(25, 5 * len(num_obs_list)))
            fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
            for j, num_obs in enumerate(num_obs_list):
                # true_theta = get_task(task_name).get_true_parameters(num_obs)[0]
                theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list.pkl")[num_obs-1]
                # theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl")[num_obs-1]
                if theta_true.ndim > 1:
                    theta_true = theta_true[0]
                # for n_obs in N_OBS:
                #     samples_ref = load_reference_samples(task_name, n_obs)
                #     plt.scatter(samples_ref[num_obs][:, 0], samples_ref[num_obs][:, 1], alpha=0.5)
                # plt.scatter(theta_true[0], theta_true[1], color="black")
                # plt.savefig(save_path + f"num_obs_{num_obs}_ana_prior.png")
                # plt.clf()
                # fig, axs = plt.subplots(1, 5, figsize=(25, 5))
                for i, n_obs in enumerate(N_OBS):
                    samples_ref = load_reference_samples(task_name, n_obs)
                    samples_gauss = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, cov_mode="GAUSS", clip=False)
                    # samples_jac = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, cov_mode="JAC", clip=False)
                    # samples_langevin = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, langevin=True, clip=False)
                    axs[j, i].scatter(samples_ref[num_obs][:, 1], samples_ref[num_obs][:, 2], alpha=0.5, label=f"ANALYTIC", color="lightgreen")
                    axs[j, i].scatter(samples_gauss[num_obs][:, 1], samples_gauss[num_obs][:, 2], alpha=0.5, label=f"GAUSS", color="blue")
                    # axs[j, i].scatter(samples_jac[num_obs][:, 0], samples_jac[num_obs][:, 1], alpha=0.5, label=f"JAC", color="orange")
                    # axs[j, i].scatter(samples_langevin[num_obs][:, 0], samples_langevin[num_obs][:, 1], alpha=0.1, label=f"LANGEVIN", color="red")
                    axs[j, i].scatter(theta_true[1], theta_true[2], color="black", s=100, label="True parameter")
                    axs[j, i].set_xticks([])
                    axs[j, i].set_yticks([])
                    if j == 0:
                        axs[j, i].set_title(fr"$n = {n_obs}$")
                    if i == 0:
                        axs[j, i].set_ylabel(f"num_obs = {num_obs}")
            plt.legend()
            plt.savefig(save_path + f"all_n_train_{n_train}_bs_{BATCH_SIZE}_lr_{LR}.png") #_prior_num_{num_obs_list[0]}_{num_obs_list[-1]}.png")
            plt.savefig(save_path + f"all_n_train_{n_train}_bs_{BATCH_SIZE}_lr_{LR}.png") #_prior_num_{num_obs_list[0]}_{num_obs_list[-1]}.pdf")
            plt.clf()
            