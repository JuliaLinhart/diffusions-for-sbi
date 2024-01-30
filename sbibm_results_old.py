import torch
import numpy as np
import matplotlib.pyplot as plt

from tasks.sbibm.data_generators import get_task

from experiment_utils import dist_to_dirac
from ot import sliced_wasserstein_distance
from sbibm. metrics import mmd

from matplotlib import colormaps as cm
from plot_utils import set_plotting_style, METHODS_STYLE, METRICS_STYLE, pairplot_with_groundtruth_md

from tqdm import tqdm

PATH_EXPERIMENT = "results/sbibm/"
TASKS = {
    "gaussian_linear": "Gaussian Linear",
    "gaussian_mixture": "Gaussian Mixture",
    "slcp": "SLCP",
    "lotka_volterra": "Lotka-Volterra",
    "sir": "SIR",
}
N_TRAIN = [1000, 3000, 10000, 30000]
BATCH_SIZE = 64
N_EPOCHS = 10000
LR = 1e-4
N_OBS = [1, 8, 14, 22, 30]
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7] #, 8, 9, 10] # waiting for slcp ref samples
METRICS = ["swd", "mmd"]
# METRICS = ["mmd_to_dirac"]

def load_losses(task_name, n_train, lr, path,):
    batch_size = 256 if n_train == 30000 else BATCH_SIZE
    n_epochs = 5000 if n_train == 30000 else N_EPOCHS
    if task_name == "sir":
        n_epochs = 5000
        if lr == 1e-3 and n_train in [3000, 10000]:
            n_epochs = 10000
    losses = torch.load(path + f'{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}/train_losses.pkl')
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    best_epoch = losses["best_epoch"]
    return train_losses, val_losses, best_epoch

def path_to_results(task_name, result_name, num_obs, n_train, lr, n_obs, cov_mode=None, langevin=False, clip=False):
    batch_size = 256 if n_train == 30000 else 64
    n_epochs = 5000 if n_train == 30000 else 10000
    if task_name == "sir":
        n_epochs = 5000
        if lr == 1e-3 and n_train in [3000, 10000]:
            n_epochs = 10000

    path = PATH_EXPERIMENT + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}/"
    path = path + "langevin_steps_400_5/" if langevin else path + "euler_steps_1000/"
    path  = path + result_name + f"_{num_obs}_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        path = path[:-4] + "_clip.pkl"
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
    path = PATH_EXPERIMENT + f"{task_name}/reference_posterior_samples/"
    samples_ref = {}
    for num_obs in NUM_OBSERVATION_LIST:
        filename = path + f"true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
        samples_ref[num_obs] = torch.load(filename)
        if task_name == "slcp":
            samples_ref[num_obs] = samples_ref[num_obs].reshape(-1, 5)[:1000]
    return samples_ref


# compute mean distance to true theta over all observations
def compute_mean_distance_to_true_theta(
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
    samples_ref = load_reference_samples(task_name, n_obs)
    # compute dist between samples and true theta
    dist_dict = {"mmd_to_dirac": [], "swd": [], "mmd": []}
    for num_obs in NUM_OBSERVATION_LIST:
        dist = {}

        # mmd to dirac
        if metric == "mmd_to_dirac":
            theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list.pkl")[num_obs-1]
            dist["mmd_to_dirac"] = dist_to_dirac(
                samples[num_obs], theta_true, percentage=percentage, scaled=True,
            )["mmd"]
        else:     
            # mmd
            if metric == "mmd":
                dist["mmd"] = mmd(
                    torch.tensor(np.array(samples_ref[num_obs])), samples[num_obs]
                )
            # sliced wasserstein distance
            elif metric == "swd":
                dist["swd"] = sliced_wasserstein_distance(
                    np.array(samples_ref[num_obs]), np.array(samples[num_obs]), n_projections=100
                )
            else:
                raise ValueError(f"Unknown metric {metric}")

        dist_dict[metric].append(dist[metric])
        
    for metric in METRICS:
        dist_dict[metric] = {
            "mean": torch.tensor(dist_dict[metric]).mean(),
            "std": torch.tensor(dist_dict[metric]).std(),
        }

    return dist_dict

def ignore_method_n_obs(metric, method, task_name, n_obs):
    ignore = False
    if metric in ["swd", "mmd"] and task_name in ["sir", "lotka_volterra"]:
        ignore = True
    if metric == "swd" and method in ["JAC", "LANGEVIN"] and n_obs > 1 and task_name in ["sir", "lotka_volterra"]:
        ignore = True
    if metric == "mmd" and method == "JAC" and n_obs == 14 and task_name == "slcp":
        ignore = True
    if metric == "mmd_to_dirac" and method == "JAC":
        if task_name in ["lotka_volterra", "sir"]:
            ignore = True
        if task_name in ["slcp"] and n_obs in [22,30]:
            ignore = True
        if task_name in ["gaussian_mixture"] and n_obs > 1:
            ignore = True
    if metric == "mmd_to_dirac" and method == "LANGEVIN":
        if task_name in ["sir", "slcp"] and n_obs in [22, 30]:
            ignore = True
    return ignore

def ignore_method_n_train(metric, method, task_name, n_train):
    ignore = False
    if metric in ["swd", "mmd"] and task_name in ["sir", "lotka_volterra"]:
        ignore = True
    if metric == "swd" and method in ["JAC", "LANGEVIN"] and task_name in ["sir", "lotka_volterra"]:
        ignore = True
    if metric == "mmd" and method == "JAC" and n_train == 3000 and task_name == "slcp":
        ignore = True
    if metric == "mmd_to_dirac" and method == "JAC":
        if task_name in ["lotka_volterra", "sir"]:
            ignore = True
        if task_name == "gaussian_mixture" and n_train in [10000, 30000]:
            ignore = True
        if task_name == "slcp" and n_train in [3000, 10000, 30000]:
            ignore = True
    if metric == "mmd_to_dirac" and method == "LANGEVIN":
        if task_name == "sir" and n_train in [30000]:
            ignore = True
    return ignore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--w_dist", action="store_true")
    parser.add_argument("--runtimes", action="store_true")
    parser.add_argument("--corner_plots", action="store_true")

    args = parser.parse_args()

    alpha, alpha_fill = set_plotting_style()

    if args.losses:
        # plot losses to select lr
        lr_list = [1e-4, 1e-3]
        fig, axs = plt.subplots(5, 4, figsize=(20, 25), constrained_layout=True, sharey=True)
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
                    axs[i, j].axvline(best_epoch, color=c, linestyle="--", linewidth=3, alpha=0.9, zorder=10000)
                # print(best_val_loss)
                # get lr for min best val loss
                # best_lr = min(best_val_loss, key=best_val_loss.get)
                best_lr = 1e-4
                axs[i, j].set_title(TASKS[task_name] + f" \n n_train = {n_train}" + f" \n best_lr={best_lr}")
                axs[i, j].set_ylim([0, 0.5])
                axs[i, j].set_xlim([0, 5000])
                if i == 3:
                    axs[i, j].set_xlabel("epochs")
                if j == 0:
                    axs[i, j].set_ylabel("loss")
        axs[i, j].legend()
        plt.savefig(PATH_EXPERIMENT + "losses.png")
        plt.savefig(PATH_EXPERIMENT + "sbibm_losses.pdf")
        plt.clf()

    if args.w_dist:
        # plot mean distance to true theta as function of n_obs
        for metric in METRICS:
            fig, axs = plt.subplots(5, 4, figsize=(20, 25), constrained_layout=True)
            for i, task_name in enumerate(TASKS.keys()):
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
                                dist = compute_mean_distance_to_true_theta(
                                    task_name=task_name,
                                    n_train=n_train,
                                    n_obs=n_obs,
                                    cov_mode=method.split("_")[0],
                                    langevin=True if "LANGEVIN" in method else False,
                                    clip=True if "clip" in method else False,
                                    metric=metric,
                                )
                                dist_mean = dist[metric]["mean"]
                                dist_std = dist[metric]["std"]

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

                    axs[i, j].set_title(TASKS[task_name] + f"\n n_train={n_train}")
                    axs[i, j].set_xlabel("n_obs")
                    axs[i, j].set_xticks(N_OBS)
                    axs[i, j].set_ylabel(METRICS_STYLE[metric]["label"])
            handles, labels = axs[0, 0].get_legend_handles_labels()
            plt.legend(handles, labels, loc="lower right")

            plt.savefig(PATH_EXPERIMENT + f"{metric}_n_obs_old.png")
            plt.savefig(PATH_EXPERIMENT + f"{metric}_n_obs_old.pdf")
            plt.clf()

        # same but as function of n_train
        for metric in METRICS:
            fig, axs = plt.subplots(5, 5, figsize=(25, 25), constrained_layout=True)
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
                                dist = compute_mean_distance_to_true_theta(
                                    task_name=task_name,
                                    n_train=n_train,
                                    n_obs=n_obs,
                                    cov_mode=method.split("_")[0],
                                    langevin=True if "LANGEVIN" in method else False,
                                    clip=True if "clip" in method else False,
                                    metric=metric,
                                )
                                dist_mean = dist[metric]["mean"]
                                dist_std = dist[metric]["std"]

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

                    axs[i, j].set_title(TASKS[task_name] + f"\n n_obs={n_obs}")
                    axs[i, j].set_xlabel("n_train")
                    if not (metric in ["mmd", "swd"] and task_name in ["sir", "lotka_volterra"]):
                        axs[i, j].set_xscale("log")
                    axs[i, j].set_xticks(N_TRAIN)
                    axs[i, j].set_ylabel(METRICS_STYLE[metric]["label"])
            handles, labels = axs[0, 0].get_legend_handles_labels()
            plt.legend(handles, labels, loc="lower right")

            plt.savefig(PATH_EXPERIMENT + f"{metric}_n_train_old.png")
            plt.savefig(PATH_EXPERIMENT + f"{metric}_n_train_old.pdf")
            plt.clf()

    if args.runtimes:
        n_train = 30000
        # plot mean run over all observations as a function of n_obs for each task
        fig, axs = plt.subplots(1, 4, figsize=(20, 20), constrained_layout=True)
        for i, task_name in enumerate(TASKS.keys()):
            runtimes_dict = {method: [] for method in METHODS_STYLE.keys()}
            for n_obs in N_OBS:
                for method in METHODS_STYLE.keys():
                    runtimes = load_runtimes(
                        task_name,
                        n_train=n_train,
                        lr=LR,
                        n_obs=n_obs,
                        cov_mode=method.split("_")[0],
                        langevin=True if "LANGEVIN" in method else False,
                        clip=True if "clip" in method else False,
                    )
                    runtimes_dict[method].append(runtimes[n_obs])

            for k, v in runtimes_dict.items():
                axs[i].plot(N_OBS, v, linewidth=3, alpha=alpha, **METHODS_STYLE[k])
            axs[i].set_title(TASKS[task_name] + f" \n n_train={n_train}")
            axs[i].set_xlabel("n_obs")
            axs[i].set_xticks(N_OBS)
            axs[i].set_ylabel("runtimes")
        handles, labels = axs[0].get_legend_handles_labels()
        plt.legend(handles, labels, loc="lower right")
        plt.savefig(PATH_EXPERIMENT + "runtimes.png")
        plt.savefig(PATH_EXPERIMENT + "runtimes.pdf")
        plt.clf()

    if args.corner_plots:
        # corner plots for every n_obs
        num_obs = 1
        n_train = 30000
        cov_mode = "GAUSS"
        clip = True
        langevin = False

        label = cov_mode if not langevin else "LANGEVIN"
        label = label if not clip else label + " (clip)"
        
        for task_name in TASKS.keys():
            theta_true = get_task(task_name).get_true_parameters(num_obs)
            if task_name in ["gaussian_linear", "gaussian_mixture"]:
                theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/heta_true_list.pkl")[num_obs-1]
            samples = []
            for n_obs in N_OBS:
                samples.append(
                    load_samples(
                        task_name, n_train, lr=LR, n_obs=n_obs, cov_mode=cov_mode, langevin=langevin, clip=clip
                    )[num_obs]
                )
                samples_ref = np.array(load_reference_samples(task_name, n_obs=1)[num_obs])

            pairplot_with_groundtruth_md(
                samples_list=[samples_ref] + samples,
                labels=[r"$p(\theta | x^\star)$"]+[label + f", n = {n_obs}" for n_obs in N_OBS],
                colors=["gray"] + [cm.get_cmap("rainbow")(i) for i in torch.linspace(1, 0, len(samples))],
                theta_true=theta_true,
                plot_bounds=None, # should be the prior
            )
            plt.savefig(
                PATH_EXPERIMENT
                + f"corner_plots_{task_name}_n_train_{n_train}_num_obs_{num_obs}.png"
            )
            plt.savefig(
                PATH_EXPERIMENT
                + f"corner_plots_{task_name}_n_train_{n_train}_num_obs_{num_obs}.pdf"
            )
            plt.clf()


    # n_train = 1000
    # n_obs = 30
    # task_names = ['gaussian_linear', 'gaussian_mixture']
    # for task_name in task_names:
    #     samples = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, cov_mode="GAUSS")
    #     for num_obs in NUM_OBSERVATION_LIST:
    #         true_theta = get_task(task_name).get_true_parameters(num_obs)
    #         dist = dist_to_dirac(samples[num_obs], true_theta)
    #         print(task_name, num_obs, dist["mse"], dist["mmd"])

    # n_train = 10000
    # n_obs = 8
    # task_name = "gaussian_mixture"

    # for num_obs in NUM_OBSERVATION_LIST:
    #     true_theta = get_task(task_name).get_true_parameters(num_obs)[0]
    #     for n_obs in N_OBS[:1]:
    #         samples_ref = load_reference_samples(task_name, n_obs)
    #         plt.scatter(samples_ref[num_obs][:, 0], samples_ref[num_obs][:, 1], alpha=0.1)
    #         samples = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, cov_mode="GAUSS", clip=True)
    #         plt.scatter(samples[num_obs][:, 0], samples[num_obs][:, 1], alpha=0.5, label=f"n_obs={n_obs}")
    #     plt.scatter(true_theta[0], true_theta[1], color="black")
    #     plt.savefig(f"samples_{task_name}_{num_obs}_.png")
    #     plt.clf()
        # for n_obs in N_OBS:
        #     samples = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, cov_mode="GAUSS", clip=True)
        #     plt.scatter(samples[num_obs][:, 0], samples[num_obs][:, 1], alpha=0.5, label=f"n_obs={n_obs}")
        # plt.scatter(true_theta[0], true_theta[1], color="black")
        # plt.legend()
        # plt.savefig(f"samples_{task_name}_{num_obs}_clip.png")
        # plt.clf()
        # for n_obs in N_OBS:
        #     samples = load_samples(task_name, n_train, lr=LR, n_obs=n_obs, langevin=True, clip=True)
        #     plt.scatter(samples[num_obs][:, 0], samples[num_obs][:, 1], alpha=0.5, label=f"n_obs={n_obs}")
        # plt.scatter(true_theta[0], true_theta[1], color="black")
        # plt.savefig(f"samples_{task_name}_{num_obs}_langevin_clip.png")
        # plt.clf()