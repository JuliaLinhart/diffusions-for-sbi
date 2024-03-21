import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from experiment_utils import dist_to_dirac
from ot import sliced_wasserstein_distance
from sbibm.metrics import mmd, c2st

from matplotlib import colormaps as cm
from plot_utils import (
    set_plotting_style,
    METHODS_STYLE,
    METRICS_STYLE,
    pairplot_with_groundtruth_md,
)

from tqdm import tqdm

# seed
torch.manual_seed(42)

PATH_EXPERIMENT = "results/sbibm/"
TASKS = {
    # "gaussian_linear": "Gaussian Linear",
    # "gaussian_mixture": "Gaussian Mixture",
    "slcp_good": "SLCP",
    "lotka_volterra_good": "Lotka-Volterra",
    "sir_good": "SIR",
}
N_TRAIN = [1000, 3000, 10000, 30000]  # , 50000]
BATCH_SIZE = 256  # 64
N_EPOCHS = 5000
LR = 1e-4

TASKS_DICT = {
    "slcp_good": {
        "lr": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
        "bs": [256, 256, 256, 256, 256],
    },
    "lotka_volterra_good": {
        "lr": [1e-4, 1e-4, 1e-4, 1e-4, 1e-3],
        "bs": [256, 256, 256, 256, 256],
    },
    "sir_good": {"lr": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4], "bs": [64, 64, 64, 64, 64]},
}

N_OBS = [1, 8, 14, 22, 30]
NUM_OBSERVATION_LIST = list(np.arange(1, 26))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

METRICS = ["mmd", "swd", "mmd_to_dirac"]


def load_losses(task_name, n_train, lr, path, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    if task_name == "lotka_volterra_good":
        path = (
            path
            + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}_new_log/"
        )
    else:
        path = (
            path
            + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}_new/"
        )
    losses = torch.load(path + f"train_losses.pkl")
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    best_epoch = losses["best_epoch"]
    return train_losses, val_losses, best_epoch


def path_to_results(
    task_name,
    result_name,
    num_obs,
    n_train,
    n_obs,
    cov_mode=None,
    sampler="euler",
    langevin=False,
    clip=False,
    ours=False,
):
    batch_size = TASKS_DICT[task_name]["bs"][N_TRAIN.index(n_train)]
    lr = TASKS_DICT[task_name]["lr"][N_TRAIN.index(n_train)]
    if task_name == "lotka_volterra_good":
        path = (
            PATH_EXPERIMENT
            + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{N_EPOCHS}_lr_{lr}_new_log/"
        )
    else:
        path = (
            PATH_EXPERIMENT
            + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{N_EPOCHS}_lr_{lr}_new/"
        )

    # path = path + "langevin_steps_400_5_new/" if langevin else path + f"{sampler}_steps_1000/" if sampler == "euler" or cov_mode == "GAUSS" else path + f"{sampler}_steps_400/"
    if langevin:
        path = path + "langevin_steps_400_5/"
        if not ours:
            path = path[:-1] + "_new/"
    else:
        path = (
            path + f"{sampler}_steps_1000/"
            if sampler == "euler" or cov_mode == "GAUSS"
            else path + f"{sampler}_steps_400/"
        )

    path = path + result_name + f"_{num_obs}_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        path = path[:-4] + "_clip.pkl"
    path = path[:-4] + "_prior.pkl"
    return path


def load_samples(
    task_name,
    n_train,
    n_obs,
    cov_mode=None,
    sampler="euler",
    langevin=False,
    clip=False,
    ours=False,
):
    samples = {}
    for num_obs in NUM_OBSERVATION_LIST:
        filename = path_to_results(
            task_name,
            "posterior_samples",
            num_obs,
            n_train,
            n_obs,
            cov_mode,
            sampler,
            langevin,
            clip,
            ours,
        )
        samples[num_obs] = torch.load(filename)
    return samples


def load_reference_samples(task_name, n_obs):
    # path = PATH_EXPERIMENT + f"{task_name}/reference_posterior_samples/"
    path = PATH_EXPERIMENT + f"{task_name}/reference_posterior_samples_prior/"
    samples_ref = {}
    for num_obs in NUM_OBSERVATION_LIST:
        filename = path + f"true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
        samples_ref[num_obs] = torch.load(filename)
        # shuffle samples to get samples from each chain
        samples_ref[num_obs] = samples_ref[num_obs][
            torch.randperm(samples_ref[num_obs].shape[0])
        ]
        samples_ref[num_obs] = samples_ref[num_obs][:1000]
    return samples_ref


# compute mean distance to true theta over all observations
def compute_mean_distance(
    metric,
    task_name,
    n_train,
    n_obs,
    cov_mode=None,
    sampler="euler",
    langevin=False,
    clip=False,
    ours=False,
    load=False,
    prec_ignore_nums=None,
):
    # load results if already computed
    save_path = (
        PATH_EXPERIMENT
        + f"{task_name}/metrics_{sampler}/cov_mode_{cov_mode}_langevin_{langevin}_clip_{clip}/"
    )
    if ours:
        save_path = save_path[:-1] + "_old/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    filename = save_path + f"n_train_{n_train}_n_obs_{n_obs}_metric_{metric}.pkl"
    if load and os.path.exists(filename):
        dist_list_all = torch.load(filename)
        dist_list = []

        ignore_nums = []

        for num_obs in NUM_OBSERVATION_LIST:
            dist = dist_list_all[num_obs - 1]
            if num_obs not in prec_ignore_nums:
                dist_list.append(dist)
            if torch.isnan(dist):
                ignore_nums.append(num_obs)
            if metric == "swd":
                if task_name == "slcp_good" and dist > 5:
                    ignore_nums.append(num_obs)
                if task_name == "lotka_volterra_good" and dist > 0.1:
                    ignore_nums.append(num_obs)
                if task_name == "sir_good" and dist > 0.01:
                    ignore_nums.append(num_obs)

    else:
        samples = load_samples(
            task_name,
            n_train=n_train,
            n_obs=n_obs,
            cov_mode=cov_mode,
            langevin=langevin,
            clip=clip,
            sampler=sampler,
        )

        dist_list = []
        ignore_nums = []
        if metric in ["mmd", "swd", "c2st"]:
            samples_ref = load_reference_samples(task_name, n_obs)
            for num_obs in NUM_OBSERVATION_LIST:
                # mmd
                if metric == "mmd":
                    dist = mmd(
                        torch.tensor(np.array(samples_ref[num_obs])), samples[num_obs]
                    )
                # sliced wasserstein distance
                if metric == "swd":
                    dist = sliced_wasserstein_distance(
                        np.array(samples_ref[num_obs]),
                        np.array(samples[num_obs]),
                        n_projections=1000,
                    )
                    dist = torch.tensor(dist)

                if metric == "c2st":
                    dist = c2st(
                        torch.tensor(np.array(samples_ref[num_obs])), samples[num_obs]
                    )

                dist_list.append(dist)

                if torch.isnan(dist):
                    ignore_nums.append(num_obs)
                if metric == "swd":
                    if task_name == "slcp_good" and dist > 5:
                        ignore_nums.append(num_obs)
                    if task_name == "lotka_volterra_good" and dist > 0.1:
                        ignore_nums.append(num_obs)
                    if task_name == "sir_good" and dist > 0.01:
                        ignore_nums.append(num_obs)

        if metric == "mmd_to_dirac":
            for num_obs in NUM_OBSERVATION_LIST:
                # mmd to dirac
                theta_true = torch.load(
                    PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl"
                )[num_obs - 1]
                dist = dist_to_dirac(
                    samples[num_obs],
                    theta_true,
                    percentage=0,
                    scaled=True,
                )["mmd"]

                dist_list.append(dist)

                if torch.isnan(dist):
                    ignore_nums.append(num_obs)

        torch.save(dist_list, filename)

    if not load:
        print()
        print(f"Computed {metric} for {len(dist_list)} observations.")
        print(f"NaNs in {len(ignore_nums)} observations: {ignore_nums}.")

    dist_dict = {
        "mean": torch.tensor(dist_list).mean(),
        "std": torch.tensor(dist_list).std(),
    }

    return dist_dict, ignore_nums


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--sampler", type=str, required=True)
    parser.add_argument("--compute_dist", action="store_true")
    parser.add_argument("--plot_dist", action="store_true")
    parser.add_argument("--plot_samples", action="store_true")

    parser.add_argument("--swd", action="store_true")
    parser.add_argument("--mmd", action="store_true")
    parser.add_argument("--c2st", action="store_true")
    parser.add_argument("--dirac", action="store_true")

    args = parser.parse_args()

    alpha, alpha_fill = set_plotting_style()

    if args.losses:
        lr_list = [1e-4, 1e-3]
        bs_list = [256, 64]

        # plot losses to select lr and bs
        n_rows = len(TASKS)
        n_cols = len(N_TRAIN)
        for bs in bs_list:
            fig, axs = plt.subplots(
                n_rows,
                n_cols,
                figsize=(5 * n_cols, 5 * n_rows),
                sharex=True,
                sharey=True,
            )  # , constrained_layout=True)
            fig.subplots_adjust(
                right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1
            )
            for i, task_name in enumerate(TASKS.keys()):
                for j, n_train in enumerate(N_TRAIN):
                    best_val_loss = {bs: {} for bs in bs_list}
                    for lr_, c in zip(lr_list, ["blue", "orange"]):
                        train_losses, val_losses, best_epoch = load_losses(
                            task_name, n_train, lr_, path=PATH_EXPERIMENT, batch_size=bs
                        )
                        axs[i, j].plot(
                            train_losses,
                            label=f"train, lr={lr_}",
                            color=c,
                            linewidth=3,
                            alpha=0.3,
                        )
                        axs[i, j].plot(
                            val_losses,
                            label=f"val, lr={lr_}",
                            color=c,
                            linewidth=3,
                            alpha=0.9,
                        )
                        axs[i, j].axvline(
                            best_epoch,
                            color=c,
                            linestyle="--",
                            linewidth=5,
                            alpha=0.9,
                            zorder=10000,
                        )

                    # set ax title as N_train only at the top
                    if i == 0:
                        axs[i, j].set_title(r"$N_\mathrm{train}$" + rf"$={n_train}$")
                    # label xaxis only at the bottom
                    if i == len(TASKS) - 1:
                        axs[i, j].set_xlabel("epochs")
                    # label yaxis only at the left
                    if j == 0:
                        axs[i, j].set_ylabel(TASKS[task_name])
                    axs[i, j].set_ylim([0, 0.5])
            axs[i, j].legend()
            plt.savefig(PATH_EXPERIMENT + f"_plots_new/losses_bs_{bs}.png")
            plt.savefig(PATH_EXPERIMENT + f"_plots_new/sbibm_losses_bs_{bs}.pdf")
            plt.clf()

        # print losses to select lr and bs
        for i, task_name in enumerate(TASKS.keys()):
            for j, n_train in enumerate(N_TRAIN):
                best_val_loss = {bs: {} for bs in bs_list}
                for lr_, c in zip(lr_list, ["blue", "orange"]):
                    for bs in bs_list:
                        train_losses, val_losses, best_epoch = load_losses(
                            task_name, n_train, lr_, path=PATH_EXPERIMENT, batch_size=bs
                        )
                        best_val_loss[bs][lr_] = val_losses[best_epoch]

                # get lr for min best val loss
                for bs in bs_list:
                    best_lr = min(best_val_loss[bs], key=best_val_loss[bs].get)
                    print(
                        f"best lr for {task_name}, {n_train}, {bs}: lr = {best_lr}, val losses = {best_val_loss[bs]}"
                    )

    if args.compute_dist:
        IGNORE_NUMS = {
            "slcp_good": {method: [] for method in METHODS_STYLE.keys()},
            "lotka_volterra_good": {method: [] for method in METHODS_STYLE.keys()},
            "sir_good": {method: [] for method in METHODS_STYLE.keys()},
        }

        final_ignore_nums = []
        for metric in METRICS:
            print(f"Computing {metric}...")
            for task_name in TASKS.keys():
                for n_train in N_TRAIN:
                    for n_obs in N_OBS:
                        for method in METHODS_STYLE.keys():
                            if not (
                                method == "JAC"
                                and task_name == "slcp_good"
                                and n_train == 30000
                            ):  # NaNs in samples
                                dist, ignore_nums = compute_mean_distance(
                                    task_name=task_name,
                                    n_train=n_train,
                                    n_obs=n_obs,
                                    cov_mode=method.split("_")[0],
                                    sampler=args.sampler,
                                    langevin=True if "LANGEVIN" in method else False,
                                    clip=True if "clip" in method else False,
                                    ours=True if "ours" in method else False,
                                    metric=metric,
                                    load=False,
                                )
                                dist_mean = dist["mean"]
                                dist_std = dist["std"]
                                print(
                                    f"{task_name}, {n_train}, {n_obs}, {method}: {dist_mean}, {dist_std}"
                                )

                                for num in ignore_nums:
                                    if num not in IGNORE_NUMS[task_name][method]:
                                        IGNORE_NUMS[task_name][method].append(num)
                                        if (
                                            metric in ["swd", "mmd"]
                                            and method in ["GAUSS"]
                                            and num not in final_ignore_nums
                                        ):
                                            final_ignore_nums.append(num)

            torch.save(
                IGNORE_NUMS,
                PATH_EXPERIMENT + f"_plots_new/ignore_nums_{metric}_{args.sampler}.pkl",
            )
            print()
            print(f"Ignored observations: {IGNORE_NUMS}")
            print()

        # final_ignore_nums.append(14) # for sir
        torch.save(
            final_ignore_nums,
            PATH_EXPERIMENT + f"_plots_new/ignore_nums_final_{args.sampler}.pkl",
        )
        print()
        print(f"Final ignored observations: {final_ignore_nums}")
        print()

    if args.plot_dist:
        prec_ignore_nums = torch.load(
            PATH_EXPERIMENT + f"_plots_new/ignore_nums_final_{args.sampler}.pkl"
        )
        print()
        print(f"Ignored observations: {prec_ignore_nums}")
        print()

        if args.swd or args.mmd or args.dirac or args.c2st:
            metric = (
                "swd"
                if args.swd
                else "mmd"
                if args.mmd
                else "mmd_to_dirac"
                if args.dirac
                else "c2st"
            )
            task_names = TASKS.keys()

            # plot mean distance as function of n_train
            n_rows = len(task_names)
            n_cols = len(N_OBS)
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True
            )
            fig.subplots_adjust(
                right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1
            )
            for i, task_name in enumerate(task_names):
                for j, n_obs in tqdm(enumerate(N_OBS), desc=f"{task_name}, {metric}"):
                    mean_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                    std_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                    for n_train in N_TRAIN:
                        for method in METHODS_STYLE.keys():
                            if (
                                method == "JAC"
                                and task_name == "slcp_good"
                                and n_train == 30000
                            ):
                                mean_dist_dict[method].append(np.nan)
                                std_dist_dict[method].append(np.nan)
                            else:
                                dist, _ = compute_mean_distance(
                                    task_name=task_name,
                                    n_train=n_train,
                                    n_obs=n_obs,
                                    cov_mode=method.split("_")[0],
                                    sampler=args.sampler,
                                    langevin=True if "LANGEVIN" in method else False,
                                    clip=True if "clip" in method else False,
                                    ours=True if "ours" in method else False,
                                    metric=metric,
                                    load=True,
                                    prec_ignore_nums=prec_ignore_nums,
                                )
                                mean_dist_dict[method].append(dist["mean"])
                                std_dist_dict[method].append(dist["std"])

                    for k, mean_, std_ in zip(
                        mean_dist_dict.keys(),
                        mean_dist_dict.values(),
                        std_dist_dict.values(),
                    ):
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
                        axs[i, j].set_title(r"$n$" + rf"$={n_obs}$")
                    # label xaxis only at the bottom
                    if i == len(task_names) - 1:
                        axs[i, j].set_xlabel(r"$N_\mathrm{train}$")
                        axs[i, j].set_xscale("log")
                        axs[i, j].set_xticks(N_TRAIN)
                    # label yaxis only at the left
                    if j == 0:
                        axs[i, j].set_ylabel(
                            TASKS[task_name] + "\n" + METRICS_STYLE[metric]["label"]
                        )
                    else:
                        axs[i, j].set_yticklabels([])
                    if metric == "mmd":
                        if "lotka" in task_name:
                            axs[i, j].set_ylim([0, 1.5])
                        if "sir" in task_name:
                            axs[i, j].set_ylim([0, 1.0])
                        if "slcp" in task_name:
                            axs[i, j].set_ylim([0, 0.6])

                    elif metric == "swd":
                        if "lotka" in task_name:
                            axs[i, j].set_ylim([0, 0.05])
                        if "sir" in task_name:
                            axs[i, j].set_ylim([0, 0.005])
                        if "slcp" in task_name:
                            axs[i, j].set_ylim([0, 2])
                    elif metric == "mmd_to_dirac":
                        if "lotka" in task_name:
                            axs[i, j].set_ylim([0, 0.5])
                        if "sir" in task_name:
                            axs[i, j].set_ylim([0, 0.05])
                        if "slcp" in task_name:
                            axs[i, j].set_ylim([0, 30])
                    else:
                        axs[i, j].set_ylim([0, 1])

            handles, labels = axs[0, 0].get_legend_handles_labels()
            plt.legend(handles, labels, loc="lower right", prop={"family": "monospace"})

            plt.savefig(
                PATH_EXPERIMENT + f"_plots_new/{metric}_n_train_{args.sampler}.png"
            )
            plt.savefig(
                PATH_EXPERIMENT + f"_plots_new/{metric}_n_train_{args.sampler}.pdf"
            )
            plt.clf()

            # plot mean distance as function of n_obs
            n_rows = len(task_names)
            n_cols = len(N_TRAIN)
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True
            )  # , constrained_layout=True)
            fig.subplots_adjust(
                right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1
            )
            for i, task_name in enumerate(task_names):
                for j, n_train in tqdm(
                    enumerate(N_TRAIN), desc=f"{task_name}, {metric}"
                ):
                    mean_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                    std_dist_dict = {method: [] for method in METHODS_STYLE.keys()}
                    for n_obs in N_OBS:
                        for method in METHODS_STYLE.keys():
                            if (
                                method == "JAC"
                                and task_name == "slcp_good"
                                and n_train == 30000
                            ):
                                mean_dist_dict[method].append(np.nan)
                                std_dist_dict[method].append(np.nan)
                            else:
                                dist, ignore_nums = compute_mean_distance(
                                    task_name=task_name,
                                    n_train=n_train,
                                    n_obs=n_obs,
                                    cov_mode=method.split("_")[0],
                                    sampler=args.sampler,
                                    langevin=True if "LANGEVIN" in method else False,
                                    clip=True if "clip" in method else False,
                                    ours=True if "ours" in method else False,
                                    metric=metric,
                                    load=True,
                                    prec_ignore_nums=prec_ignore_nums,
                                )
                                mean_dist_dict[method].append(dist["mean"])
                                std_dist_dict[method].append(dist["std"])

                    for k, mean_, std_ in zip(
                        mean_dist_dict.keys(),
                        mean_dist_dict.values(),
                        std_dist_dict.values(),
                    ):
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
                        axs[i, j].set_title(r"$N_\mathrm{train}$" + rf"$={n_train}$")
                    # label xaxis only at the bottom
                    if i == len(task_names) - 1:
                        axs[i, j].set_xlabel(r"$n$")
                        axs[i, j].set_xticks(N_OBS)
                    # label yaxis only at the left
                    if j == 0:
                        axs[i, j].set_ylabel(
                            TASKS[task_name] + "\n" + METRICS_STYLE[metric]["label"]
                        )
                    else:
                        axs[i, j].set_yticklabels([])
                    if metric == "swd":
                        if "lotka" in task_name:
                            axs[i, j].set_ylim([0, 0.05])
                        if "sir" in task_name:
                            axs[i, j].set_ylim([0, 0.005])
                        if "slcp" in task_name:
                            axs[i, j].set_ylim([0, 2])
                    elif metric == "mmd":
                        if "lotka" in task_name:
                            axs[i, j].set_ylim([0, 1.5])
                        if "sir" in task_name:
                            axs[i, j].set_ylim([0, 1.0])
                        if "slcp" in task_name:
                            axs[i, j].set_ylim([0, 0.6])
                    elif metric == "mmd_to_dirac":
                        if "lotka" in task_name:
                            axs[i, j].set_ylim([0, 0.5])
                        if "sir" in task_name:
                            axs[i, j].set_ylim([0, 0.05])
                        if "slcp" in task_name:
                            axs[i, j].set_ylim([0, 30])
                    else:
                        axs[i, j].set_ylim([0, 1])
            handles, labels = axs[0, 0].get_legend_handles_labels()
            plt.legend(handles, labels, loc="lower right", prop={"family": "monospace"})

            plt.savefig(
                PATH_EXPERIMENT + f"_plots_new/{metric}_n_obs_{args.sampler}.png"
            )
            plt.savefig(
                PATH_EXPERIMENT + f"_plots_new/{metric}_n_obs_{args.sampler}.pdf"
            )
            plt.clf()

    if args.plot_samples:
        n_train = 30000
        task_name = "slcp_good"
        save_path = PATH_EXPERIMENT + f"_samples/{task_name}/"
        os.makedirs(save_path, exist_ok=True)

        # # Analytical
        # for j, num_obs in enumerate(NUM_OBSERVATION_LIST):
        #     theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl")[num_obs-1]
        #     if theta_true.ndim > 1:
        #         theta_true = theta_true[0]
        #     for n_obs in N_OBS:
        #         samples_ref = load_reference_samples(task_name, n_obs)
        #         plt.scatter(samples_ref[num_obs][:, 0], samples_ref[num_obs][:, 1], alpha=0.5)
        #     plt.scatter(theta_true[0], theta_true[1], color="black")
        #     plt.savefig(save_path + f"analytical_num_obs_{num_obs}_prior.png")
        #     plt.clf()

        # # Analytical vs. GAUSS first dims
        # num_obs_lists = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
        # for num_obs_list in num_obs_lists:
        #     fig, axs = plt.subplots(len(num_obs_list), 5, figsize=(25, 5 * len(num_obs_list)))
        #     fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
        #     for j, num_obs in enumerate(num_obs_list):
        #         theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl")[num_obs-1]
        #         if theta_true.ndim > 1:
        #             theta_true = theta_true[0]

        #         for i, n_obs in enumerate(N_OBS):
        #             samples_ref = load_reference_samples(task_name, n_obs)
        #             samples_gauss = load_samples(task_name, n_train, n_obs=n_obs, cov_mode="GAUSS", clip=False, sampler=args.sampler)
        #             # samples_jac = load_samples(task_name, n_train, n_obs=n_obs, cov_mode="JAC", clip=False, sampler=args.sampler)
        #             samples_langevin = load_samples(task_name, n_train, n_obs=n_obs, langevin=True, clip=False)
        #             axs[j, i].scatter(samples_ref[num_obs][:, 0], samples_ref[num_obs][:, 1], alpha=0.5, label=f"ANALYTIC", color="lightgreen")
        #             axs[j, i].scatter(samples_gauss[num_obs][:, 0], samples_gauss[num_obs][:, 1], alpha=0.5, label=f"GAUSS", color="blue")
        #             # axs[j, i].scatter(samples_jac[num_obs][:, 0], samples_jac[num_obs][:, 1], alpha=0.5, label=f"JAC", color="orange")
        #             axs[j, i].scatter(samples_langevin[num_obs][:, 0], samples_langevin[num_obs][:, 1], alpha=0.1, label=f"LANGEVIN", color="#92374D")
        #             axs[j, i].scatter(theta_true[0], theta_true[1], color="black", s=100, label="True parameter")
        #             axs[j, i].set_xticks([])
        #             axs[j, i].set_yticks([])
        #             if j == 0:
        #                 axs[j, i].set_title(fr"$n = {n_obs}$")
        #             if i == 0:
        #                 axs[j, i].set_ylabel(f"num_obs = {num_obs}")
        #     plt.legend()
        #     plt.savefig(save_path + f"all_n_train_{n_train}_prior_num_{num_obs_list[0]}_{num_obs_list[-1]}_{args.sampler}.png")
        #     plt.savefig(save_path + f"all_n_train_{n_train}_prior_num_{num_obs_list[0]}_{num_obs_list[-1]}_{args.sampler}.pdf")
        #     plt.clf()

        # # Pairplot for all methods with increasing n_obs
        # from plot_utils import pairplot_with_groundtruth_md
        # for num_obs in np.arange(1, 26):
        #     theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl")[num_obs-1]
        #     for method, color in zip(["TRUE", "GAUSS", "JAC_clip", "LANGEVIN"], ["Greens", "Blues", "Oranges", "Reds"]):
        #         samples = []
        #         for n_obs in N_OBS:
        #             if method == "TRUE":
        #                 samples.append(load_reference_samples(task_name, n_obs)[num_obs])
        #             else:
        #                 samples.append(load_samples(task_name, n_train, n_obs=n_obs, cov_mode=method.split("_")[0], sampler=args.sampler, langevin=True if "LANGEVIN" in method else False, clip=True if "clip" in method else False)[num_obs])

        #         colors = cm.get_cmap(color)(np.linspace(1, 0.2, len(samples))).tolist()
        #         labels = [rf"$n = {n_obs}$" for n_obs in N_OBS]
        #         pairplot_with_groundtruth_md(
        #             samples_list=samples,
        #             labels=labels,
        #             colors=colors,
        #             theta_true=theta_true,
        #             ignore_ticks=True,
        #             ignore_xylabels=True,
        #             legend=False,
        #             size=5.5,
        #             title = METHODS_STYLE[method]["label"] if method != "TRUE" else "TRUE",
        #         )
        #         plt.savefig(save_path + f"num_{num_obs}_{method}_pairplot_n_train_{n_train}_{args.sampler}.png")
        #         plt.savefig(save_path + f"num_{num_obs}_{method}_pairplot_n_train_{n_train}_{args.sampler}.pdf")
        #         plt.clf()

        # Analytical vs. algorithms pairplot for all n_obs
        from plot_utils import pairplot_with_groundtruth_md

        for n_obs in N_OBS:
            samples_ref = load_reference_samples(task_name, n_obs)
            samples_gauss = load_samples(
                task_name,
                n_train,
                n_obs=n_obs,
                cov_mode="GAUSS",
                clip=False,
                sampler=args.sampler,
            )
            # samples_jac_clip = load_samples(task_name, n_train, n_obs=n_obs, cov_mode="JAC", clip=True)
            samples_langevin = load_samples(
                task_name, n_train, n_obs=n_obs, langevin=True, clip=False
            )
            for num_obs in [1]:  # np.arange(1, 26):
                theta_true = torch.load(
                    PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl"
                )[num_obs - 1]
                samples_list = [
                    samples_ref[num_obs],
                    samples_gauss[num_obs],
                    samples_langevin[num_obs],
                ]  # , samples_jac_clip[num_obs], samples_langevin[num_obs]]
                pairplot_with_groundtruth_md(
                    samples_list=samples_list,
                    labels=[
                        "ANALYTIC",
                        "GAUSS",
                        "LANGEVIN",
                    ],  # , "JAC (clip)", "LANGEVIN"],
                    colors=["lightgreen", "blue", "#92374D"],  # , "orange", "#92374D"],
                    theta_true=theta_true,
                    ignore_ticks=True,
                    ignore_xylabels=True,
                    size=5.5,
                )
                plt.savefig(
                    save_path
                    + f"pairplot_n_train_{n_train}_num_{num_obs}_n_obs_{n_obs}_{args.sampler}.png"
                )
                plt.savefig(
                    save_path
                    + f"pairplot_n_train_{n_train}_num_{num_obs}_n_obs_{n_obs}_{args.sampler}.pdf"
                )
                plt.clf()

    # # std of swd
    # task_name = "sir_good"
    # for n_train in N_TRAIN:
    #     n_obs = 8
    #     samples_ref = load_reference_samples(task_name, n_obs)
    #     samples_gauss = load_samples(task_name, n_train, n_obs=n_obs, cov_mode="GAUSS", clip=False)
    #     sw = []
    #     for num_obs in np.arange(1, 26):
    #         if num_obs in FINAL_IGNORE_NUMS: # or num_obs == 14:
    #             continue
    #         theta_true = torch.load(PATH_EXPERIMENT + f"{task_name}/theta_true_list_prior.pkl")[num_obs-1]
    #         # sw_seed = []
    #         # for i in range(10):
    #         sw_small = sliced_wasserstein_distance(
    #             np.array(samples_ref[num_obs]), np.array(samples_gauss[num_obs]), n_projections=100
    #         )
    #         # sw_seed.append(sw_small)
    #         # sw_mean = np.mean(sw_seed)
    #         # sw_std = np.std(sw_seed)
    #         # print("swd: ", sw_mean, sw_std)
    #         sw.append(sw_small)

    #         # plt.scatter(samples_ref[num_obs][:, 0], samples_ref[num_obs][:, 1], alpha=0.5, label=f"ANALYTIC", color="lightgreen")
    #         # plt.scatter(samples_gauss[num_obs][:, 0], samples_gauss[num_obs][:, 1], alpha=0.5, label=f"GAUSS", color="blue")
    #         # plt.scatter(theta_true[0], theta_true[1], color="black", s=100, label="True parameter")
    #         # plt.legend()
    #         # plt.savefig(f"num_obs_{num_obs}_n_obs_{n_obs}_n_train_{n_train}_swd.png")
    #         # plt.clf()
    #     print("mean swd: ", np.mean(sw), np.std(sw))
