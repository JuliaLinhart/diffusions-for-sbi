import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from tasks.sbibm import get_task

from experiment_utils import dist_to_dirac
from ot import sliced_wasserstein_distance
from sbibm.metrics import mmd, c2st

from matplotlib import colormaps as cm
from plot_utils import (
    set_plotting_style,
    METHODS_STYLE,
    plots_dist_n_train,
    plots_dist_n_obs,
    plots_dist_n_train_pf_nse,
    pairplot_with_groundtruth_md,
)

from tqdm import tqdm

# seed
torch.manual_seed(42)

PATH_EXPERIMENT = "results/sbibm/"
TASKS_MAIN = {
    "slcp": "SLCP", 
    "lotka_volterra": "Lotka-Volterra", 
    "sir": "SIR", 
}
TASKS_EXTRA = {
    "gaussian_linear": "Gaussian",
    "gaussian_mixture": "GMM", 
    "gaussian_mixture_uniform": "GMM (uniform)", 
    "bernoulli_glm": "B-GLM",
    "bernoulli_glm_raw": "B-GLM (raw)", 
    "two_moons": "Two Moons", 
}

N_TRAIN = [1000, 3000, 10000, 30000]
BATCH_SIZE = 256  # 64
N_EPOCHS = 5000
LR = 1e-4

TASKS_DICT = {
    "gaussian_linear": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256]},
    "gaussian_mixture": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256]},
    "gaussian_mixture_uniform": {
        "lr": [1e-4, 1e-4, 1e-4, 1e-4], 
        "bs": [256, 256, 256, 256],
    },
    "bernoulli_glm": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256]},
    "bernoulli_glm_raw": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256]},
    "two_moons": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [64, 64, 64, 64]},
    "slcp": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256]},
    "lotka_volterra": {"lr": [1e-3, 1e-3, 1e-3, 1e-3], "bs": [256, 256, 256, 256]},
    "sir": {"lr": [1e-4, 1e-4, 1e-4, 1e-4], "bs": [256, 256, 256, 256]},
}

N_OBS = [1, 8, 14, 22, 30]
NUM_OBSERVATION_LIST = list(np.arange(1, 26))
N_MAX_LIST = [3,6,30] # for pf_nse

METRICS = ["mmd", "swd", "mmd_to_dirac"]


def load_losses(task_name, n_train, lr, path, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    path = (
        path
        + f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{n_epochs}_lr_{lr}/"
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
    sampler="ddim",
    langevin=False,
    ours=False,
    clip=False,
    clf_free_guidance=False,
    pf_nse=False,
    n_max=1,
):
    batch_size = TASKS_DICT[task_name]["bs"][N_TRAIN.index(n_train)]
    lr = TASKS_DICT[task_name]["lr"][N_TRAIN.index(n_train)]
    path = PATH_EXPERIMENT+ f"{task_name}/n_train_{n_train}_bs_{batch_size}_n_epochs_{N_EPOCHS}_lr_{lr}/"
    
    if pf_nse:
        path = path + f"pf_n_max_{n_max}/"
    if clf_free_guidance:
        path = path + f"clf_free_guidance/"
    
    if langevin:
        path = path + "langevin_steps_400_5/"
        if ours:
            path = path[:-1] + "_ours/"
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
    num_obs,
    n_obs,
    cov_mode=None,
    sampler="ddim",
    langevin=False,
    ours=False,
    clip=False,
    clf_free_guidance=False,
    pf_nse=False,
    n_max=1,
):
    filename = path_to_results(
        task_name,
        "posterior_samples",
        num_obs,
        n_train,
        n_obs,
        cov_mode,
        sampler,
        langevin,
        ours,
        clip,
        clf_free_guidance,
        pf_nse,
        n_max,
    )
    samples = torch.load(filename)
    return samples


# compute mean distance to true theta over all observations
def compute_mean_distance(
    metric,
    task_name,
    n_train,
    n_obs,
    cov_mode=None,
    sampler="ddim",
    langevin=False,
    ours=False,
    clip=False,
    clf_free_guidance=False,
    pf_nse=False,
    n_max=1,
    load=False,
    prec_ignore_nums=None,
):
    # load results if already computed
    save_path = PATH_EXPERIMENT+ f"{task_name}/metrics/cov_mode_{cov_mode}_langevin_{langevin}_clip_{clip}/"
    if ours:
        save_path = save_path[:-1] + "_ours/"
    if pf_nse:
        save_path = save_path + f"pf_n_max_{n_max}/"
    if clf_free_guidance:
        save_path = save_path + "clf_free_guidance/"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    filename = save_path + f"n_train_{n_train}_n_obs_{n_obs}_metric_{metric}.pkl"
    if load:
        dist_list_all = torch.load(filename)
        dist_list = []

        ignore_nums = []

        for num_obs in NUM_OBSERVATION_LIST:
            dist = dist_list_all[num_obs - 1]
            if num_obs not in prec_ignore_nums:
                dist_list.append(dist)
            if torch.isnan(dist):
                ignore_nums.append(num_obs)
            if metric == "mmd" and n_train > 3000:
                if task_name == "two_moons" and dist > 0.2:
                    ignore_nums.append(num_obs)
            if metric == "swd" and n_train > 3000:
                if task_name == "gaussian_linear" and dist > 0.8:
                    ignore_nums.append(num_obs)
                if task_name == "gaussian_mixture" and dist > 1:
                    ignore_nums.append(num_obs)
                if task_name == "gaussian_mixture_uniform" and dist > 2:
                    ignore_nums.append(num_obs)
                if task_name == "two_moons" and dist > 0.4:
                    ignore_nums.append(num_obs)
                if "bernoulli_glm" in task_name and dist > 1:
                    ignore_nums.append(num_obs)
                if "sir" in task_name and dist > 0.02:
                    ignore_nums.append(num_obs)
                if "slcp" in task_name and dist > 2:
                    ignore_nums.append(num_obs)
                if "lotka" in task_name and dist > 0.05:
                    ignore_nums.append(num_obs)
                

    else:
        task = get_task(task_name, save_path="tasks/sbibm/data/")

        dist_list = []
        ignore_nums = []
        if metric in ["mmd", "swd", "c2st"]:
            for num_obs in NUM_OBSERVATION_LIST:
                samples_ref = task.get_reference_posterior_samples(num_obs, n_obs, verbose=False)
                samples = load_samples(
                    task_name,
                    n_train=n_train,
                    num_obs=num_obs,
                    n_obs=n_obs,
                    cov_mode=cov_mode,
                    langevin=langevin,
                    ours=ours,
                    clip=clip,
                    clf_free_guidance=clf_free_guidance,
                    pf_nse=pf_nse,
                    n_max=n_max,
                    sampler=sampler,
                )
                # mmd
                if metric == "mmd":
                    dist = mmd(torch.tensor(np.array(samples_ref)), samples)
                # sliced wasserstein distance
                if metric == "swd":
                    dist = sliced_wasserstein_distance(
                        np.array(samples_ref), np.array(samples), n_projections=1000
                    )
                    dist = torch.tensor(dist)

                if metric == "c2st":
                    dist = c2st(torch.tensor(np.array(samples_ref)), samples)

                dist_list.append(dist)

                if torch.isnan(dist):
                    ignore_nums.append(num_obs)

        if metric == "mmd_to_dirac":
            for num_obs in NUM_OBSERVATION_LIST:
                # mmd to dirac
                theta_true = task.get_reference_parameters(verbose=False)[num_obs - 1]
                samples = load_samples(
                    task_name,
                    n_train=n_train,
                    num_obs=num_obs,
                    n_obs=n_obs,
                    cov_mode=cov_mode,
                    langevin=langevin,
                    ours=ours,
                    clip=clip,
                    clf_free_guidance=clf_free_guidance,
                    pf_nse=pf_nse,
                    n_max=n_max,
                    sampler=sampler,
                )
                dist = dist_to_dirac(
                    samples,
                    theta_true,
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
    parser.add_argument("--compute_dist", action="store_true")
    parser.add_argument("--plot_dist", action="store_true")
    parser.add_argument("--plot_samples", action="store_true")

    parser.add_argument("--swd", action="store_true")
    parser.add_argument("--mmd", action="store_true")
    parser.add_argument("--c2st", action="store_true")
    parser.add_argument("--dirac", action="store_true")

    parser.add_argument("--tasks", type=str, default="main", choices=["main", "extra", "all"])
    parser.add_argument("--clf_free_guidance", action="store_true")
    parser.add_argument("--langevin_comparison", action="store_true")
    parser.add_argument("--pf_nse", action="store_true")

    args = parser.parse_args()

    alpha, alpha_fill = set_plotting_style()

    if args.tasks == "main":
        TASKS = TASKS_MAIN
    elif args.tasks == "extra":
        TASKS = TASKS_EXTRA
    else:
        TASKS = {**TASKS_MAIN, **TASKS_EXTRA}

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
            plt.savefig(PATH_EXPERIMENT + f"_plots_rebuttal/losses_bs_{bs}_{args.tasks}.png")
            plt.savefig(PATH_EXPERIMENT + f"_plots_rebuttal/sbibm_losses_bs_{bs}_{args.tasks}.pdf")
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
        n_max_list = [1]

        if args.clf_free_guidance:
            tasks_dict = TASKS_MAIN
            method_names = ["GAUSS_cfg"]
        elif args.langevin_comparison:
            tasks_dict = TASKS_MAIN
            method_names = ["LANGEVIN", "LANGEVIN_clip"]
        elif args.pf_nse:
            # no sir
            tasks_dict = TASKS_MAIN 
            del tasks_dict["sir"]
            method_names = ["LANGEVIN"] #, "LANGEVIN_clip"] #"GAUSS_clip", "GAUSS", 
            N_OBS = [30]
            n_max_list = N_MAX_LIST
        else:
            tasks_dict= {**TASKS_MAIN, **TASKS_EXTRA}
            method_names = METHODS_STYLE.keys()
            method_names = [method for method in method_names if "cfg" not in method and "ours" not in method]

        IGNORE_NUMS = {task_name: [] for task_name in tasks_dict.keys()}

        final_ignore_nums = []
        for metric in METRICS:
            print(f"Computing {metric}...")
            for task_name in tasks_dict.keys():
                for n_train in N_TRAIN:
                    for n_obs in N_OBS:
                        for n_max in n_max_list:
                            for method in method_names:
                                dist, ignore_nums = compute_mean_distance(
                                    task_name=task_name,
                                    n_train=n_train,
                                    n_obs=n_obs,
                                    cov_mode=method.split("_")[0],
                                    langevin=True if "LANGEVIN" in method else False,
                                    ours=True if "ours" in method else False,
                                    clip=True if "clip" in method else False,
                                    clf_free_guidance=args.clf_free_guidance,
                                    pf_nse=args.pf_nse,
                                    n_max=n_max,
                                    metric=metric,
                                    load=False,
                                )
                                dist_mean = dist["mean"]
                                dist_std = dist["std"]
                                print(
                                    f"{task_name}, {n_train}, {n_obs}, {n_max}, {method}: {dist_mean}, {dist_std}"
                                )

                                if method == "GAUSS":
                                    for num in ignore_nums:
                                        if num not in IGNORE_NUMS[task_name]:
                                            IGNORE_NUMS[task_name].append(num)
                                            if (
                                                metric in ["swd", "mmd"]
                                                and num not in final_ignore_nums
                                            ):
                                                final_ignore_nums.append(num)

        print()
        print(f"Ignored observations: {IGNORE_NUMS}")
        print()

        print()
        print(f"Final ignored observations: {final_ignore_nums}")
        print()

        if not args.clf_free_guidance and not args.langevin_comparison and not args.pf_nse:
            torch.save(IGNORE_NUMS, PATH_EXPERIMENT + f"ignore_nums_per_task.pkl")
            torch.save(
                final_ignore_nums,
                PATH_EXPERIMENT + f"ignore_nums_final.pkl",
            )

    if args.plot_dist:
        prec_ignore_nums = torch.load(
            # PATH_EXPERIMENT + f"ignore_nums_final.pkl"
            PATH_EXPERIMENT + f"ignore_nums_per_task.pkl"
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
            if args.clf_free_guidance:
                tasks_dict = TASKS_MAIN
                method_names = ["GAUSS", "GAUSS_cfg"]
                title_ext = "_cfg"
            elif args.langevin_comparison:
                tasks_dict = TASKS_MAIN
                method_names = ["LANGEVIN", "LANGEVIN_clip", "LANGEVIN_ours", "LANGEVIN_ours_clip"]
                title_ext = "_lgv_comp"
            elif args.pf_nse:
                tasks_dict = TASKS_MAIN
                del tasks_dict["sir"]
                method_names = ["GAUSS", "LANGEVIN"]
            else:
                tasks_dict = TASKS
                method_names = METHODS_STYLE.keys()
                # remove cfg 
                method_names = [method for method in method_names if "cfg" not in method and "ours" not in method]
                title_ext = f"_{args.tasks}"
            
            if not args.pf_nse:
                # plot mean distance as function of n_train
                fig = plots_dist_n_train(
                    metric, 
                    tasks_dict, 
                    method_names, 
                    prec_ignore_nums, 
                    N_TRAIN, 
                    N_OBS, 
                    compute_dist_fn=compute_mean_distance,
                    title_ext=title_ext,
                )
                plt.savefig(
                    PATH_EXPERIMENT + f"_plots_rebuttal/{metric}_n_train{title_ext}.png"
                )
                plt.savefig(
                    PATH_EXPERIMENT + f"_plots_rebuttal/{metric}_n_train{title_ext}.pdf"
                )
                plt.clf()
                # plot mean distance as function of n_obs
                fig = plots_dist_n_obs(
                    metric, 
                    tasks_dict, 
                    method_names, 
                    prec_ignore_nums, 
                    N_TRAIN,
                    N_OBS,
                    compute_dist_fn=compute_mean_distance,
                    title_ext=title_ext,
                )
                plt.savefig(
                    PATH_EXPERIMENT + f"_plots_rebuttal/{metric}_n_obs{title_ext}.png"
                )
                plt.savefig(
                    PATH_EXPERIMENT + f"_plots_rebuttal/{metric}_n_obs{title_ext}.pdf"
                )
                plt.clf()
            else:
                fig = plots_dist_n_train_pf_nse(
                    metric, 
                    tasks_dict, 
                    method_names, 
                    prec_ignore_nums, 
                    N_TRAIN, 
                    N_MAX_LIST,
                    compute_dist_fn=compute_mean_distance,
                )
                plt.savefig(
                    PATH_EXPERIMENT + f"_plots_rebuttal/{metric}_n_train_pf_nse.png"
                )
                plt.savefig(
                    PATH_EXPERIMENT + f"_plots_rebuttal/{metric}_n_train_pf_nse.pdf"
                )
                plt.clf()
            

    if args.plot_samples:
        n_train = 30000
        task_name = "slcp"
        task = get_task(task_name, save_path="tasks/sbibm/data/")
        save_path = PATH_EXPERIMENT + f"_samples/{task_name}/"
        os.makedirs(save_path, exist_ok=True)

        # Pairplot for all methods with increasing n_obs
        from plot_utils import pairplot_with_groundtruth_md

        for num_obs in np.arange(1, 26):
            theta_true = task.get_reference_parameters(verbose=False)[num_obs - 1]
            for method, color in zip(
                ["TRUE", "GAUSS", "JAC_clip", "LANGEVIN"],
                ["Greens", "Blues", "Oranges", "Reds"],
            ):
                samples = []
                for n_obs in N_OBS:
                    if method == "TRUE":
                        samples.append(
                            task.get_reference_posterior_samples(num_obs, n_obs, verbose=False)
                        )
                    else:
                        samples.append(
                            load_samples(
                                task_name,
                                n_train,
                                num_obs=num_obs,
                                n_obs=n_obs,
                                cov_mode=method.split("_")[0],
                                langevin=True if "LANGEVIN" in method else False,
                                clip=True if "clip" in method else False,
                            )
                        )

                colors = cm.get_cmap(color)(np.linspace(1, 0.2, len(samples))).tolist()
                labels = [rf"$n = {n_obs}$" for n_obs in N_OBS]
                pairplot_with_groundtruth_md(
                    samples_list=samples,
                    labels=labels,
                    colors=colors,
                    theta_true=theta_true,
                    ignore_ticks=True,
                    ignore_xylabels=True,
                    legend=False,
                    size=5.5,
                    title=METHODS_STYLE[method]["label"]
                    if method != "TRUE"
                    else "TRUE",
                )
                plt.savefig(
                    save_path
                    + f"num_{num_obs}_{method}_pairplot_n_train_{n_train}.png"
                )
                plt.savefig(
                    save_path
                    + f"num_{num_obs}_{method}_pairplot_n_train_{n_train}.pdf"
                )
                plt.clf()


    # # Analytical vs. GAUSS first dims
    # task_name = "two_moons"
    # task = get_task(task_name, save_path="tasks/sbibm/data/")

    # save_path = PATH_EXPERIMENT + f"_samples/{task_name}/"
    # os.makedirs(save_path, exist_ok=True)
    # n_train = 30000
    # num_obs_lists = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    # for num_obs_list in num_obs_lists:
    #     fig, axs = plt.subplots(len(num_obs_list), 5, figsize=(25, 5 * len(num_obs_list)))
    #     fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
    #     for j, num_obs in enumerate(num_obs_list):
    #         theta_true = task.get_reference_parameters(verbose=False)[num_obs - 1]
    #         if theta_true.ndim > 1:
    #             theta_true = theta_true[0]

    #         for i, n_obs in enumerate(N_OBS):
    #             samples_ref = task.get_reference_posterior_samples(num_obs, n_obs, verbose=False)
    #             samples_gauss = load_samples(task_name, n_train, num_obs=num_obs, n_obs=n_obs, cov_mode="GAUSS", clip=False)
    #             # samples_jac = load_samples(task_name, n_train, n_obs=n_obs, cov_mode="JAC", clip=False, sampler=args.sampler)
    #             # samples_langevin = load_samples(task_name, n_train, num_obs=num_obsn_obs=n_obs, langevin=True, clip=False)
    #             axs[j, i].scatter(samples_ref[:, 0], samples_ref[:, 1], alpha=0.5, label=f"ANALYTIC", color="lightgreen")
    #             axs[j, i].scatter(samples_gauss[:, 0], samples_gauss[:, 1], alpha=0.5, label=f"GAUSS", color="blue")
    #             # axs[j, i].scatter(samples_jac[:, 0], samples_jac[:, 1], alpha=0.5, label=f"JAC", color="orange")
    #             # axs[j, i].scatter(samples_langevin[:, 0], samples_langevin[:, 1], alpha=0.1, label=f"LANGEVIN", color="#92374D")
    #             axs[j, i].scatter(theta_true[0], theta_true[1], color="black", s=100, label="True parameter")
    #             axs[j, i].set_xticks([])
    #             axs[j, i].set_yticks([])
    #             if j == 0:
    #                 axs[j, i].set_title(fr"$n = {n_obs}$")
    #             if i == 0:
    #                 axs[j, i].set_ylabel(f"num_obs = {num_obs}")
    #     plt.legend()
    #     plt.savefig(save_path + f"all_n_train_{n_train}_prior_num_{num_obs_list[0]}_{num_obs_list[-1]}.png")
    #     plt.savefig(save_path + f"all_n_train_{n_train}_prior_num_{num_obs_list[0]}_{num_obs_list[-1]}.pdf")
    #     plt.clf()