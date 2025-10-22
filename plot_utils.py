import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from matplotlib import colormaps as cm
from tueplots import fonts, axes
from tqdm import tqdm


def set_plotting_style(size=5):
    style = fonts.neurips2022()
    # delete the font.serif key to use the default font
    del style["font.serif"]
    plt.rcParams.update(style)
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = size * 5
    plt.rcParams["xtick.labelsize"] = size * 5
    plt.rcParams["ytick.labelsize"] = size * 5
    plt.rcParams["axes.labelsize"] = size * 6
    plt.rcParams["font.size"] = size * 6
    plt.rcParams["axes.titlesize"] = size * 6
    alpha = 0.9
    alpha_fill = 0.1
    return alpha, alpha_fill


markersize = plt.rcParams["lines.markersize"] * 1.5

METHODS_STYLE = {
    "LANGEVIN": {
        "label": "LANGEVIN",
        "color": "#92374D",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": markersize,
    },
    "LANGEVIN_clip": {
        "label": "LANGEVIN (clip)",
        "color": "#92374D",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 4,
        "markersize": markersize,
    },
    "LANGEVIN_tamed": {
        "label": "tamed ULA",
        "color": "#E5A4CB",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": markersize,
    },
    "LANGEVIN_tamed_clip": {
        "label": "tamed ULA (clip)",
        "color": "#E5A4CB",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 4,
        "markersize": markersize,
    },
    "GAUSS": {
        "label": "GAUSS",
        "color": "blue",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": markersize + 10,
    },
    "GAUSS_clip": {
        "label": "GAUSS (clip)",
        "color": "blue",
        "marker": "*",
        "linestyle": "--",
        "linewidth": 4,
        "markersize": markersize + 10,
    },
    "GAUSS_cfg": {
        "label": "GAUSS (CFG)",
        "color": "#00BBFF",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 3,
        "markersize": markersize + 10,
    },
    "JAC": {"label":"JAC", "color": "orange", "marker": "^", "linestyle": "-", "linewidth":3, "markersize": markersize + 2},
    "JAC_clip": {
        "label": "JAC (clip)",
        "color": "orange",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 4,
        "markersize": markersize + 2,
    },
}

METRICS_STYLE = {
    "swd": {"label": "sW"},
    "mmd": {"label": "MMD"},
    "c2st": {"label": "C2ST"},
    "mmd_to_dirac": {"label": "MMD to Dirac"},
}

CMAPS_METHODS = {
    "LANGEVIN": "Reds",
    "LANGEVIN_clip": "Reds",
    "LANGEVIN_tamed": "Purples",
    "LANGEVIN_tamed_clip": "Purples",
    "GAUSS": "Blues",
    "GAUSS_clip": "Blues",
    "GAUSS_cfg": "Blues",
    "JAC": "Oranges",
    "JAC_clip": "Oranges",
}


# Plot learned posterior P(theta | x_obs)
def pairplot_with_groundtruth_md(
    samples_list,
    labels,
    colors,
    theta_true=None,
    param_names=None,
    title="",
    plot_bounds=None,
    ignore_ticks=False,
    ignore_xylabels=False,
    legend=True,
    size=5,
):
    # # adjust marker size
    # markersize = plt.rcParams['lines.markersize']
    # plt.rcParams['lines.markersize'] = markersize - (samples_list[0].shape[-1] * 0.2)

    columns = [rf"$\theta_{i+1}$" for i in range(len(samples_list[0][0]))]
    if param_names is not None:
        columns = param_names

    dfs = []
    for samples, label in zip(samples_list, labels):
        df = pd.DataFrame(samples, columns=columns)
        df["Distribution"] = label
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    pg = sns.PairGrid(
        dfs,
        hue="Distribution",
        palette=dict(zip(labels, colors)),
        corner=True,
        diag_sharey=False,
    )

    pg.fig.set_size_inches(size, size)

    pg.map_lower(sns.kdeplot, linewidths=2, constrained_layout=False, zorder=1)
    pg.map_diag(sns.kdeplot, fill=True, linewidths=2, alpha=0.1)

    if theta_true is not None:
        if theta_true.ndim > 1:
            theta_true = theta_true[0]
        dim = len(theta_true)
        for i in range(dim):
            # plot dirac on diagonal
            pg.axes.ravel()[i * (dim + 1)].axvline(
                x=theta_true[i], linewidth=2, ls="--", c="black"
            )
            # place above the kdeplots
            pg.axes.ravel()[i * (dim + 1)].set_zorder(1000)
            # plot point on off-diagonal, lower triangle
            for j in range(i):
                pg.axes.ravel()[i * dim + j].scatter(
                    theta_true[j],
                    theta_true[i],
                    marker="o",
                    c="black",
                    edgecolor="white",  # s=plt.rcParams['lines.markersize'] - (dim * 0.1),
                )
                # place above the kdeplots
                pg.axes.ravel()[i * dim + j].set_zorder(10000)

    if plot_bounds is not None:
        # set plot bounds
        for i in range(dim):
            pg.axes.ravel()[i * (dim + 1)].set_xlim(plot_bounds[i])
            for j in range(i):
                pg.axes.ravel()[i * dim + j].set_xlim(plot_bounds[j])
                pg.axes.ravel()[i * dim + j].set_ylim(plot_bounds[i])

    if ignore_ticks:
        # remove x and y tick labels
        for ax in pg.axes.ravel():
            if ax is not None:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    if ignore_xylabels:
        # remove xlabels and ylabels
        for ax in pg.axes.ravel():
            if ax is not None:
                ax.set_xlabel("")
                ax.set_ylabel("")

    if legend:
        # add legend
        pg.add_legend(title=title)

    return pg


# Plot functions for SBIBM
def set_axs_lims_sbibm(metric, ax, task_name):
    if metric == "mmd":
        if "lotka" in task_name:
            ax.set_ylim([0, 1.5])
        elif "sir" in task_name:
            ax.set_ylim([0, 1.0])
        elif "slcp" in task_name:
            ax.set_ylim([0, 0.6])
        elif task_name == "two_moons":
            ax.set_ylim([0, 0.3])
        elif "gaussian_mixture" in task_name:
            ax.set_ylim([0, 2])
        elif "bernoulli_glm" in task_name:
            ax.set_ylim([0, 2])
        else:
            ax.set_ylim([0, 1.5])

    elif metric == "swd":
        if task_name == "gaussian_linear":
            ax.set_ylim([0, 0.8])
        elif task_name == "gaussian_mixture":
            ax.set_ylim([0, 1])
        elif task_name == "gaussian_mixture_uniform":
            ax.set_ylim([0, 2])
        elif "bernoulli_glm" in task_name:
            ax.set_ylim([0, 1])
        elif task_name == "two_moons":
            ax.set_ylim([0, 0.4])
        elif task_name == "slcp":
            ax.set_ylim([0, 2])
        elif task_name == "sir":
            ax.set_ylim([0, 0.01])
        elif task_name == "lotka_volterra":
            ax.set_ylim([0, 0.05])
        else:
            ax.set_ylim([0, 1])

    elif metric == "mmd_to_dirac":
        if "lotka" in task_name:
            ax.set_ylim([0, 0.5])
        if "sir" in task_name:
            ax.set_ylim([0, 0.05])
        if "slcp" in task_name:
            ax.set_ylim([0, 30])
        if "gaussian_mixture" in task_name:
            ax.set_ylim([0, 6])
        if "bernoulli_glm" in task_name:
            ax.set_ylim([0, 10])
        if "two_moons" in task_name:
            ax.set_ylim([0, 2])
        if task_name == "gaussian_linear":
            ax.set_ylim([0, 6])
    elif metric == "c2st":
        ax.set_ylim([0.5, 1.1])
    else:
        ax.set_ylim([0, 1])


def plots_dist_n_train(
    metric,
    tasks_dict,
    method_names,
    prec_ignore_nums,
    n_train_list,
    n_obs_list,
    compute_dist_fn,
    title_ext="",
):
    alpha, alpha_fill = set_plotting_style()

    n_rows = len(tasks_dict.keys())
    n_cols = len(n_obs_list)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True
    )
    fig.subplots_adjust(right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1)
    for i, task_name in enumerate(tasks_dict.keys()):
        for j, n_obs in tqdm(enumerate(n_obs_list), desc=f"{task_name}, {metric}"):
            mean_dist_dict = {method: [] for method in method_names}
            std_dist_dict = {method: [] for method in method_names}
            for n_train in n_train_list:
                for method in method_names:
                    dist, _ = compute_dist_fn(
                        task_name=task_name,
                        n_train=n_train,
                        n_obs=n_obs,
                        cov_mode=method.split("_")[0],
                        langevin=True if "LANGEVIN" in method else False,
                        tamed_ula=True if "tamed" in method else False,
                        clip=True if "clip" in method else False,
                        clf_free_guidance=True if "cfg" in method else False,
                        metric=metric,
                        load=True,
                        prec_ignore_nums=prec_ignore_nums[task_name],
                    )
                    mean_dist_dict[method].append(dist["mean"])
                    std_dist_dict[method].append(dist["std"])
            print()
            print(n_obs)
            print()
            for k, mean_, std_ in zip(
                mean_dist_dict.keys(),
                mean_dist_dict.values(),
                std_dist_dict.values(),
            ):
                mean_, std_ = torch.FloatTensor(mean_), torch.FloatTensor(std_)
                print(k, mean_, std_)
                # print()
                axs[i, j].fill_between(
                    n_train_list,
                    mean_ - std_,
                    mean_ + std_,
                    alpha=alpha_fill,
                    color=METHODS_STYLE[k]["color"],
                )
                axs[i, j].plot(
                    n_train_list,
                    mean_,
                    alpha=alpha,
                    **METHODS_STYLE[k],
                )

            # set ax title as N_obs only at the top
            if i == 0:
                axs[i, j].set_title(r"$n$" + rf"$={n_obs}$")
            # label xaxis only at the bottom
            if i == len(tasks_dict.keys()) - 1:
                axs[i, j].set_xlabel(r"$N_\mathrm{train}$")
                axs[i, j].set_xscale("log")
                axs[i, j].set_xticks(n_train_list)
            # label yaxis only at the left
            if j == 0:
                axs[i, j].set_ylabel(
                    tasks_dict[task_name] + "\n" + METRICS_STYLE[metric]["label"]
                )
            else:
                axs[i, j].set_yticklabels([])

            set_axs_lims_sbibm(metric, axs[i, j], task_name)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    plt.legend(handles, labels, loc="lower right", prop={"family": "monospace"})

    return fig


def plots_dist_n_obs(
    metric,
    tasks_dict,
    method_names,
    prec_ignore_nums,
    n_train_list,
    n_obs_list,
    compute_dist_fn,
    title_ext="",
):
    alpha, alpha_fill = set_plotting_style()

    n_rows = len(tasks_dict.keys())
    n_cols = len(n_train_list)

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True
    )  # , constrained_layout=True)
    fig.subplots_adjust(right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1)
    for i, task_name in enumerate(tasks_dict.keys()):
        for j, n_train in tqdm(enumerate(n_train_list), desc=f"{task_name}, {metric}"):
            mean_dist_dict = {method: [] for method in method_names}
            std_dist_dict = {method: [] for method in method_names}
            for n_obs in n_obs_list:
                for method in method_names:
                    dist, _ = compute_dist_fn(
                        task_name=task_name,
                        n_train=n_train,
                        n_obs=n_obs,
                        cov_mode=method.split("_")[0],
                        langevin=True if "LANGEVIN" in method else False,
                        tamed_ula=True if "tamed" in method else False,
                        clip=True if "clip" in method else False,
                        clf_free_guidance=True if "cfg" in method else False,
                        metric=metric,
                        load=True,
                        prec_ignore_nums=prec_ignore_nums[task_name],
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
                    n_obs_list,
                    mean_ - std_,
                    mean_ + std_,
                    alpha=alpha_fill,
                    color=METHODS_STYLE[k]["color"],
                )
                axs[i, j].plot(
                    n_obs_list,
                    mean_,
                    alpha=alpha,
                    **METHODS_STYLE[k],
                )

            # set ax title as N_train only at the top
            if i == 0:
                axs[i, j].set_title(r"$N_\mathrm{train}$" + rf"$={n_train}$")
            # label xaxis only at the bottom
            if i == len(tasks_dict.keys()) - 1:
                axs[i, j].set_xlabel(r"$n$")
                axs[i, j].set_xticks(n_obs_list)
            # label yaxis only at the left
            if j == 0:
                axs[i, j].set_ylabel(
                    tasks_dict[task_name] + "\n" + METRICS_STYLE[metric]["label"]
                )
            else:
                axs[i, j].set_yticklabels([])

            set_axs_lims_sbibm(metric, axs[i, j], task_name)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    plt.legend(handles, labels, loc="lower right", prop={"family": "monospace"})

    return fig


def plots_dist_n_train_pf_nse(
    metric,
    tasks_dict,
    method_names,
    prec_ignore_nums,
    n_train_list,
    n_max_list,
    compute_dist_fn,
    title_ext="",
):
    alpha, alpha_fill = set_plotting_style()

    n_cols = len(tasks_dict.keys())
    fig, axs = plt.subplots(1, n_cols + 1, figsize=(5 * (n_cols + 1), 5), sharex=True)
    fig.subplots_adjust(
        right=0.995, top=0.9, bottom=0.2, hspace=0, wspace=0.3, left=0.1
    )
    n_max_list = [1] + n_max_list
    for i, task_name in enumerate(tasks_dict.keys()):
        mean_dist_dict = {
            method: {n_max: [] for n_max in n_max_list} for method in method_names
        }
        std_dist_dict = {
            method: {n_max: [] for n_max in n_max_list} for method in method_names
        }

        for n_train in tqdm(n_train_list, desc=f"{task_name}, {metric}"):
            for n_max in n_max_list:
                for method in method_names:
                    dist, _ = compute_dist_fn(
                        task_name=task_name,
                        n_train=n_train,
                        n_obs=30,
                        cov_mode=method.split("_")[0],
                        langevin=True if "LANGEVIN" in method else False,
                        tamed_ula=True if "tamed" in method else False,
                        clip=True if "clip" in method else False,
                        clf_free_guidance=True if "cfg" in method else False,
                        pf_nse=True if n_max > 1 else False,
                        n_max=n_max,
                        metric=metric,
                        load=True,
                        prec_ignore_nums=prec_ignore_nums[task_name],
                    )
                    mean_dist_dict[method][n_max].append(dist["mean"])
                    std_dist_dict[method][n_max].append(dist["std"])

        for k in mean_dist_dict.keys():
            colors = cm.get_cmap(CMAPS_METHODS[k])(
                np.linspace(1, 0.2, len(n_max_list))
            ).tolist()
            colors[0] = METHODS_STYLE[k]["color"]
            labels = [
                METHODS_STYLE[k]["label"] + r", $n_\mathrm{max}=$" + rf"${n_max}$"
                for n_max in mean_dist_dict[k].keys()
            ]
            for n_max, color, label in zip(mean_dist_dict[k].keys(), colors, labels):
                mean_, std_ = torch.FloatTensor(
                    mean_dist_dict[k][n_max]
                ), torch.FloatTensor(std_dist_dict[k][n_max])
                axs[i].fill_between(
                    n_train_list,
                    mean_ - std_,
                    mean_ + std_,
                    alpha=alpha_fill,
                    color=color,
                )
                methods_style = METHODS_STYLE[k].copy()
                methods_style["color"] = color
                methods_style["label"] = label
                axs[i].plot(
                    n_train_list,
                    mean_,
                    alpha=alpha,
                    **methods_style,
                )

        axs[i].set_title(tasks_dict[task_name])
        # label yaxis only at the left
        if i == 0:
            axs[i].set_ylabel(METRICS_STYLE[metric]["label"])
        # label xaxis
        axs[i].set_xlabel(r"$N_\mathrm{train}$")
        axs[i].set_xscale("log")
        axs[i].set_xticks(n_train_list)

        set_axs_lims_sbibm(metric, axs[i], task_name)

    handles, labels = axs[0].get_legend_handles_labels()
    # remove axs from last axs
    axs[-1].axis("off")
    plt.legend(
        handles, labels, prop={"family": "monospace"}, bbox_to_anchor=(1.05, 1.1)
    )

    return fig


# Plot functions for JR-NMM
def plot_pairgrid_with_groundtruth_jrnnm(samples, theta_gt, labels, colors):
    plt.rcParams["xtick.labelsize"] = 20.0
    plt.rcParams["ytick.labelsize"] = 20.0

    dim = len(theta_gt[0])

    dfs = []
    for n in range(len(samples)):
        df = pd.DataFrame(
            samples[n].detach().numpy(),
            columns=[r"$C$", r"$\mu$", r"$\sigma$", r"$g$"][:dim],
        )
        df["Distribution"] = labels[n]
        dfs.append(df)

    joint_df = pd.concat(dfs, ignore_index=True)

    g = sns.PairGrid(
        joint_df,
        hue="Distribution",
        palette=dict(zip(labels, colors)),
        diag_sharey=False,
        corner=True,
    )

    g.fig.set_size_inches(8, 8)

    g.map_lower(sns.kdeplot, linewidths=3, constrained_layout=False)
    g.map_diag(sns.kdeplot, fill=True, linewidths=3)

    g.axes[1][0].set_xlim(10.0, 300.0)  # C
    g.axes[1][0].set_ylim(50.0, 500.0)  # mu
    g.axes[1][0].set_yticks([200, 400])

    g.axes[2][0].set_xlim(10.0, 300.0)  # C
    g.axes[2][0].set_ylim(100.0, 5000.0)  # sigma
    g.axes[2][0].set_yticks([1000, 3500])

    g.axes[2][1].set_xlim(50.0, 500.0)  # mu
    g.axes[2][1].set_ylim(100.0, 5000.0)  # sigma
    # g.axes[2][1].set_xticks([])

    if dim == 4:
        g.axes[3][0].set_xlim(10.0, 300.0)  # C
        g.axes[3][0].set_ylim(-22.0, 22.0)  # gain
        g.axes[3][0].set_yticks([-20, 0, 20])
        g.axes[3][0].set_xticks([100, 250])

        g.axes[3][1].set_xlim(50.0, 500.0)  # mu
        g.axes[3][1].set_ylim(-22.0, 22.0)  # gain
        g.axes[3][1].set_xticks([200, 400])

        g.axes[3][2].set_xlim(100.0, 5000.0)  # sigma
        g.axes[3][2].set_ylim(-22.0, 22.0)  # gain
        g.axes[3][2].set_xticks([1000, 3500])

        g.axes[3][3].set_xlim(-22.0, 22.0)  # gain

    if theta_gt is not None:
        # get groundtruth parameters
        for gt in theta_gt:
            C, mu, sigma = gt[:3]
            gain = gt[3] if dim == 4 else None

            # plot points
            g.axes[1][0].scatter(C, mu, color="black", zorder=2, s=8)
            g.axes[2][0].scatter(C, sigma, color="black", zorder=2, s=8)
            g.axes[2][1].scatter(mu, sigma, color="black", zorder=2, s=8)
            if dim == 4:
                g.axes[3][0].scatter(C, gain, color="black", zorder=2, s=8)
                g.axes[3][1].scatter(mu, gain, color="black", zorder=2, s=8)
                g.axes[3][2].scatter(sigma, gain, color="black", zorder=2, s=8)

            # plot dirac
            g.axes[0][0].axvline(x=C, ls="--", c="black", linewidth=1, zorder=100)
            g.axes[1][1].axvline(x=mu, ls="--", c="black", linewidth=1, zorder=100)
            g.axes[2][2].axvline(x=sigma, ls="--", c="black", linewidth=1, zorder=100)
            if dim == 4:
                g.axes[3][3].axvline(
                    x=gain, ls="--", c="black", linewidth=1, zorder=100
                )

    handles, labels = g.axes[0][0].get_legend_handles_labels()
    # make handle lines larger
    for h in handles:
        h.set_linewidth(3)
    g.add_legend(handles=handles, labels=labels, title="")

    return g
