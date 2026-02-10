# Script to load and plot results of the Gaussian toy example experiment (Section 4.1 of the paper).

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from plot_utils import METHODS_STYLE, METRICS_STYLE, set_plotting_style

if __name__ == "__main__":
    import sys

    destination_folder = sys.argv[1]
    os.makedirs(f"{destination_folder}/figures_geffner_all", exist_ok=True)
    os.makedirs(f"{destination_folder}/data_geffner_all", exist_ok=True)
    alpha, alpha_fill = set_plotting_style()

    # Read all the data!
    all_data = pd.read_csv(
        f"{destination_folder}/gaussian_mixture_exp_geffner_all_treated.csv"
    ).reset_index()

    # Make Time table
    dim = 10
    n_obs = 32
    eps = 1e-2
    time_info = (
        all_data.groupby(["dim", "N_OBS", "sampling_steps", "alg", "eps"])[["dt", "sw"]]
        .agg(lambda x: f"{x.mean():.2f} +/- {x.std()*1.96 / x.shape[0]**.5:.2f}")
        .reset_index()
    )
    table_to_save = time_info.loc[
        (time_info.dim == dim) & (time_info.N_OBS == n_obs) & (time_info.eps == eps),
        ["alg", "sampling_steps", "dt", "sw"],
    ]
    print(table_to_save)
    table_to_save.reset_index().to_csv(
        f"{destination_folder}/data_geffner_all/table_time_sw_comparison.csv", index=False
    )

    # Load data of "equivalent speed"
    equivalent_speed_data = all_data.loc[
        (
            ((all_data.alg == "GAUSS") & (all_data.sampling_steps == 1000))
            | ((all_data.alg == "JAC") & (all_data.sampling_steps == 400))
            | ((all_data.alg == "Langevin") & (all_data.sampling_steps == 400))
            | ((all_data.alg == "DET_GEF") & (all_data.sampling_steps == 1000))
        )
    ]

    # Selecting only dim 10 and eps < 1
    DIM = 10
    equivalent_speed_data = equivalent_speed_data.loc[
        (equivalent_speed_data.dim == DIM) & (all_data.eps < 1)
    ]

    # plotting per epsilon
    n_plots = len(equivalent_speed_data["eps"].unique())
    fig, axes_all = plt.subplots(
        len(equivalent_speed_data["dim"].unique()),
        n_plots,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_plots, 5),
    )  # , constrained_layout=True)
    fig.subplots_adjust(right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1)
    for ax, eps in zip(axes_all[0], equivalent_speed_data["eps"].unique()):
        ax.set_title(rf"$ϵ = {eps}$")
    for ax in axes_all[-1]:
        ax.set_xlabel(r"$n$")
    axes_all[0, 0].set_ylabel(rf"$m = {DIM}$" + f'\n {METRICS_STYLE["swd"]["label"]}')
    # Group data by dim (row)
    for (eps, eps_data), ax in zip(
        equivalent_speed_data.groupby(["eps"]), axes_all.flatten()
    ):
        ax.set_ylim([-1e-1, 1e0])

        # ax.set_yticks([])

        # Group by eps and iter over columns of axis
        # Group by alg to plot
        for (alg, sampling_steps), dt in eps_data.groupby(["alg", "sampling_steps"]):
            # Plotting
            if alg == "Langevin":
                alg = "LANGEVIN"
            label = f"{alg} ({sampling_steps})"
            plot_kwars = METHODS_STYLE[alg]
            plot_kwars["label"] = label
            ax.plot(
                dt.groupby(["N_OBS"]).first().index,
                dt.groupby(["N_OBS"]).sw_norm.mean(),
                **plot_kwars,
            )
            ax.fill_between(
                dt.groupby(["N_OBS"]).first().index,
                dt.groupby(["N_OBS"]).sw_norm.mean()
                - 1.96
                * dt.groupby(["N_OBS"]).sw_norm.std()
                / (dt.groupby(["N_OBS"])["seed"].count() ** 0.5),
                dt.groupby(["N_OBS"]).sw_norm.mean()
                + 1.96
                * dt.groupby(["N_OBS"]).sw_norm.std()
                / (dt.groupby(["N_OBS"])["seed"].count() ** 0.5),
                alpha=alpha_fill,
                color=plot_kwars["color"],
            )
            # ax.errorbar(x=dt.groupby(['N_OBS']).first().index,
            #             y=dt.groupby(['N_OBS']).sw_norm.mean(),
            #             yerr=1.96*dt.groupby(['N_OBS']).sw.std() / (dt.groupby(['N_OBS'])['seed'].count()**.5),
            #             color=c, label=label, alpha=.8, marker='o', linestyle=lst,
            #             capsize=10, elinewidth=2)
        # ax.set_yscale('log')
        ax.set_xscale("log")
        # ax.set_ylabel('Sliced Wasserstein')
        # ax.set_xlabel('Number of Observations')
        ax.set_xlim(1.5, 110)
    axes_all[-1, -1].legend(prop={"family": "monospace"})
    # axes[0].set_xlim(1.5, 100.5)
    fig.savefig(f"{destination_folder}/figures_geffner_all/n_obs_vs_sw_per_eps_dim.pdf")
    fig.savefig(f"{destination_folder}/figures_geffner_all/n_obs_vs_sw_per_eps_dim.png")
    fig.show()
    plt.close(fig)

    # Same thing but here cols are algs!
    n_cols = len(equivalent_speed_data["alg"].unique())
    n_rows = 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_cols, 5 * n_rows),
    )  # , constrained_layout=True)
    fig.subplots_adjust(
        right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.15
    )
    for ax, d in zip(axes[:, 0], equivalent_speed_data["dim"].unique()):
        ax.set_ylabel(rf"$m = {d}$" + "\n" + METRICS_STYLE["swd"]["label"])
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$n$")
    # for ax in axes[:, 1:].flatten():
    # ax.set_yticks([])

    for ax, ((dim, alg), alg_data) in zip(
        axes.flatten(), equivalent_speed_data.groupby(["dim", "alg"])
    ):
        if alg == "Langevin":
            alg = "LANGEVIN"
        #     n_obs = [i[1] for i in ref_sw_per_dim.keys() if i[0] == dim[0]]

        ax.set_title(f"{alg}")

        print(alg)
        # ax.fill_between(n_obs, -np.array(yerr_ref), yerr_ref, color='red', alpha=.5)
        for i, (eps, dt) in enumerate(alg_data.groupby(["eps"])):
            label = rf"$\epsilon = {eps[0]}$"
            c = cm.get_cmap("coolwarm")(-math.log10(eps[0] + 1e-5) / 5)
            # c = cm.get_cmap('coolwarm')(i / len(alg_data['eps'].unique()))
            plot_kwars = METHODS_STYLE[alg]
            plot_kwars["label"] = label
            plot_kwars["color"] = c
            ax.plot(
                dt.groupby(["N_OBS"]).first().index,
                dt.groupby(["N_OBS"]).sw_norm.mean(),
                **plot_kwars,
            )
            ax.fill_between(
                dt.groupby(["N_OBS"]).first().index,
                dt.groupby(["N_OBS"]).sw_norm.mean()
                - 1.96
                * dt.groupby(["N_OBS"]).sw_norm.std()
                / (dt.groupby(["N_OBS"])["seed"].count() ** 0.5),
                dt.groupby(["N_OBS"]).sw_norm.mean()
                + 1.96
                * dt.groupby(["N_OBS"]).sw_norm.std()
                / (dt.groupby(["N_OBS"])["seed"].count() ** 0.5),
                alpha=alpha_fill,
                color=c,
            )
            # ax.errorbar(x=dt.groupby(['N_OBS']).first().index,
            #             y=dt.groupby(['N_OBS']).sw_norm.mean(),
            #             yerr=1.96*dt.groupby(['N_OBS']).sw.std() / (dt.groupby(['N_OBS'])['seed'].count()**.5),
            #             capsize=10, elinewidth=2, **METHODS_STYLE[alg])
        ax.set_ylim([-1e-1, 1e0])
        ax.set_xscale("log")
        ax.set_xlim(1.5, 110)
    axes[-1, -1].legend()
    fig.savefig(f"{destination_folder}/figures_geffner_all/n_obs_vs_sw_per_alg_dim.pdf")
    fig.savefig(f"{destination_folder}/figures_geffner_all/n_obs_vs_sw_per_alg_dim.png")
    fig.show()
    plt.close(fig)
