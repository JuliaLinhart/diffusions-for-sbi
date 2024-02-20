# Script to load and plot results of the Gaussian Mixture toy example experiment (Section 4.1 of the paper).

import matplotlib.pyplot as plt
import math
import os
import pandas as pd

from matplotlib import cm
from plot_utils import METHODS_STYLE, METRICS_STYLE, set_plotting_style

PATH_EXPERIMENT = "results/toy_models/gaussian"
os.makedirs(f"{PATH_EXPERIMENT}/figures", exist_ok=True)
os.makedirs(f"{PATH_EXPERIMENT}/data", exist_ok=True)

if __name__ == "__main__":
    alpha, alpha_fill = set_plotting_style()

    # Read all the data!
    all_data = pd.read_csv(f"{PATH_EXPERIMENT}/gaussian_exp_treated.csv").reset_index()

    # Make Tite table
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
    time_data = all_data.pivot(
        index=["dim", "N_OBS", "eps", "sampling_steps", "seed"],
        columns="alg",
        values="dt",
    )
    time_data = time_data.assign(
        speed_up_gauss=time_data.GAUSS / time_data.Langevin,
        speed_up_jac=time_data.JAC / time_data.Langevin,
    )
    agg_time_data = (
        time_data.groupby(["dim"])[["speed_up_gauss", "speed_up_jac"]]
        .agg(lambda x: f"{x.mean():.2f} ± {x.std()*1.96 / x.shape[0]**.5:.2f}")
        .reset_index()
    )
    agg_time_data = agg_time_data.loc[agg_time_data["dim"] < 64]
    agg_time_data.reset_index().to_csv(
        f"{PATH_EXPERIMENT}/data/speed_up_comparison.csv", index=False
    )

    # Load data of "equivalent speed"
    equivalent_speed_data = all_data.loc[
        (
            ((all_data.alg == "GAUSS") & (all_data.sampling_steps == 1000))
            | ((all_data.alg == "JAC") & (all_data.sampling_steps == 400))
            | ((all_data.alg == "Langevin") & (all_data.sampling_steps == 400))
        )
    ]
    # Remove eps high since it does not work!
    equivalent_speed_data = equivalent_speed_data.loc[
        (equivalent_speed_data.eps <= 1e-1) & (equivalent_speed_data.dim < 64)
    ]

    # Should we remove some dims as well? Maybe only plot one?
    # consider a selection of dims
    # dim_dim_idxs = [2, 3] # only consider dims 8, 10
    # condisder all dims
    dim_idxs = range(len(equivalent_speed_data["dim"].unique()))

    # Plot Normalized Wasserstein cols = eps, rows = dims
    n_plots = len(equivalent_speed_data["eps"].unique())
    n_rows = len(dim_idxs)

    fig, axes_all = plt.subplots(
        n_rows,
        n_plots,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_plots, 5 * n_rows),
    )  # , constrained_layout=True)
    # fig, axes_all = plt.subplots(n_rows, n_plots, sharex=True, sharey=True, squeeze=False, figsize=(1 + 4 * n_plots, 1 + 1.5*n_rows))
    fig.subplots_adjust(right=0.995, top=0.95, bottom=0.2, hspace=0, wspace=0, left=0.1)

    for ax, eps in zip(axes_all[0], equivalent_speed_data["eps"].unique()):
        ax.set_title(rf"$ϵ = {eps}$")
    for ax in axes_all[-1]:
        ax.set_xlabel(r"$n$")

    print(equivalent_speed_data.groupby(["dim"]))
    # remove some dims
    equivalent_speed_data_small = equivalent_speed_data.loc[
        equivalent_speed_data.dim.isin(equivalent_speed_data.dim.unique()[dim_idxs])
    ]

    # Group data per dim (row)
    for axes, (dim, dim_data) in zip(
        axes_all, equivalent_speed_data_small.groupby(["dim"])
    ):
        n_plots = len(dim_data["eps"].unique())
        axes[0].set_ylabel(rf"$m = {dim[0]}$" + "\n " + METRICS_STYLE["swd"]["label"])
        for ax in axes.flatten():
            ax.set_ylim([-1e-1, 1e0])
        # for ax in axes[1:]:
        #     ax.set_yticks([])
        # Group data per eps
        for (eps, eps_data), ax in zip(dim_data.groupby(["eps"]), axes):
            # Group data per algo
            for (alg, sampling_steps), dt in eps_data.groupby(
                ["alg", "sampling_steps"]
            ):
                # PLOT!
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
                #             capsize=10, elinewidth=2, **METHODS_STYLE[alg])
            # ax.set_yscale('log')
            ax.set_xscale("log")
            # ax.set_ylabel('Sliced Wasserstein')
            # ax.set_xlabel('Number of Observations')
            ax.set_xlim(1.5, 110)
    axes_all[-1, -1].legend()
    # axes[0].set_xlim(1.5, 100.5)
    fig.savefig(f"{PATH_EXPERIMENT}/figures/n_obs_vs_sw_per_eps_dim.pdf")
    fig.savefig(f"{PATH_EXPERIMENT}/figures/n_obs_vs_sw_per_eps_dim.png")
    fig.show()
    plt.close(fig)

    # Same thing but here cols are algs!
    n_cols = len(equivalent_speed_data["alg"].unique())
    n_rows = len(dim_idxs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_cols, 5 * n_rows),
    )  # , constrained_layout=True)
    fig.subplots_adjust(
        right=0.995, top=0.95, bottom=0.2, hspace=0, wspace=0, left=0.15
    )
    for ax, d in zip(axes[:, 0], equivalent_speed_data_small["dim"].unique()):
        ax.set_ylabel(rf"$m = {d}$" + "\n" + METRICS_STYLE["swd"]["label"])
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$n$")
    # for ax in axes[:, 1:].flatten():
    # ax.set_yticks([])

    for ax, ((dim, alg), alg_data) in zip(
        axes.flatten(), equivalent_speed_data_small.groupby(["dim", "alg"])
    ):
        if alg == "Langevin":
            alg = "LANGEVIN"
        #     n_obs = [i[1] for i in ref_sw_per_dim.keys() if i[0] == dim[0]]
        if dim == 2:
            ax.set_title(f"{alg}")

        print(alg)
        # ax.fill_between(n_obs, -np.array(yerr_ref), yerr_ref, color='red', alpha=.5)
        for i, (eps, dt) in enumerate(alg_data.groupby(["eps"])):
            label = eps[0]
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
    fig.savefig(f"{PATH_EXPERIMENT}/figures/n_obs_vs_sw_per_alg_dim.pdf")
    fig.savefig(f"{PATH_EXPERIMENT}/figures/n_obs_vs_sw_per_alg_dim.png")
    fig.show()
    plt.close(fig)

    # Same thing but inverted rows and cols!
    n_rows = len(equivalent_speed_data["alg"].unique())
    n_cols = len(dim_idxs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * n_cols, 5 * n_rows),
    )  # , constrained_layout=True)
    fig.subplots_adjust(
        right=0.995, top=0.95, bottom=0.2, hspace=0, wspace=0, left=0.15
    )
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$n$")
    # for ax in axes[:, 1:].flatten():
    # ax.set_yticks([])

    # figure with dimension as col and alg as row
    for ax, ((alg, dim), alg_data) in zip(
        axes.flatten(), equivalent_speed_data_small.groupby(["alg", "dim"])
    ):
        if alg == "Langevin":
            alg = "LANGEVIN"
        # ax.fill_between(n_obs, -np.array(yerr_ref), yerr_ref, color='red', alpha=.5)
        for i, (eps, dt) in enumerate(alg_data.groupby(["eps"])):
            label = eps[0]
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

        # set ylabel for first col
        if dim == 2:
            ax.set_ylabel(f"{alg}" + "\n" + METRICS_STYLE["swd"]["label"])
        # set title for first row
        if alg == "GAUSS":
            ax.set_title(rf"$m = {dim}$")
    axes[-1, -1].legend()
    fig.savefig(f"{PATH_EXPERIMENT}/figures/n_obs_vs_sw_per_alg_dim_inverted.pdf")
    fig.savefig(f"{PATH_EXPERIMENT}/figures/n_obs_vs_sw_per_alg_dim_inverted.png")
    # fig.show()
    plt.close(fig)
