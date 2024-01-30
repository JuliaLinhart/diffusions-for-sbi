import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lampe.plots import corner

from tueplots import fonts, axes

def set_plotting_style():
    plt.rcParams.update(fonts.neurips2022())
    plt.rcParams.update(axes.color(base="black"))
    plt.rcParams["legend.fontsize"] = 23.0
    plt.rcParams["xtick.labelsize"] = 23.0
    plt.rcParams["ytick.labelsize"] = 23.0
    plt.rcParams["axes.labelsize"] = 23.0
    plt.rcParams["font.size"] = 23.0
    plt.rcParams["axes.titlesize"] = 27.0
    alpha = 0.9
    alpha_fill = 0.1
    return alpha, alpha_fill

markersize = plt.rcParams['lines.markersize'] * 1.5
METHODS_STYLE = {
    "GAUSS": {"label":"GAUSS", "color": "blue", "marker": "o", "linestyle": "-", "linewidth":3, "markersize": markersize},
    "GAUSS_clip": {"label":"GAUSS (clip)", "color": "blue", "marker": "o", "linestyle": "--", "linewidth":4, "markersize": markersize},
    "JAC": {"label":"JAC", "color": "orange", "marker": "o", "linestyle": "-", "linewidth":3, "markersize": markersize},
    "JAC_clip": {"label":"JAC (clip)", "color": "orange", "marker": "o", "linestyle": "--", "linewidth":4, "markersize": markersize},
    "LANGEVIN": {"label":"LANGEVIN", "color": "#92374D", "marker": "o", "linestyle": "-", "linewidth":3, "markersize": markersize}, 
    "LANGEVIN_clip": {"label":"LANGEVIN (clip)", "color": "#92374D", "marker": "o", "linestyle": "--", "linewidth":4, "markersize": markersize},
}

METRICS_STYLE = {
    "swd": {"label": "Sliced W2"},
    "mmd": {"label": "MMD"},
    "mmd_to_dirac": {"label": "MMD to Dirac"},
}

def multi_corner_plots(samples_list, legends, colors, title, **kwargs):
    fig = None
    for s, l, c in zip(samples_list, legends, colors):
        fig = corner(s, legend=l, color=c, figure=fig, smooth=2, **kwargs)
        plt.suptitle(title)

# Plot learned posterior P(theta | x_obs)
def pairplot_with_groundtruth_2d(
    samples_list,
    labels,
    colors,
    theta_true=None,
    prior_bounds=None,
    param_names=None,
    plot_bounds=None,
):
    columns = [r"$\theta_1$", rf"$\theta_2$"]
    if param_names is not None:
        columns = param_names

    dfs = []
    for samples, label in zip(samples_list, labels):
        df = pd.DataFrame(samples, columns=columns)
        df["Distribution"] = label
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    pg = sns.pairplot(
        dfs,
        hue="Distribution",
        corner=True,
        palette=dict(zip(labels, colors)),
    )

    if theta_true is not None:
        pg.axes.ravel()[0].axvline(x=theta_true[0], ls="--", linewidth=2, c="black")
        pg.axes.ravel()[3].axvline(x=theta_true[1], ls="--", linewidth=2, c="black")
        pg.axes.ravel()[2].scatter(
            theta_true[0], theta_true[1], marker="o", c="black", s=50, edgecolor="white"
        )

    if prior_bounds is not None:
        pg.axes.ravel()[0].axvline(x=prior_bounds[0][0], ls="--", linewidth=1, c="red")
        pg.axes.ravel()[0].axvline(x=prior_bounds[0][1], ls="--", linewidth=1, c="red")
        pg.axes.ravel()[3].axvline(x=prior_bounds[1][0], ls="--", linewidth=1, c="red")
        pg.axes.ravel()[3].axvline(x=prior_bounds[1][1], ls="--", linewidth=1, c="red")

    if plot_bounds is not None:
        pg.axes.ravel()[2].set_xlim(plot_bounds[0])
        pg.axes.ravel()[3].set_xlim(plot_bounds[1])
        pg.axes.ravel()[2].set_ylim(plot_bounds[1])

    return pg

# Plot learned posterior P(theta | x_obs)
def pairplot_with_groundtruth_md(
    samples_list,
    labels,
    colors,
    theta_true=None,
    param_names=None,
    plot_bounds=None,
):
    columns = [r"$\theta_1$", rf"$\theta_2$"]
    if param_names is not None:
        columns = param_names

    dfs = []
    for samples, label in zip(samples_list, labels):
        df = pd.DataFrame(samples, columns=columns)
        df["Distribution"] = label
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    pg = sns.pairplot(
        dfs,
        hue="Distribution",
        corner=True,
        palette=dict(zip(labels, colors)),
    )

    if theta_true is not None:
        if theta_true.ndim > 1:
            theta_true = theta_true[0]
            dim = len(theta_true)
        for i in range(dim):
            # plot dirac on diagonal
            pg.axes.ravel()[i*(dim+1)].axvline(x=theta_true[i], ls="--", linewidth=2, c="black")
            # plot point on off-diagonal, lower triangle
            for j in range(i):
                pg.axes.ravel()[i*dim+j].scatter(
                    theta_true[j], theta_true[i], marker="o", c="black", s=50, edgecolor="white"
                )        

    if plot_bounds is not None:
        # set plot bounds
        for i in range(dim):
            pg.axes.ravel()[i*(dim+1)].set_xlim(plot_bounds[i])
            for j in range(i):
                pg.axes.ravel()[i*dim+j].set_xlim(plot_bounds[j])
                pg.axes.ravel()[i*dim+j].set_ylim(plot_bounds[i])

    return pg


def plot_pairgrid_with_groundtruth_jrnnm(samples, theta_gt, labels, colors):
    dim = len(theta_gt[0])

    dfs = []
    for n in range(len(samples)):
        df = pd.DataFrame(
            samples[n].detach().numpy(), columns=[r"$C$", r"$\mu$", r"$\sigma$", r"$g$"][:dim]
        )
        df["Distribution"] = labels[n]
        dfs.append(df)

    joint_df = pd.concat(dfs, ignore_index=True)

    g = sns.PairGrid(
        joint_df,
        hue="Distribution",
        palette=dict(zip(labels, colors)),
        diag_sharey=False,
        corner=True
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
            g.axes[0][0].axvline(x=C, ls="--", c="black", linewidth=1)
            g.axes[1][1].axvline(x=mu, ls="--", c="black", linewidth=1)
            g.axes[2][2].axvline(x=sigma, ls="--", c="black", linewidth=1)
            if dim == 4:
                g.axes[3][3].axvline(x=gain, ls="--", c="black", linewidth=1)

    handles, labels = g.axes[0][0].get_legend_handles_labels()
    # make handle lines larger
    for h in handles:
        h.set_linewidth(3)
    g.add_legend(handles=handles, labels=labels, title="")

    return g



