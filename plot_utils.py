import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lampe.plots import corner


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


def plot_pairgrid_with_groundtruth_jrnnm(samples, theta_gt, labels, colors, n_samples=10000):
    dfs = []
    for n in range(len(samples)):
        df = pd.DataFrame(
            samples[n].detach().numpy(), columns=[r"$C$", r"$\mu$", r"$\sigma$", r"$g$"]
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

    g.map_lower(sns.kdeplot, linewidths=1)
    g.map_diag(sns.kdeplot, fill=True, linewidths=1)

    g.axes[1][0].set_xlim(10.0, 300.0)  # C
    g.axes[1][0].set_ylim(50.0, 500.0)  # mu
    g.axes[1][0].set_yticks([200, 400])

    g.axes[2][0].set_xlim(10.0, 300.0)  # C
    g.axes[2][0].set_ylim(100.0, 5000.0)  # sigma
    g.axes[2][0].set_yticks([1000, 3500])

    g.axes[2][1].set_xlim(50.0, 500.0)  # mu
    g.axes[2][1].set_ylim(100.0, 5000.0)  # sigma
    # g.axes[2][1].set_xticks([])

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
            C, mu, sigma, gain = gt

            # plot points
            g.axes[1][0].scatter(C, mu, color="black", zorder=2, s=8)
            g.axes[2][0].scatter(C, sigma, color="black", zorder=2, s=8)
            g.axes[2][1].scatter(mu, sigma, color="black", zorder=2, s=8)
            g.axes[3][0].scatter(C, gain, color="black", zorder=2, s=8)
            g.axes[3][1].scatter(mu, gain, color="black", zorder=2, s=8)
            g.axes[3][2].scatter(sigma, gain, color="black", zorder=2, s=8)

            # plot dirac
            g.axes[0][0].axvline(x=C, ls="--", c="black", linewidth=1)
            g.axes[1][1].axvline(x=mu, ls="--", c="black", linewidth=1)
            g.axes[2][2].axvline(x=sigma, ls="--", c="black", linewidth=1)
            g.axes[3][3].axvline(x=gain, ls="--", c="black", linewidth=1)

    g.add_legend()

    return g
