import torch
import matplotlib.pyplot as plt

# from ot import sliced_wasserstein_distance
from experiment_utils import dist_to_dirac
from plot_utils import (
    METHODS_STYLE,
    METRICS_STYLE,
    set_plotting_style,
    plot_pairgrid_with_groundtruth_jrnnm,
)

PATH_EXPERIMENT = "results/jrnnm/"
DIMS = [3, 4]
N_OBS = [1, 8, 14, 22, 30]
N_EPOCHS = 5000
LR = 1e-4

method_names = ["GAUSS", "GAUSS_clip", "LANGEVIN", "LANGEVIN_clip", "JAC", "JAC_clip"]
gain = 0.0

def load_losses(dim, lr=LR, n_epochs=N_EPOCHS):
    filename = (
        PATH_EXPERIMENT
        + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/train_losses.pkl"
    )
    losses = torch.load(filename)
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    best_epoch = losses["best_epoch"]
    return train_losses, val_losses, best_epoch


def load_results(
    dim,
    result_name,
    n_obs,
    lr=LR,
    n_epochs=N_EPOCHS,
    gain=0.0,
    cov_mode=None,
    sampler="ddim",
    langevin=False,
    clip=False,
    single_obs=None,
):
    theta_true = [135.0, 220.0, 2000.0, gain]
    if dim == 3:
        theta_true = theta_true[:3]
    path = PATH_EXPERIMENT + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/"
    if langevin:
        path = path + "langevin_steps_400_5/"
    else:
        path = (
            path + f"{sampler}_steps_1000/"
            if sampler == "euler" or cov_mode == "GAUSS"
            else path + f"{sampler}_steps_400/"
        )
    if single_obs is not None:
        path = (
            path + f"single_obs/num_{single_obs}_" + result_name + f"_{theta_true}_n_obs_1.pkl"
        )
    else:
        path = path + result_name + f"_{theta_true}_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        path = path[:-4] + "_clip.pkl"
    results = torch.load(path)
    return results

def load_lc2st_results(
    dim,
    method,
    n_obs,
):
    save_path = (
        PATH_EXPERIMENT
        + f"{dim}d/n_train_50000_n_epochs_{N_EPOCHS}_lr_{LR}/"
    )

    theta_true = [135.0, 220.0, 2000.0, 0.0][:dim]

    if "GAUSS" in method:
        sampler_path = "ddim_steps_1000/"
        ext = f"_{method}"
    elif "JAC" in method:
        sampler_path = "ddim_steps_400/"
        ext = f"_{method}"
    else:
        sampler_path = "langevin_steps_400_5/"
        ext = "_clip" if "clip" in method else ""
    lc2st_path = save_path + sampler_path + f"lc2st_results/"
    
    return torch.load(lc2st_path + f"results_ensemble_5_{theta_true}_lc2st_n_cal_10000_n_obs_{n_obs}{ext}.pkl")



# compute mean distance to true theta over all observations
def compute_distance_to_true_theta(
    dim, gain=0.0, cov_mode=None, langevin=False, clip=False
):
    true_parameters = torch.tensor([135.0, 220.0, 2000.0, gain])
    if dim == 3:
        true_parameters = true_parameters[:3]
    dist_dict = dict(zip(N_OBS, [[]] * len(N_OBS)))
    for n_obs in N_OBS:
        samples = load_results(
            dim,
            result_name="posterior_samples",
            n_obs=n_obs,
            gain=gain,
            cov_mode=cov_mode,
            langevin=langevin,
            clip=clip,
        )
        dist_dict[n_obs] = dist_to_dirac(
            samples, true_parameters, metrics=["mmd"], scaled=True
        )
    return dist_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--dirac_dist", action="store_true")
    parser.add_argument("--lc2st", action="store_true")
    parser.add_argument("--pairplot", action="store_true")
    parser.add_argument("--single_multi_obs", action="store_true")
    parser.add_argument("--dim", type=int, default=3)

    args = parser.parse_args()

    alpha, alpha_fill = set_plotting_style()

    if args.losses:
        # plot losses function to select lr
        lr_list = [1e-4, 1e-3]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.subplots_adjust(
            right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.2
        )
        for i, dim in enumerate([3, 4]):
            best_val_loss = {}
            for lr_, c in zip(lr_list, ["blue", "orange"]):
                train_losses, val_losses, best_epoch = load_losses(
                    dim, lr_, n_epochs=N_EPOCHS
                )
                best_val_loss[lr_] = val_losses[best_epoch]
                axs[i].plot(
                    train_losses,
                    label=f"train (lr={lr_})",
                    color=c,
                    linewidth=3,
                    alpha=0.3,
                )
                axs[i].plot(val_losses, label=f"val", color=c, linewidth=3, alpha=0.9)
                axs[i].axvline(
                    best_epoch,
                    color=c,
                    linestyle="--",
                    linewidth=5,
                    alpha=0.9,
                    zorder=10000,
                )
            # print(best_val_loss)
            # get lr for min best val loss
            best_lr = min(best_val_loss, key=best_val_loss.get)
            print(f"best lr for {dim}D: {best_lr}")
            axs[i].set_title(rf"${dim}$D")  # + f" \n best_lr={best_lr}")
            axs[i].set_ylim([0, 0.5])
            axs[i].set_xlim([0, 5000])
            axs[i].set_xlabel("epochs")
        # axs[i].set_ylabel("loss")
        axs[i].legend()
        plt.savefig(PATH_EXPERIMENT + "jrnnm_losses.png")
        plt.savefig(PATH_EXPERIMENT + "jrnnm_losses.pdf")
        plt.clf()

    if args.dirac_dist:

        # plot mean distance to true theta as function of n_obs
        for i, dim in enumerate(DIMS):
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            fig.subplots_adjust(
                right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.25
            )
            for method in method_names:
                if method == "JAC":
                    continue
                else:
                    dist_dict = compute_distance_to_true_theta(
                        dim,
                        cov_mode=method.split("_")[0],
                        langevin=True if "LANGEVIN" in method else False,
                        clip=True if "clip" in method else False,
                    )
                    axs.plot(
                        N_OBS,
                        [dist_dict[n_obs]["mmd"] for n_obs in N_OBS],
                        alpha=alpha,
                        **METHODS_STYLE[method],
                    )
            axs.set_xticks(N_OBS)
            axs.set_xlabel(r"$n$")
            axs.set_ylabel(f"{METRICS_STYLE[metric]['label']}")
            if dim == 4:
                axs.set_ylim([0, 600])
            # axs.legend()
            # axs.set_title(rf"${dim}$D")

            # plt.suptitle(rf"MMD to $\theta^\star = (135, 220, 2000, {gain})$")
            plt.savefig(PATH_EXPERIMENT + f"mmd_to_dirac_n_obs_g_{gain}_dim_{dim}.png")
            plt.savefig(PATH_EXPERIMENT + f"mmd_to_dirac_n_obs_g_{gain}_dim_{dim}.pdf")
            plt.clf()
    
    if args.lc2st:
        dim = args.dim

        # load results
        results_dict = {method: {} for method in method_names}
        for method in method_names:
            if method == "JAC":
                continue
            for n_obs in [1,8,14,22,30]:
                results_dict[method][n_obs] = load_lc2st_results(dim, method, n_obs)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
        for method in method_names:
            if method == "JAC":
                continue

            t_stats_mean = [results_dict[method][n_obs]["scores_data"].mean() for n_obs in [1,8,14,22,30]]
            t_stats_std = [results_dict[method][n_obs]["scores_data"].std() for n_obs in [1,8,14,22,30]]
            p_values = [results_dict[method][n_obs]["p_value"] for n_obs in [1,8,14,22,30]]
            axs[0].plot([1,8,14,22,30], t_stats_mean, **METHODS_STYLE[method])
            axs[0].fill_between([1,8,14,22,30], [t_stats_mean[i] - t_stats_std[i] for i in range(5)], [t_stats_mean[i] + t_stats_std[i] for i in range(5)], alpha=0.2, color=METHODS_STYLE[method]["color"])
            axs[1].plot([1,8,14,22,30], p_values, **METHODS_STYLE[method])
        axs[1].axhline(0.05, color="black", label="significance level", linewidth=3, linestyle="--")

        axs[0].set_xticks([1,8,14,22,30])
        axs[1].set_xticks([1,8,14,22,30])
        axs[0].set_xlabel(r"$n$")
        axs[1].set_xlabel(r"$n$")
        axs[0].set_ylim(-0.01, 0.25)
        axs[1].set_ylim(-0.01, 1)
        axs[0].set_title(r"$\ell$-C2ST statistics")
        axs[1].set_title("p-values")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(PATH_EXPERIMENT + f"lc2st_results_dim_{dim}.png")
        plt.savefig(PATH_EXPERIMENT + f"lc2st_results_dim_{dim}.pdf")


    if args.pairplot:
        from matplotlib import colormaps as cm

        method = "GAUSS"
        for dim in DIMS:
            theta_true = [135.0, 220.0, 2000.0, gain]
            if dim == 3:
                theta_true = theta_true[:3]
            samples = []
            labels = []
            for n_obs in N_OBS:
                samples_ = load_results(
                    dim,
                    result_name="posterior_samples",
                    n_obs=n_obs,
                    gain=gain,
                    cov_mode=method.split("_")[0],
                    langevin=True if "LANGEVIN" in method else False,
                    clip=True if "clip" in method else False,
                )
                samples.append(samples_)
                labels.append(rf"$n={n_obs}$")
            colors = [
                cm.get_cmap("viridis")(i) for i in torch.linspace(0.2, 1, len(N_OBS))
            ]
            plot_pairgrid_with_groundtruth_jrnnm(samples, [theta_true], labels, colors)
            plt.savefig(PATH_EXPERIMENT + f"pairplot_{method}_g_{gain}_{dim}d.png")
            plt.savefig(PATH_EXPERIMENT + f"pairplot_{method}_g_{gain}_{dim}d.pdf")
            plt.clf()

    if args.single_multi_obs:
        # plot independent posterior samples for single and multi-observation

        from matplotlib import colormaps as cm
        import seaborn as sns

        colors = [cm.get_cmap("viridis")(i) for i in torch.linspace(0.2, 1, len(N_OBS))]
        method = "GAUSS"
        dim = 3

        theta_true = [135.0, 220.0, 2000.0, gain]
        param_names = ["$C$", "$\mu$", "$\sigma$", "$g$"]
        if dim == 3:
            theta_true = theta_true[:3]
            param_names = param_names[:3]

        n_obs = 30
        fig, axs = plt.subplots(1, dim, figsize=(5 * dim, 5))
        fig.subplots_adjust(
            right=0.995, top=0.92, bottom=0.2, hspace=0, wspace=0, left=0.1
        )
        for i in range(n_obs):
            samples_single = load_results(
                dim,
                result_name="posterior_samples",
                n_obs=n_obs,
                gain=gain,
                cov_mode=method.split("_")[0],
                langevin=True if "LANGEVIN" in method else False,
                clip=True if "clip" in method else False,
                single_obs=i,
            )
            for k in range(dim):
                # plot density of marginals
                if k == 2 and i == 0:
                    sns.kdeplot(
                        samples_single[:, k],
                        alpha=0.1,
                        color=colors[0],
                        ax=axs[k],
                        linewidth=3,
                        fill=True,
                        label=rf"$n=1$",
                    )
                else:
                    sns.kdeplot(
                        samples_single[:, k],
                        alpha=0.1,
                        color=colors[0],
                        ax=axs[k],
                        linewidth=3,
                        fill=True,
                    )
            axs[k].legend()
        for j, n_obs in enumerate(N_OBS[1:]):
            samples = load_results(
                dim,
                result_name="posterior_samples",
                n_obs=n_obs,
                gain=gain,
                cov_mode=method.split("_")[0],
                langevin=True if "LANGEVIN" in method else False,
                clip=True if "clip" in method else False,
            )

            for k, name in enumerate(param_names):
                sns.kdeplot(
                    samples[:, k],
                    alpha=0.1,
                    label=rf"$n={n_obs}$",
                    color=colors[j + 1],
                    ax=axs[k],
                    linewidth=3,
                    fill=True,
                )
                # line for theta_true
                axs.ravel()[k].axvline(
                    theta_true[k], ls="--", linewidth=3, c="black"
                )
                axs[k].set_xlabel(name)
                axs[k].set_ylabel("")
                # emply list for ytick labels
                axs[k].set_yticklabels([])
            axs[k].legend()
        plt.savefig(
            PATH_EXPERIMENT + f"single_multi_obs_{method}_g_{gain}_{dim}d.png"
        )
        plt.savefig(
            PATH_EXPERIMENT + f"single_multi_obs_{method}_g_{gain}_{dim}d.pdf"
        )
        plt.clf()


    # dim = 4
    # gain_list = [-10., 0., 10.]
    # theta_true_list = [[135.0, 220.0, 2000.0, gain] for gain in gain_list]
    # samples = []
    # labels = []
    # for gain in gain_list:
    #     samples_ = load_results(
    #         dim,
    #         result_name="posterior_samples",
    #         n_obs=1,
    #         gain=gain,
    #         cov_mode="GAUSS"
    #     )
    #     samples.append(samples_)
    #     labels.append(r"$g_\mathrm{o}$"+rf"$={gain}$")
    # colors = ['#32327B', '#3838E2', '#52A9F5']
    # plot_pairgrid_with_groundtruth_jrnnm(samples, theta_true_list, labels, colors)
    # plt.savefig(PATH_EXPERIMENT + f"pairplot_GAUSS_varying_gain_{dim}d.png")
    # plt.savefig(PATH_EXPERIMENT + f"pairplot_GAUSS_varying_gain_{dim}d.pdf")
    # plt.clf()