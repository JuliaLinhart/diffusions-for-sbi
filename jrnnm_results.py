import torch
import matplotlib.pyplot as plt

# from ot import sliced_wasserstein_distance
from experiment_utils import dist_to_dirac
from plot_utils import METHODS_STYLE, METRICS_STYLE, set_plotting_style, plot_pairgrid_with_groundtruth_jrnnm

PATH_EXPERIMENT = "results/jrnnm/"
DIMS = [3, 4]
N_OBS = [1, 8, 14, 22, 30]
METRICS = ["mmd_to_dirac"]
N_EPOCHS = 5000
LR = 1e-4

method_names = ["GAUSS", "GAUSS_clip", "LANGEVIN", "LANGEVIN_clip", "JAC", "JAC_clip"]

def load_losses(dim, lr=LR, n_epochs=N_EPOCHS):
    filename = PATH_EXPERIMENT + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/train_losses.pkl"
    losses = torch.load(filename)
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    best_epoch = losses["best_epoch"]
    return train_losses, val_losses, best_epoch


def load_results(dim, result_name, n_obs, lr=LR, n_epochs=N_EPOCHS, gain=0.0, cov_mode=None, sampler="ddim", langevin=False, clip=False, single_obs=None):
    theta_true = [135.0, 220.0, 2000.0, gain]
    if dim == 3:
        theta_true = theta_true[:3]
    path = PATH_EXPERIMENT + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/"
    if langevin:
        path = path + "langevin_steps_400_5/" 
    else:
        path = path + f"{sampler}_steps_1000/" if sampler == "euler" or cov_mode == "GAUSS" else path + f"{sampler}_steps_400/"
    if single_obs is not None:
        path = path + f"single_obs/num_{i}_" + result_name + f"_{theta_true}_n_obs_1.pkl"
    else:
        path  = path + result_name + f"_{theta_true}_n_obs_{n_obs}.pkl"
    if not langevin:
        path = path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        path = path[:-4] + "_clip.pkl"
    results = torch.load(path)
    return results

# compute mean distance to true theta over all observations
def compute_distance_to_true_theta(dim, gain=0.0, cov_mode=None, langevin=False, clip=False):
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
        dist_dict[n_obs] = dist_to_dirac(samples, true_parameters, metrics=["mmd"], scaled=True)
    return dist_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", action="store_true")
    parser.add_argument("--dirac_dist", action="store_true")
    parser.add_argument("--pairplot", action="store_true")
    parser.add_argument("--single_multi_obs", action="store_true")

    args = parser.parse_args()

    alpha, alpha_fill= set_plotting_style()

    if args.losses:
        # plot losses function to select lr
        lr_list = [1e-4, 1e-3]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.2)
        for i, dim in enumerate([3,4]):
            best_val_loss = {}
            for lr_, c in zip(lr_list, ["blue", "orange"]):
                train_losses, val_losses, best_epoch = load_losses(
                    dim, lr_, n_epochs=N_EPOCHS
                )
                best_val_loss[lr_] = val_losses[best_epoch]
                axs[i].plot(train_losses, label=f"train (lr={lr_})", color=c, linewidth=3, alpha=0.3)
                axs[i].plot(val_losses, label=f"val", color=c, linewidth=3, alpha=0.9)
                axs[i].axvline(best_epoch, color=c, linestyle="--", linewidth=5, alpha=0.9, zorder=10000)
            # print(best_val_loss)
            # get lr for min best val loss
            best_lr = min(best_val_loss, key=best_val_loss.get)
            print(f"best lr for {dim}D: {best_lr}")
            axs[i].set_title(fr"${dim}$D") #+ f" \n best_lr={best_lr}")
            axs[i].set_ylim([0, 0.5])
            axs[i].set_xlim([0, 5000])
            axs[i].set_xlabel("epochs")
        # axs[i].set_ylabel("loss")
        axs[i].legend()
        plt.savefig(PATH_EXPERIMENT + "jrnnm_losses.png")
        plt.savefig(PATH_EXPERIMENT + "jrnnm_losses.pdf")
        plt.clf()

    if args.dirac_dist:
        gain = 0.0

        # plot mean distance to true theta as function of n_obs
        for metric in METRICS:
            for i, dim in enumerate(DIMS):
                fig, axs = plt.subplots(1, 1, figsize=(5, 5))
                fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.25)
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
                        axs.plot(N_OBS, [dist_dict[n_obs]["mmd"] for n_obs in N_OBS], alpha=alpha, **METHODS_STYLE[method])
                axs.set_xticks(N_OBS)
                axs.set_xlabel(r"$n$")
                axs.set_ylabel(f"{METRICS_STYLE[metric]['label']}")
                if dim == 4:
                    axs.set_ylim([0, 600])
                # axs.legend()
                # axs.set_title(rf"${dim}$D")

                # plt.suptitle(rf"MMD to $\theta^\star = (135, 220, 2000, {gain})$")
                plt.savefig(PATH_EXPERIMENT + f"{metric}_n_obs_g_{gain}_dim_{dim}.png")
                plt.savefig(PATH_EXPERIMENT + f"{metric}_n_obs_g_{gain}_dim_{dim}.pdf")
                plt.clf()

    if args.pairplot:
        from matplotlib import colormaps as cm
        gain = 0.0
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
            colors = [cm.get_cmap("viridis")(i) for i in torch.linspace(0.2, 1, len(N_OBS))]
            plot_pairgrid_with_groundtruth_jrnnm(samples, [theta_true], labels, colors)
            plt.savefig(PATH_EXPERIMENT + f"pairplot_{method}_g_{gain}_{dim}d.png")
            plt.savefig(PATH_EXPERIMENT + f"pairplot_{method}_g_{gain}_{dim}d.pdf")
            plt.clf()

    if args.single_multi_obs:
        # plot independent posterior samples for single and multi-observation

        from matplotlib import colormaps as cm
        import seaborn as sns

        colors = [cm.get_cmap("viridis")(i) for i in torch.linspace(0.2, 1, len(N_OBS))]
        gain = 0.0
        method = "GAUSS"
        for dim in DIMS:
            theta_true = [135.0, 220.0, 2000.0, gain]
            param_names = ["$C$", "$\mu$", "$\sigma$", "$g$"]
            if dim == 3:
                theta_true = theta_true[:3]
                param_names = param_names[:3]

            # for j, n_obs in enumerate(N_OBS):
            #     if n_obs == 1:
            #         continue
            #     else:
            #         samples = load_results(
            #             dim,
            #             result_name="posterior_samples",
            #             n_obs=n_obs,
            #             gain=gain,
            #             cov_mode=method.split("_")[0],
            #             langevin=True if "LANGEVIN" in method else False,
            #             clip=True if "clip" in method else False,
            #         )
            n_obs = 30
            fig, axs = plt.subplots(1, dim, figsize=(5*dim, 5))
            fig.subplots_adjust(right=.995, top=.92, bottom=.2, hspace=0, wspace=0, left=.1)
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
                        sns.kdeplot(samples_single[:, k], alpha=0.1, color=colors[0], ax=axs[k], linewidth=3, fill=True, label=rf"$n=1$")
                    else:
                        sns.kdeplot(samples_single[:, k], alpha=0.1, color=colors[0], ax=axs[k], linewidth=3, fill=True)
                axs[k].legend()
            for j,n_obs in enumerate(N_OBS[1:]):
                samples = load_results(
                    dim,
                    result_name="posterior_samples",
                    n_obs=n_obs,
                    gain=gain,
                    cov_mode=method.split("_")[0],
                    langevin=True if "LANGEVIN" in method else False,
                    clip=True if "clip" in method else False,
                )

                for k,name in enumerate(param_names):
                    sns.kdeplot(samples[:, k], alpha=0.1, label=rf"$n={n_obs}$", color=colors[j+1], ax=axs[k], linewidth=3, fill=True)
                    # line for theta_true 
                    axs.ravel()[k].axvline(theta_true[k], ls="--", linewidth=3, c="black")
                    axs[k].set_xlabel(name)
                    axs[k].set_ylabel("")
                    # emply list for ytick labels
                    axs[k].set_yticklabels([])
                axs[k].legend()
            plt.savefig(PATH_EXPERIMENT + f"single_multi_obs_{method}_g_{gain}_{dim}d.png")
            plt.savefig(PATH_EXPERIMENT + f"single_multi_obs_{method}_g_{gain}_{dim}d.pdf")
            plt.clf()
            