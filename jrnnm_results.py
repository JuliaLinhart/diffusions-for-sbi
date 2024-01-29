import torch
import matplotlib.pyplot as plt

# from ot import sliced_wasserstein_distance
from experiment_utils import dist_to_dirac
from plot_utils import METHODS_STYLE, METRICS_STYLE, set_plotting_style

PATH_EXPERIMENT = "results/jrnnm/"
DIMS = [3, 4]
N_OBS = [1, 8, 14, 22, 30]
METRICS = ["mmd_to_dirac"]
N_EPOCHS = 1000
LR = 1e-3

def load_losses(dim, lr=LR, n_epochs=N_EPOCHS):
    filename = PATH_EXPERIMENT + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/train_losses.pkl"
    losses = torch.load(filename)
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    return train_losses, val_losses


def load_results(dim, result_name, n_obs, lr=LR, n_epochs=N_EPOCHS, gain=0.0, cov_mode=None, langevin=False, clip=False):
    theta_true = [135.0, 220.0, 2000.0, gain]
    if dim == 3:
        theta_true = theta_true[:3]
    path = PATH_EXPERIMENT + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/"
    path = path + "langevin_steps_400_5/" if langevin else path + "euler_steps_1000/"
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
    parser.add_argument("--w_dist", action="store_true")

    args = parser.parse_args()

    set_plotting_style()

    if args.losses:

        # plot losses function to select lr
        fig, axs = plt.subplots(2, 1, figsize=(5, 5), constrained_layout=True)
        for i, dim in enumerate(DIMS):
            # for j, lr in enumerate(LR):
            train_losses, val_losses = load_losses(dim)
            axs[i].plot(train_losses, label=f"train")#, lr={lr}")
            axs[i].plot(val_losses, label=f"val")#, lr={lr}")
            axs[i].set_title(rf"${dim}$D")
            axs[i].set_xlabel("epochs")
            axs[i].set_ylabel("loss")
            axs[i].set_ylim([0,0.5])
            axs[i].legend()
        plt.savefig(PATH_EXPERIMENT + "losses.png")
        plt.clf()

    if args.w_dist:
        gain = 0.0

        # plot mean distance to true theta as function of n_obs
        for metric in METRICS:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            for i, dim in enumerate(DIMS):
                for method in METHODS_STYLE.keys():
                    dist_dict = compute_distance_to_true_theta(
                        dim, 
                        cov_mode=method.split("_")[0],
                        langevin=True if "LANGEVIN" in method else False,
                        clip=True if "clip" in method else False,
                    )
                    axs[i].plot(N_OBS, [dist_dict[n_obs]["mmd"] for n_obs in N_OBS], alpha=0.7, linewidth=3, **METHODS_STYLE[method])
                axs[i].set_xticks(N_OBS)
                axs[i].set_xlabel(r"$n$")
                axs[i].set_ylabel(f"{METRICS_STYLE[metric]['label']}")
                # axs[i].set_ylim([0, 1000])
                axs[i].legend()
                axs[i].set_title(rf"${dim}$D")

            # plt.suptitle(rf"MMD to $\theta^\star = (135, 220, 2000, {gain})$")
            plt.savefig(PATH_EXPERIMENT + f"{metric}_n_obs_g_{gain}.png")
            plt.savefig(PATH_EXPERIMENT + f"{metric}_n_obs_g_{gain}.pdf")
            plt.clf()

    # # runtime comparison
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    # for i, task_name in enumerate(TASKS):
    #     times_jac = []
    #     times_gauss = []
    #     times_langevin = []
    #     for n_obs in N_OBS:
    #         _, time_j = load_results(
    #             task_name, lr=1e-3, n_obs=n_obs, gain=gain, cov_mode="JAC"
    #         )
    #         _, time_g = load_results(
    #             task_name, lr=1e-3, n_obs=n_obs, gain=gain, cov_mode="GAUSS"
    #         )
    #         _, time_l = load_results(
    #             task_name, lr=1e-3, n_obs=n_obs, gain=gain, langevin=True
    #         )
    #         times_jac.append(time_j)
    #         times_gauss.append(time_g)
    #         times_langevin.append(time_l)
    #     axs[i].plot(N_OBS, times_jac, marker="o", label=f"JAC")
    #     axs[i].plot(N_OBS, times_gauss, marker="o", label=f"GAUSS")
    #     axs[i].plot(N_OBS, times_langevin, marker="o", label=f"langevin")
    #     axs[i].set_xticks(N_OBS)
    #     axs[i].set_xlabel("n_obs")
    #     axs[i].legend()
    #     axs[i].set_title(f"{task_name}")
    # plt.suptitle(f"Runtime comparison")
    # plt.savefig(PATH_EXPERIMENT + f"runtime_comparison_n_obs_g_{gain}.png")
    # plt.savefig(PATH_EXPERIMENT + f"runtime_comparison_n_obs_g_{gain}.pdf")
    # plt.clf()
