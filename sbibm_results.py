import torch
import matplotlib.pyplot as plt

# from ot import sliced_wasserstein_distance
from experiment_utils import load_losses, dist_to_dirac, count_outliers
from tasks.sbibm.data_generators import get_task
from plot_utils import multi_corner_plots, set_plotting_style

from tqdm import tqdm

PATH_EXPERIMENT = 'results/sbibm/'
TASKS = ["gaussian_linear", "gaussian_mixture","sir", "lotka_volterra"]
N_TRAIN = [1000, 3000, 10000, 30000] 
LR = [1e-3, 1e-4]
N_OBS = [1,8,14,22,30]
NUM_OBSERVATION_LIST = [1,2,3,4,5,6,7,8,9,10]
METRICS = ["mmd", "mse"]

def load_samples(task_name, n_train, lr, n_obs, cov_mode=None, langevin=False):
    samples = dict(zip(NUM_OBSERVATION_LIST, [[]]*10))
    for num_obs in NUM_OBSERVATION_LIST:
        if langevin:
            filename = PATH_EXPERIMENT + f'{task_name}/n_train_{n_train}_n_epochs_1000_lr_{lr}/langevin_steps_400_5/posterior_samples_{num_obs}_n_obs_{n_obs}.pkl'
        else:
            filename = PATH_EXPERIMENT + f'{task_name}/n_train_{n_train}_n_epochs_1000_lr_{lr}/euler_steps_1000/posterior_samples_{num_obs}_n_obs_{n_obs}_{cov_mode}.pkl'
        samples[num_obs] = torch.load(filename)
    return samples

# compute mean distance to true theta over all observations 
def compute_mean_distance_to_true_theta(task_name, n_train, n_obs, cov_mode=None, langevin=False, percentage=0):
    task = get_task(task_name)
    samples = load_samples(task_name, n_train=n_train, lr=1e-4, n_obs=n_obs, cov_mode=cov_mode, langevin=langevin)
    
    # outliers
    outliers = [count_outliers(samples[num_obs], task.get_true_parameters(num_obs) ) for num_obs in NUM_OBSERVATION_LIST]
    outlier_dict = {"mean": torch.tensor(outliers).mean().item(), "std": torch.tensor(outliers).std().item()}
    # compute dist between samples and true theta
    dist_dict = {"mmd": [], "mse": []}
    for num_obs in NUM_OBSERVATION_LIST:
        dist = dist_to_dirac(samples[num_obs], task.get_true_parameters(num_obs), percentage=percentage)
        for metric in METRICS:
            dist_dict[metric].append(dist[metric])
    for metric in METRICS:
        dist_dict[metric] = {"mean": torch.tensor(dist_dict[metric]).mean(), "std": torch.tensor(dist_dict[metric]).std()}    
    
    return dist_dict, outlier_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--losses', action='store_true')
    parser.add_argument('--dist_nobs', action='store_true')
    parser.add_argument('--dist_ntrain', action='store_true')
    parser.add_argument('--corner_plots', action='store_true')
    parser.add_argument('--outliers', action='store_true')

    args = parser.parse_args()

    set_plotting_style()

    if args.losses:
        # plot losses to select lr
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        for i, task_name in enumerate(TASKS):
            for j, n_train in enumerate(N_TRAIN):
                for k, lr in enumerate(LR):
                    train_losses, val_losses = load_losses(task_name, n_train, lr, path=PATH_EXPERIMENT)
                    axs[i, j].plot(train_losses, label=f"train, lr={lr}")
                    axs[i, j].plot(val_losses, label=f"val, lr={lr}")
                    axs[i, j].set_title(f"{task_name}, n_train={n_train}")
                    axs[i, j].set_xlabel("epochs")
                    axs[i, j].set_ylim([0,0.5])
                    axs[i, j].legend()
        plt.savefig(PATH_EXPERIMENT + "losses.png")
        plt.clf()

    if args.dist_nobs:
        # plot mean distance to true theta as function of n_obs
        for metric in METRICS:
            fig, axs = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)
            for i, task_name in enumerate(TASKS):
                task = get_task(task_name)
                for j,n_train in tqdm(enumerate(N_TRAIN), desc=f"{task_name}, {metric}"):
                    mean_dist_l = []
                    std_dist_l = []
                    mean_out_l = []
                    mean_dist_g = []
                    std_dist_g = []
                    mean_out_g = []
                    # mean_dist_j = []
                    # std_dist_j = []
                    # mean_out_j = []
                    for n_obs in N_OBS:
                        # langevin
                        dist_dict, outlier_dict = compute_mean_distance_to_true_theta(task_name, n_train=n_train, n_obs=n_obs, langevin=True)
                        mean_dist_l.append(dist_dict[metric]["mean"])
                        std_dist_l.append(dist_dict[metric]["std"])
                        mean_out_l.append(outlier_dict["mean"])
                        # gauss
                        dist_dict, outlier_dict = compute_mean_distance_to_true_theta(task_name, n_train=n_train, n_obs=n_obs, cov_mode="GAUSS")
                        mean_dist_g.append(dist_dict[metric]["mean"])
                        std_dist_g.append(dist_dict[metric]["std"])
                        mean_out_g.append(outlier_dict["mean"])
                        # jac
                        # if task_name in ["sir", "lotka_volterra"]:
                        #     dist_dict, outlier_dict = compute_mean_distance_to_true_theta(task_name, n_train=n_train, n_obs=n_obs, cov_mode="JAC")
                        #     mean_dist_j.append(dist_dict[metric]["mean"])
                        #     std_dist_j.append(dist_dict[metric]["std"])
                        #     mean_out_j.append(outlier_dict["mean"])
                    mean_dist_l, std_dist_l = torch.FloatTensor(mean_dist_l), torch.FloatTensor(std_dist_l)
                    mean_dist_g, std_dist_g = torch.FloatTensor(mean_dist_g), torch.FloatTensor(std_dist_g)
                    # mean_dist_j, std_dist_j = torch.FloatTensor(mean_dist_j), torch.FloatTensor(std_dist_j)
                    axs[i, j].fill_between(N_OBS, mean_dist_l - std_dist_l, mean_dist_l + std_dist_l, alpha=0.2)
                    axs[i, j].plot(N_OBS, mean_dist_l, label=f"{metric} (langevin)", linewidth=3, marker='o')
                    axs[i, j].fill_between(N_OBS, mean_dist_g - std_dist_g, mean_dist_g + std_dist_g, alpha=0.2)
                    axs[i, j].plot(N_OBS, mean_dist_g, label=f"{metric} (gauss)", linewidth=3, marker='o')
                    # if task_name in ["sir", "lotka_volterra"]:
                    #     axs[i, j].fill_between(N_OBS, mean_dist_j - std_dist_j, mean_dist_j + std_dist_j, alpha=0.2)
                    #     axs[i, j].plot(N_OBS, mean_dist_j, label=f"{metric} (jac)", linewidth=3, marker='o')
                    axs[i, j].set_title(f"{task_name}, n_train={n_train}")
                    axs[i, j].set_xlabel("n_obs")
                    axs[i, j].set_xticks(N_OBS)
                    axs[i, j].legend()
            plt.savefig(PATH_EXPERIMENT + f"{metric}_distance_to_true_theta_n_obs.png")
            plt.savefig(PATH_EXPERIMENT + f"{metric}_distance_to_true_theta_n_obs.pdf")
            plt.clf()

    if args.dist_ntrain:
        PERCENTAGES = [0, 0.01, 0.05, 0.1]
        for percentage in PERCENTAGES:
            # plot mean distance to true theta as function of n_train
            for metric in METRICS:
                fig, axs = plt.subplots(4, 5, figsize=(25, 20), constrained_layout=True)
                for i, task_name in enumerate(TASKS):
                    task = get_task(task_name)
                    for j,n_obs in tqdm(enumerate(N_OBS), desc=f"{task_name}, {metric}"):
                        mean_dist_l = []
                        std_dist_l = []
                        mean_out_l = []
                        mean_dist_g = []
                        std_dist_g = []
                        mean_out_g = []
                        # mean_dist_j = []
                        # std_dist_j = []
                        # mean_out_j = []
                        for n_train in N_TRAIN:
                            # langevin
                            dist_dict, outlier_dict = compute_mean_distance_to_true_theta(task_name, n_train=n_train, n_obs=n_obs, langevin=True, percentage=percentage)
                            mean_dist_l.append(dist_dict[metric]["mean"])
                            std_dist_l.append(dist_dict[metric]["std"])
                            mean_out_l.append(outlier_dict["mean"])
                            # gauss
                            dist_dict, outlier_dict = compute_mean_distance_to_true_theta(task_name, n_train=n_train, n_obs=n_obs, cov_mode="GAUSS", percentage=percentage)
                            mean_dist_g.append(dist_dict[metric]["mean"])
                            std_dist_g.append(dist_dict[metric]["std"])
                            mean_out_g.append(outlier_dict["mean"])
                            # jac
                            # if task_name in ["sir", "lotka_volterra"]:
                            #     dist_dict, outlier_dict = compute_mean_distance_to_true_theta(task_name, n_train=n_train, n_obs=n_obs, cov_mode="JAC", percentage=percentage)
                            #     mean_dist_j.append(dist_dict[metric]["mean"])
                            #     std_dist_j.append(dist_dict[metric]["std"])
                            #     mean_out_j.append(outlier_dict["mean"])
                        mean_dist_l, std_dist_l = torch.FloatTensor(mean_dist_l), torch.FloatTensor(std_dist_l)
                        mean_dist_g, std_dist_g = torch.FloatTensor(mean_dist_g), torch.FloatTensor(std_dist_g)
                        # mean_dist_j, std_dist_j = torch.FloatTensor(mean_dist_j), torch.FloatTensor(std_dist_j)
                        axs[i, j].fill_between(N_TRAIN, mean_dist_l - std_dist_l, mean_dist_l + std_dist_l, alpha=0.2)
                        axs[i, j].plot(N_TRAIN, mean_dist_l, label=f"{metric} (langevin)", linewidth=3, marker='o')
                        axs[i, j].fill_between(N_TRAIN, mean_dist_g - std_dist_g, mean_dist_g + std_dist_g, alpha=0.2)
                        axs[i, j].plot(N_TRAIN, mean_dist_g, label=f"{metric} (gauss)", linewidth=3, marker='o')
                        # outlier: {mean_out_g}
                        # if task_name in ["sir", "lotka_volterra"]:
                        #     axs[i, j].fill_between(N_TRAIN, mean_dist_j - std_dist_j, mean_dist_j + std_dist_j, alpha=0.2)
                        #     axs[i, j].plot(N_TRAIN, mean_dist_j, label=f"{metric} (jac)", linewidth=3, marker='o')
                        axs[i, j].set_title(f"{task_name}, n_obs={n_obs}")
                        axs[i, j].set_xlabel("n_train")
                        axs[i, j].set_xticks(N_TRAIN)
                        axs[i, j].set_xscale("log")
                        axs[i, j].legend()
                plt.savefig(PATH_EXPERIMENT + f"{metric}_distance_to_true_theta_n_train_percentage_{percentage}.png")
                plt.savefig(PATH_EXPERIMENT + f"{metric}_distance_to_true_theta_n_train_percentage_{percentage}.pdf")
                plt.clf()

    if args.corner_plots:
        # corner plots for every n_obs
        num_obs = 1
        for task_name in TASKS:
            true_thetas = get_task(task_name).get_true_parameters(num_obs).repeat(1000,1)
            print(task_name, true_thetas.shape, true_thetas[0])
            if task_name == "sir":
                cov_mode = "JAC"
            else:
                cov_mode = "GAUSS"
            for n_train in [30000]:
                samples = []
                for n_obs in N_OBS:
                    samples.append(load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, cov_mode=cov_mode)[num_obs])
                samples.append(true_thetas)
                print(len(samples), samples[-1].shape)
                multi_corner_plots(samples, legends=[f"n_obs={n_obs}" for n_obs in N_OBS] + ["true theta"], colors=["blue", "orange", "green", "red", "purple", "black"], title=f"{task_name}, n_train={n_train}, num_obs={num_obs}")
                plt.savefig(PATH_EXPERIMENT + f"corner_plots_{task_name}_n_train_{n_train}_num_obs_{num_obs}.png")
                plt.savefig(PATH_EXPERIMENT + f"corner_plots_{task_name}_n_train_{n_train}_num_obs_{num_obs}.pdf")
                plt.clf()

        num_obs = 1
        for task_name in TASKS:
            true_theta = get_task(task_name).get_true_parameters(num_obs).repeat(1000,1)
            print(task_name, true_theta.shape, true_theta[0])
            for n_train in [30000]:
                samples = []
                for n_obs in N_OBS:
                    samples.append(load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, langevin=True)[num_obs])
                multi_corner_plots(samples.append(true_theta), legends=[f"n_obs={n_obs}" for n_obs in N_OBS] + ["true theta"], colors=["blue", "orange", "green", "red", "purple", "black"], title=f"{task_name}, n_train={n_train}, num_obs={num_obs}")
                plt.savefig(PATH_EXPERIMENT + f"corner_plots_{task_name}_n_train_{n_train}_num_obs_{num_obs}_langevin.png")
                plt.savefig(PATH_EXPERIMENT + f"corner_plots_{task_name}_n_train_{n_train}_num_obs_{num_obs}_langevin.pdf")
                plt.clf()

    if args.outliers:
        # outliers percentage for every n_obs and n_train
        fig, axs = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)
        for i, task_name in enumerate(TASKS):
            task = get_task(task_name)
            for j, n_train in tqdm(enumerate(N_TRAIN), desc=f"{task_name}"):
                outliers_mean_langevin = []
                outliers_std_langevin = []
                outliers_mean_gauss = []
                outliers_std_gauss = []
                # outliers_mean_jac = []
                # outliers_std_jac = []
                for k, n_obs in enumerate(N_OBS):
                    true_parameters = [task.get_true_parameters(num_obs) for num_obs in NUM_OBSERVATION_LIST]
                    # langevin
                    samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, langevin=True)
                    outliers = [count_outliers(samples[num_obs], true_parameters[num_obs -1]) for num_obs in NUM_OBSERVATION_LIST]
                    outliers_mean_langevin.append(torch.FloatTensor(outliers).mean())
                    outliers_std_langevin.append(torch.FloatTensor(outliers).std())
                    # gauss
                    samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, cov_mode="GAUSS")
                    outliers = [count_outliers(samples[num_obs], true_parameters[num_obs -1]) for num_obs in NUM_OBSERVATION_LIST]
                    outliers_mean_gauss.append(torch.FloatTensor(outliers).mean())
                    outliers_std_gauss.append(torch.FloatTensor(outliers).std())
                    # # jac
                    # if task_name in ["sir", "lotka_volterra"]:
                    #     samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, cov_mode="JAC")
                    #     outliers = [count_outliers(samples[num_obs], true_parameters[num_obs -1]) for num_obs in NUM_OBSERVATION_LIST]
                    #     outliers_mean_jac.append(torch.FloatTensor(outliers).mean())
                    #     outliers_std_jac.append(torch.FloatTensor(outliers).std())

                outliers_mean_langevin, outliers_std_langevin = torch.FloatTensor(outliers_mean_langevin), torch.FloatTensor(outliers_std_langevin)
                outliers_mean_gauss, outliers_std_gauss = torch.FloatTensor(outliers_mean_gauss), torch.FloatTensor(outliers_std_gauss)
                # outliers_mean_jac, outliers_std_jac = torch.FloatTensor(outliers_mean_jac), torch.FloatTensor(outliers_std_jac)
                axs[i, j].fill_between(N_OBS, outliers_mean_langevin - outliers_std_langevin, outliers_mean_langevin + outliers_std_langevin, alpha=0.2)
                axs[i, j].plot(N_OBS, outliers_mean_langevin, label=f"LANGEVIN", linewidth=3, marker='o')
                axs[i, j].fill_between(N_OBS, outliers_mean_gauss - outliers_std_gauss, outliers_mean_gauss + outliers_std_gauss, alpha=0.2)
                axs[i, j].plot(N_OBS, outliers_mean_gauss, label=f"GAUSS", linewidth=3, marker='o')
                # if task_name in ["sir", "lotka_volterra"]:
                #     axs[i, j].fill_between(N_OBS, outliers_mean_jac - outliers_std_jac, outliers_mean_jac + outliers_std_jac, alpha=0.2)
                #     axs[i, j].plot(N_OBS, outliers_mean_jac, label=f"JAC", linewidth=3, marker='o')
                axs[i, j].set_title(f"{task_name}, n_train={n_train}")
                axs[i, j].set_xlabel("n_obs")
                axs[i, j].set_xticks(N_OBS)
                axs[i, j].legend()
        plt.savefig(PATH_EXPERIMENT + f"outliers_nobs.png")
        plt.savefig(PATH_EXPERIMENT + f"outliers_nobs.pdf")
        plt.clf()

        # same as function of n_train
        fig, axs = plt.subplots(4, 5, figsize=(50, 40), constrained_layout=True)
        for i, task_name in enumerate(TASKS):
            task = get_task(task_name)
            for j, n_obs in tqdm(enumerate(N_OBS), desc=f"{task_name}"):
                outliers_mean_langevin = []
                outliers_std_langevin = []
                outliers_mean_gauss = []
                outliers_std_gauss = []
                # outliers_mean_jac = []
                # outliers_std_jac = []
                for k, n_train in enumerate(N_TRAIN):
                    true_parameters = [task.get_true_parameters(num_obs) for num_obs in NUM_OBSERVATION_LIST]
                    # langevin
                    samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, langevin=True)
                    outliers = [count_outliers(samples[num_obs], true_parameters[num_obs -1]) for num_obs in NUM_OBSERVATION_LIST]
                    outliers_mean_langevin.append(torch.FloatTensor(outliers).mean())
                    outliers_std_langevin.append(torch.FloatTensor(outliers).std())
                    # gauss
                    samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, cov_mode="GAUSS")
                    outliers = [count_outliers(samples[num_obs], true_parameters[num_obs -1]) for num_obs in NUM_OBSERVATION_LIST]
                    outliers_mean_gauss.append(torch.FloatTensor(outliers).mean())
                    outliers_std_gauss.append(torch.FloatTensor(outliers).std())
                    # # jac
                    # if task_name in ["sir", "lotka_volterra"]:
                    #     samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, cov_mode="JAC")
                    #     outliers = [count_outliers(samples[num_obs], true_parameters[num_obs -1]) for num_obs in NUM_OBSERVATION_LIST]
                    #     outliers_mean_jac.append(torch.FloatTensor(outliers).mean())
                    #     outliers_std_jac.append(torch.FloatTensor(outliers).std())

                outliers_mean_langevin, outliers_std_langevin = torch.FloatTensor(outliers_mean_langevin), torch.FloatTensor(outliers_std_langevin)
                outliers_mean_gauss, outliers_std_gauss = torch.FloatTensor(outliers_mean_gauss), torch.FloatTensor(outliers_std_gauss)
                # outliers_mean_jac, outliers_std_jac = torch.FloatTensor(outliers_mean_jac), torch.FloatTensor(outliers_std_jac)
                axs[i, j].fill_between(N_TRAIN, outliers_mean_langevin - outliers_std_langevin, outliers_mean_langevin + outliers_std_langevin, alpha=0.2)
                axs[i, j].plot(N_TRAIN, outliers_mean_langevin, label=f"LANGEVIN", linewidth=3, marker='o')
                axs[i, j].fill_between(N_TRAIN, outliers_mean_gauss - outliers_std_gauss, outliers_mean_gauss + outliers_std_gauss, alpha=0.2)
                axs[i, j].plot(N_TRAIN, outliers_mean_gauss, label=f"GAUSS", linewidth=3, marker='o')
                # if task_name in ["sir", "lotka_volterra"]:
                #     axs[i, j].fill_between(N_TRAIN, outliers_mean_jac - outliers_std_jac, outliers_mean_jac + outliers_std_jac, alpha=0.2)
                #     axs[i, j].plot(N_TRAIN, outliers_mean_jac, label=f"JAC", linewidth=3, marker='o')
                axs[i, j].set_title(f"{task_name}, n_obs={n_obs}")
                axs[i, j].set_xlabel("n_train")
                axs[i, j].set_xticks(N_TRAIN)
                axs[i, j].set_xscale("log")
                axs[i, j].legend()
        plt.savefig(PATH_EXPERIMENT + f"outliers_n_train.png")
        plt.savefig(PATH_EXPERIMENT + f"outliers_n_train.pdf")
        plt.clf()

    # n_train = 1000
    # n_obs = 30
    # task_names = ['gaussian_linear', 'gaussian_mixture']
    # for task_name in task_names:
    #     samples = load_samples(task_name, n_train, lr=1e-3, n_obs=n_obs, cov_mode="GAUSS")
    #     for num_obs in NUM_OBSERVATION_LIST:
    #         true_theta = get_task(task_name).get_true_parameters(num_obs)
    #         dist = dist_to_dirac(samples[num_obs], true_theta)
    #         print(task_name, num_obs, dist["mse"], dist["mmd"])