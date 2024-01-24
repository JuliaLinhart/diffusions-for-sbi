import torch
import matplotlib.pyplot as plt

# from ot import sliced_wasserstein_distance
from experiment_utils import load_losses, dist_to_dirac

PATH_EXPERIMENT = 'results/jrnnm/'
TASKS = ["3d", "4d"]
LR = [1e-3, 1e-4]
N_OBS = [1,8,14,22,30]
METRICS = ["mse", "mmd"]

def load_results(task_name, lr, n_obs, gain=0.0, cov_mode=None, langevin=False):
    theta_true = [135.0, 220.0, 2000.0, gain]
    if task_name == "3d":
        theta_true = theta_true[:3]
    if langevin:
        filename = PATH_EXPERIMENT + f'{task_name}/n_train_50000_n_epochs_1000_lr_{lr}/langevin_steps_400_5/'
    else:
        filename = PATH_EXPERIMENT + f'{task_name}/n_train_50000_n_epochs_1000_lr_{lr}/euler_steps_1000/'
    filename_samples = filename + f'posterior_samples_{theta_true}_n_obs_{n_obs}.pkl'
    filename_time = filename + f'time_{theta_true}_n_obs_{n_obs}.pkl'
    if cov_mode is not None:
        filename_samples = filename_samples[:-4] + f"_{cov_mode}.pkl"
        filename_time = filename_time[:-4] + f"_{cov_mode}.pkl"
    samples = torch.load(filename_samples)
    time = torch.load(filename_time)
    return samples, time

# # plot losses function to select lr
# fig, axs = plt.subplots(2, 1, figsize=(5, 5), constrained_layout=True)
# for i, task_name in enumerate(TASKS):
#     for j, lr in enumerate(LR):
#         train_losses, val_losses = load_losses(task_name, n_train=50000, lr=lr, path=PATH_EXPERIMENT)
#         axs[i].plot(train_losses, label=f"train, lr={lr}")
#         axs[i].plot(val_losses, label=f"val, lr={lr}")
#         axs[i].set_title(f"{task_name}")
#         axs[i].set_xlabel("epochs")
#         axs[i].set_ylim([0,0.5])
#         axs[i].legend()
# plt.savefig(PATH_EXPERIMENT + "losses.png")
# plt.clf()


# compute mean distance to true theta over all observations 
def compute_distance_to_true_theta(task_name, gain=0.0, cov_mode=None, langevin=False):
    true_parameters = torch.tensor([135.0, 220.0, 2000.0, gain])
    if task_name == "3d":
        true_parameters = true_parameters[:3]
    dist_dict = dict(zip(N_OBS, [[]]*len(N_OBS)))
    for n_obs in N_OBS:
        samples, _ = load_results(task_name, lr=1e-3, n_obs=n_obs, gain=gain, cov_mode=cov_mode, langevin=langevin)
        dist_dict[n_obs] = dist_to_dirac(samples, true_parameters)
    return dist_dict


gain = 0.0
# # plot mean distance to true theta as function of n_obs
# for metric in METRICS:
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
#     for i, task_name in enumerate(TASKS):
#         for cov_mode in ["JAC", "GAUSS", "GAUSS_clip"]:
#             dist_dict = compute_distance_to_true_theta(task_name, cov_mode=cov_mode)
#             axs[i].plot(N_OBS, [dist_dict[n_obs][metric] for n_obs in N_OBS], marker = 'o', label=f"{metric} ({cov_mode})")
#         dist_dict = compute_distance_to_true_theta(task_name, gain=gain, langevin=True)
#         axs[i].plot(N_OBS, [dist_dict[n_obs][metric] for n_obs in N_OBS], marker = 'o', label=f"{metric} (langevin)")
#         axs[i].set_xticks(N_OBS)
#         axs[i].set_xlabel("n_obs")
#         axs[i].legend()
#         axs[i].set_title(f"{task_name}")
        
#     plt.suptitle(f"Distance to true theta = (135, 220, 2000, {gain})")
#     plt.savefig(PATH_EXPERIMENT + f"{metric}_distance_to_true_theta_n_obs_g_{gain}.png")
#     plt.savefig(PATH_EXPERIMENT + f"{metric}_distance_to_true_theta_n_obs_g_{gain}.pdf")
#     plt.clf()

# runtime comparison 
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for i, task_name in enumerate(TASKS):
    times_jac = []
    times_gauss = []
    times_langevin = []
    for n_obs in N_OBS:
        _, time_j = load_results(task_name, lr=1e-3, n_obs=n_obs, gain=gain, cov_mode="JAC")
        _, time_g = load_results(task_name, lr=1e-3, n_obs=n_obs, gain=gain, cov_mode="GAUSS")
        _, time_l = load_results(task_name, lr=1e-3, n_obs=n_obs, gain=gain, langevin=True)
        times_jac.append(time_j)
        times_gauss.append(time_g)
        times_langevin.append(time_l)
    axs[i].plot(N_OBS, times_jac, marker = 'o', label=f"JAC")
    axs[i].plot(N_OBS, times_gauss, marker = 'o', label=f"GAUSS")
    axs[i].plot(N_OBS, times_langevin, marker = 'o', label=f"langevin")
    axs[i].set_xticks(N_OBS)
    axs[i].set_xlabel("n_obs")
    axs[i].legend()
    axs[i].set_title(f"{task_name}")
plt.suptitle(f"Runtime comparison")
plt.savefig(PATH_EXPERIMENT + f"runtime_comparison_n_obs_g_{gain}.png")
plt.savefig(PATH_EXPERIMENT + f"runtime_comparison_n_obs_g_{gain}.pdf")
plt.clf()
