import torch
import matplotlib.pyplot as plt

# from ot import sliced_wasserstein_distance
from experiment_utils import load_losses, dist_to_dirac

EXPERIMENT_PATH = 'results/jrnnm/'
TASKS = ["3d", "4d"]
LR = [1e-3, 1e-4]
N_OBS = [1,8,14,22,30]
METRICS = ["mse", "mmd"]

def load_samples(task_name, lr, n_obs, gain=0.0):
    theta_true = [135.0, 220.0, 2000.0, gain]
    if task_name == "3d":
        theta_true = theta_true[:3]
    samples = torch.load(EXPERIMENT_PATH + f'{task_name}/n_train_50000_n_epochs_1000_lr_{lr}/euler_steps_1000/posterior_samples_{theta_true}_n_obs_{n_obs}.pkl')
    return samples

# plot losses function to select lr
fig, axs = plt.subplots(2, 1, figsize=(5, 5), constrained_layout=True)
for i, task_name in enumerate(TASKS):
    for j, lr in enumerate(LR):
        train_losses, val_losses = load_losses(task_name, n_train=50000, lr=lr, path=EXPERIMENT_PATH)
        axs[i].plot(train_losses, label=f"train, lr={lr}")
        axs[i].plot(val_losses, label=f"val, lr={lr}")
        axs[i].set_title(f"{task_name}")
        axs[i].set_xlabel("epochs")
        axs[i].set_ylim([0,0.5])
        axs[i].legend()
plt.savefig(EXPERIMENT_PATH + "losses.png")
plt.clf()


# compute mean distance to true theta over all observations 
def compute_distance_to_true_theta(task_name, gain=0.0):
    true_parameters = torch.tensor([135.0, 220.0, 2000.0, gain])
    if task_name == "3d":
        true_parameters = true_parameters[:3]
    dist_dict = dict(zip(N_OBS, [[]]*len(N_OBS)))
    for n_obs in N_OBS:
        samples = load_samples(task_name, lr=1e-3, n_obs=n_obs, gain=gain)
        dist_dict[n_obs] = dist_to_dirac(samples, true_parameters)
    return dist_dict

gain = 0.0
# plot mean distance to true theta as function of n_obs
for metric in METRICS:
    for i, task_name in enumerate(TASKS):
        dist_dict = compute_distance_to_true_theta(task_name)
        plt.plot(N_OBS, [dist_dict[n_obs][metric] for n_obs in N_OBS], marker = 'o', label=f"{metric} ({task_name})")
    plt.xticks(N_OBS)
    plt.xlabel("n_obs")
    plt.legend()
    plt.title(f"Distance to true theta = (135.0, 220.0, 2000.0, {gain})")
    plt.savefig(EXPERIMENT_PATH + f"{metric}_distance_to_true_theta_n_obs_g_{gain}.png")
    plt.savefig(EXPERIMENT_PATH + f"{metric}_distance_to_true_theta_n_obs_g_{gain}.pdf")
    plt.clf()