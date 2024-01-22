import torch
import matplotlib.pyplot as plt
import sbibm

# from ot import sliced_wasserstein_distance
from experiment_utils import load_losses, dist_to_dirac
from tasks.sbibm.data_generators import get_task

EXPERIMENT_PATH = 'results/sbibm/'
TASKS = ["gaussian_linear", "gaussian_mixture"] #, "sir", "lotka_volterra"]
N_TRAIN = [1000, 3000] #, 10000, 30000]
LR = [1e-3, 1e-4]
N_OBS = [1,8,14,22,30]
NUM_OBSERVATION_LIST = list(range(1,11))
METRICS = ["mse", "mmd"]

def load_samples(task_name, n_train, lr, n_obs):
    samples = dict(zip(NUM_OBSERVATION_LIST, [[]]*10))
    for num_obs in NUM_OBSERVATION_LIST:
        samples[num_obs] = torch.load(EXPERIMENT_PATH + f'{task_name}/n_train_{n_train}_n_epochs_1000_lr_{lr}/euler_steps_1000/posterior_samples_{num_obs}_n_obs_{n_obs}.pkl')
    return samples

# plot losses to select lr
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
for i, task_name in enumerate(TASKS):
    for j, n_train in enumerate(N_TRAIN):
        for k, lr in enumerate(LR):
            train_losses, val_losses = load_losses(task_name, n_train, lr, path=EXPERIMENT_PATH)
            axs[i, j].plot(train_losses, label=f"train, lr={lr}")
            axs[i, j].plot(val_losses, label=f"val, lr={lr}")
            axs[i, j].set_title(f"{task_name}, n_train={n_train}")
            axs[i, j].set_xlabel("epochs")
            axs[i, j].set_ylim([0,0.5])
            axs[i, j].legend()
plt.savefig(EXPERIMENT_PATH + "losses.png")
plt.clf()

# compute mean distance to true theta over all observations 
def compute_mean_distance_to_true_theta(task_name):
    task = get_task(task_name)
    dist_dict = dict(zip(N_OBS, [[]]*len(N_OBS)))
    
    for n_obs in N_OBS:
        dist_dict[n_obs] = {"mse": {"mean":[], "std": []}, "mmd": {"mean":[], "std": []}}
        for n_train in N_TRAIN:
            samples_dict = load_samples(task_name, n_train=n_train, lr=1e-3, n_obs=n_obs)
            # compute dist between samples and true theta
            distances = [dist_to_dirac(samples_dict[num_obs], task.get_true_parameters(num_obs) ) for num_obs in NUM_OBSERVATION_LIST]

            # compute mean and std over num_obs
            for metric in distances[0].keys():
                dist_dict[n_obs][metric]["mean"].append(torch.tensor([distances[num_obs-1][metric] for num_obs in NUM_OBSERVATION_LIST]).mean())
                dist_dict[n_obs][metric]["std"].append(torch.tensor([distances[num_obs-1][metric] for num_obs in NUM_OBSERVATION_LIST]).std())
    return dist_dict

# # plot mean distance to true theta as function of n_train
# fig, axs = plt.subplots(4, 5, figsize=(20, 25))
# for i, task_name in enumerate(TASKS):
#     task = get_task(task_name)
#     mean_distance_to_true_theta, distance_to_true_theta_per_obs = compute_mean_distance_to_true_theta(task)
#     for j, n_obs in enumerate(N_OBS):
#         for num_obs in NUM_OBSERVATION_LIST:
#             axs[i, j].plot(N_TRAIN, distance_to_true_theta_per_obs[n_obs][num_obs], '--', label=f"MSE, num_obs={num_obs}")
#         axs[i, j].plot(N_TRAIN, mean_distance_to_true_theta[n_obs], label=f"mean MSE", color="black", linewidth=3)
#         axs[i, j].set_title(f"{task_name}, n_obs={n_obs}")
#         axs[i, j].set_xlabel("n_train")
#         axs[i, j].legend()

# plt.suptitle("Distance to true theta")
# plt.savefig(EXPERIMENT_PATH + "distance_to_true_theta_n_train.png")

# plot mean distance to true theta as function of n_obs
n_train = 3000
fig, axs = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
for i, task_name in enumerate(TASKS):
    task = get_task(task_name)
    dist_dict = compute_mean_distance_to_true_theta(task_name)
    for metric in METRICS:
        mean_dist = torch.tensor([dist_dict[n_obs][metric]["mean"][N_TRAIN.index(n_train)] for n_obs in N_OBS])
        std_dist = torch.tensor([dist_dict[n_obs][metric]["std"][N_TRAIN.index(n_train)] for n_obs in N_OBS])
        axs[i].fill_between(N_OBS, mean_dist - std_dist, mean_dist + std_dist, alpha=0.2)
        axs[i].plot(N_OBS, mean_dist, label=f"mean {metric}", linewidth=3, marker='o')
    axs[i].set_title(f"{task_name}, n_train={n_train}")
    axs[i].set_xlabel("n_obs")
    axs[i].set_xticks(N_OBS)
    axs[i].legend()
plt.suptitle("Distance to true theta")
plt.savefig(EXPERIMENT_PATH + "distance_to_true_theta_n_obs.png")
plt.savefig(EXPERIMENT_PATH + "distance_to_true_theta_n_obs.pdf")
plt.clf()

