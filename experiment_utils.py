import torch

def load_losses(task_name, n_train, lr, path):
    losses = torch.load(path + f'{task_name}/n_train_{n_train}_n_epochs_1000_lr_{lr}/train_losses.pkl')
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    return train_losses, val_losses

def dist_to_dirac(samples, theta_true, metrics=["mse", "mmd"]):
    dist_to_dirac = {metric: [] for metric in metrics}

    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    if "mse" in metrics:
        mse = []
        for j in range(len(theta_true)):
            samples_coordj = samples[:, j]
            mse.append((samples_coordj - theta_true[j]).square().mean())
        dist_to_dirac["mse"] = torch.stack(mse).mean()

    # if "wasserstein" in metrics:
    #     wd = []
    #     for j in range(len(theta_true)):
    #         samples_coordj = samples[:, j]
    #         sd = torch.sqrt(samples_coordj.var())
    #         wd.append((samples_coordj.var() + (samples_coordj.mean() - theta_true[j])**2) / sd)
    #     dist_to_dirac["wasserstein"] = torch.stack(wd).mean()
    if "mmd" in metrics:
        mmd = []
        for j in range(len(theta_true)):
            samples_coordj = samples[:, j]
            sd = torch.sqrt(samples_coordj.var())
            mmd.append((samples_coordj.var() + (samples_coordj.mean() - theta_true[j])**2) / sd)
        dist_to_dirac["mmd"] = torch.stack(mmd).mean()
    return dist_to_dirac