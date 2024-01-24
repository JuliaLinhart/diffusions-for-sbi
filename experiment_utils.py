import torch

def load_losses(task_name, n_train, lr, path):
    losses = torch.load(path + f'{task_name}/n_train_{n_train}_n_epochs_1000_lr_{lr}/train_losses.pkl')
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    return train_losses, val_losses

def dist_to_dirac(samples, theta_true, metrics=["mse", "mmd"], percentage=0.01):
    samples = samples.clone()

    dist_to_dirac = {metric: [] for metric in metrics}

    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    mmd, mse = [], []
    for j in range(len(theta_true)):
        samples_coordj = samples[:, j]
        # remove threshold percent of largest squared values
        if percentage > 0:
            samples_coordj = samples_coordj[samples_coordj.square().argsort()[:-int(len(samples_coordj)*percentage)]]
        # samples_coordj[samples_coordj.square() > (theta_true[j].square()*threshold)] = theta_true[j]
        if "mse" in metrics:
            mse.append((samples_coordj - theta_true[j]).square().mean())
        if "mmd" in metrics:
            sd = torch.sqrt(samples_coordj.var())
            mmd.append((samples_coordj.var() * sd + (samples_coordj.mean() - theta_true[j]).square()))
    dist_to_dirac["mmd"] = torch.stack(mmd).mean()
    dist_to_dirac["mse"] = torch.stack(mse).mean()

    return dist_to_dirac

def count_outliers(samples, theta_true, threshold=100):
    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    outliers = 0
    for j in range(len(theta_true)):
        samples_coordj = samples[:, j]
        threshold = theta_true[j].square()*threshold
        outliers += (samples_coordj.square() > threshold).sum().item()
    return outliers * 100 / (len(theta_true) * len(samples))

def remove_outliers(samples, theta_true, threshold=100, percentage=None):
    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    for j in range(len(theta_true)):
        samples_coordj = samples[:, j]
        if percentage is not None:
            if percentage > 0:
                # remove the percentage of largest squared values
                samples[:, j][samples_coordj.square().argsort()[-int(len(samples_coordj)*percentage):]] = theta_true[j]
        else:
            samples[:, j][samples_coordj.square() > (theta_true[j].square()*threshold)] = theta_true[j]
    return samples

def gaussien_wasserstein(X1, X2):
    mean1 = torch.mean(X1, dim=1)
    mean2 = torch.mean(X2, dim=1)
    cov1 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X1)
    cov2 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X2)
    return (torch.linalg.norm(mean1 - mean2, dim=-1)**2 + torch.linalg.matrix_norm(cov1 - cov2, dim=(-2, -1))**2).item()