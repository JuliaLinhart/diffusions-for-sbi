import torch

def dist_to_dirac(samples, theta_true, metrics=["mse", "mmd"], scaled=False, percentage=0):
    dict = {metric: [] for metric in metrics}

    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    for j in range(len(theta_true)):
        samples_coordj = samples[:, j]
        # remove threshold percent of largest squared values
        if percentage > 0:
            samples_coordj = samples_coordj[samples_coordj.square().argsort()[:-int(len(samples_coordj)*percentage)]]
        # samples_coordj[samples_coordj.square() > (theta_true[j].square()*threshold)] = theta_true[j]
        if "mse" in metrics:
            dict["mse"].append((samples_coordj - theta_true[j]).square().mean())
        if "mmd" in metrics:
            sd = torch.sqrt(samples_coordj.var())
            if scaled:
                mmd = (samples_coordj.var() + (samples_coordj.mean() - theta_true[j]).square()) / sd
            else:
                mmd = samples_coordj.var() + (samples_coordj.mean() - theta_true[j]).square()
            dict["mmd"].append(mmd)
    for metric in metrics:
        dict[metric] = torch.stack(dict[metric]).mean()

    return dict

# def dist_to_dirac(samples, theta_true):
#     if theta_true.ndim > 1:
#         theta_true = theta_true[0]
#     wd = []
#     for j in range(len(theta_true)):
#         samples_coordj = samples[:, j]
#         sd = torch.sqrt(samples_coordj.var())
#         wd.append((samples_coordj.var() + (samples_coordj.mean() - theta_true[j])**2) / sd)
#     return torch.stack(wd).mean()

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

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    L, V = torch.linalg.eig(matrix)
    L = L.real
    V = V.real
    return V @ torch.diag_embed(L.pow(p)) @ torch.linalg.inv(V)

def gaussien_wasserstein(ref_mu, ref_cov, X2):
    mean2 = torch.mean(X2, dim=1)
    sqrtcov1 = _matrix_pow(ref_cov, .5)
    cov2 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X2)
    covterm = torch.func.vmap(torch.trace)(ref_cov + cov2 - 2 * _matrix_pow(sqrtcov1 @ cov2 @ sqrtcov1, .5))
    return (1*torch.linalg.norm(ref_mu - mean2, dim=-1)**2 + 1*covterm)**.5