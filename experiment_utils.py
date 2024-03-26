import torch

def dist_to_dirac(samples, theta_true, metrics=["mse", "mmd"], scaled=False):
    dict = {metric: [] for metric in metrics}

    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    for j in range(len(theta_true)):
        samples_coordj = samples[:, j]
        
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