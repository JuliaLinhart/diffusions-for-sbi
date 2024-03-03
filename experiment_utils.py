from typing import List

import torch


def dist_to_dirac(
    samples: torch.Tensor,
    theta_true: torch.Tensor,
    metrics: List[str] = ["mse", "mmd"],
    scaled: bool = True,
) -> dict:
    r"""
    Compute distance to true theta.

    Args:
        samples (torch.Tensor): posterior samples (n_samples, dim)
        theta_true (torch.Tensor): true parameter used to simulate the reference observations (*,dim)
        metrics (List[str]): name of the metrics to compute
        scaled (bool): whether to scale the mmd by the standard deviation

    Returns:
        (dict): dictionary of the computed metrics.
            - Keys are the names of the metrics.
            - Values are the computed metrics.
    """

    # Initialize dictionary
    dict = {metric: [] for metric in metrics}

    # Ensure theta_true is a 1D tensor
    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    # Compute metrics
    # Loop over the dimensions of theta
    for j in range(len(theta_true)):
        # extract the j-th coordinate of the samples
        samples_coordj = samples[:, j]
        if "mse" in metrics:
            # compute the mean squared error
            dict["mse"].append((samples_coordj - theta_true[j]).square().mean())
        if "mmd" in metrics:
            # compute the maximum mean discrepancy
            sd = torch.sqrt(samples_coordj.var())
            if scaled:
                mmd = (
                    samples_coordj.var()
                    + (samples_coordj.mean() - theta_true[j]).square()
                ) / sd
            else:
                mmd = (
                    samples_coordj.var()
                    + (samples_coordj.mean() - theta_true[j]).square()
                )
            dict["mmd"].append(mmd)

    # Compute the mean metric over the dimensions of theta
    for metric in metrics:
        dict[metric] = torch.stack(dict[metric]).mean()

    return dict


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.

    Args:
        matrix (torch.Tensor): matrix to be powered
        p (float): power to raise the matrix to

    Returns:
        (torch.Tensor): powered matrix
    """
    # Compute the eigen decomposition
    L, V = torch.linalg.eig(matrix)
    # Ensure the eigenvalues and eigenvectors are real
    L = L.real
    V = V.real
    # Raise the eigenvalues to the power p and reconstruct the powered matrix
    return V @ torch.diag_embed(L.pow(p)) @ torch.linalg.inv(V)


def gaussian_wasserstein(
    ref_mu: torch.Tensor, ref_cov: torch.Tensor, X2: torch.Tensor
) -> torch.Tensor:
    r"""
    Compute the Wasserstein distance between two multivariate Gaussian distributions.
    The closed-form writes:

        W2 = ||μ1 - μ2||^2 + tr(Σ1 + Σ2 - 2(Σ1^1/2 Σ2 Σ1^1/2)^1/2)

    Args:
        ref_mu (torch.Tensor): mean of the reference distribution
        ref_cov (torch.Tensor): covariance matrix of the reference distribution
        X2 (torch.Tensor): samples from the second distribution

    Returns:
        (torch.Tensor): Gaussian Wasserstein distance
    """
    # Compute the mean and covariance matrix of the second distribution
    mean2 = torch.mean(X2, dim=1)
    cov2 = torch.func.vmap(lambda x: torch.cov(x.mT))(X2)
    # Square root of the covariance matrix of the reference distribution
    sqrtcov1 = _matrix_pow(ref_cov, 0.5)
    # Compute the covariance term of the Wasserstein distance
    covterm = torch.func.vmap(torch.trace)(
        ref_cov + cov2 - 2 * _matrix_pow(sqrtcov1 @ cov2 @ sqrtcov1, 0.5)
    )
    return (1 * torch.linalg.norm(ref_mu - mean2, dim=-1) ** 2 + 1 * covterm) ** 0.5
