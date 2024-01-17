import torch
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from .prior import UniformPrior, GaussianPrior


class SBIGaussian2d:
    def __init__(self, prior_type, rho=0.8) -> None:
        """2d Gaussian: x ~ N(theta, rho * I).
        SBI task: infer theta from x.
        """
        self.prior_type = prior_type
        self.rho = rho

        self.prior = self.get_prior()

    def get_prior(self):
        if self.prior_type == "uniform":
            return UniformPrior()
        elif self.prior_type == "gaussian":
            return GaussianPrior()
        else:
            raise NotImplementedError

    def simulator(self, theta):
        cov = torch.FloatTensor([[1, self.rho], [self.rho, 1]])
        samples_x = MultivariateNormal(loc=theta, covariance_matrix=cov).sample()
        return samples_x

    def true_posterior(self, x_obs, return_loc_cov=False):
        cov = torch.FloatTensor([[1, self.rho], [self.rho, 1]])
        if self.prior_type == "uniform":
            return MultivariateNormal(loc=x_obs, covariance_matrix=cov, validate_args=False)
        elif self.prior_type == "gaussian":
            cov_prior = self.prior.prior.covariance_matrix
            cov_posterior = torch.linalg.inv(
                torch.linalg.inv(cov) + torch.linalg.inv(cov_prior)
            )
            loc_posterior = cov_posterior @ (
                torch.linalg.inv(cov) @ x_obs
                + torch.linalg.inv(cov_prior) @ self.prior.prior.loc
            )
            if not return_loc_cov:
                return MultivariateNormal(
                    loc=loc_posterior, covariance_matrix=cov_posterior, validate_args=False
                )
            return loc_posterior, cov_posterior
