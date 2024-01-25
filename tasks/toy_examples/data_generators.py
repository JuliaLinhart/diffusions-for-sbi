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

        self.simulator_cov = torch.eye(2) * (1 - rho) + rho
        self.simulator_precision = torch.linalg.inv(self.simulator_cov)

    def get_prior(self):
        if self.prior_type == "uniform":
            return UniformPrior()
        elif self.prior_type == "gaussian":
            return GaussianPrior()
        else:
            raise NotImplementedError

    def simulator(self, theta):
        samples_x = MultivariateNormal(loc=theta, covariance_matrix=self.simulator_cov).sample()
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
        
    def true_tall_posterior(self, x_obs):
        """Gets posterior for the case with multiple observations

        Args:
            x_obs: observations to condition on

        Returns:
            Posterior distribution
        """

        N = len(x_obs)

        covariance_matrix = torch.linalg.inv(
            self.prior.prior.precision_matrix
            + N * self.simulator_precision
        )
        loc = covariance_matrix @ (
            N * self.simulator_precision @ torch.mean(x_obs, dim=0).reshape(-1)
            + self.prior.prior.precision_matrix @ self.prior.prior.loc
        )

        posterior = torch.distributions.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior



class Gaussian_Gaussian_mD:
    def __init__(self, dim, rho=0.8, means=None, stds=None) -> None:
        """
        Prior: mD Gaussian: theta ~ N(means, diag(scales)).
        Simulator: mD Gaussian: x ~ N(theta, rho * I_m).
        SBI task: infer theta from x.
        """
        self.rho = rho
        self.dim = dim

        if means is None:
            means = torch.zeros(dim)
        if stds is None:
            stds = torch.ones(dim)
        
        self.prior = torch.distributions.MultivariateNormal(
            loc=means, covariance_matrix=torch.diag_embed(stds.square())
        )
        
        # cov is torch.eye with rho on the off-diagonal
        self.simulator_cov = torch.eye(dim) * (1 - rho) + rho
        self.simulator_precision = torch.linalg.inv(self.simulator_cov)

    def simulator(self, theta):
        samples_x = MultivariateNormal(loc=theta, covariance_matrix=self.simulator_cov).sample()
        return samples_x

    def true_posterior(self, x_obs):
        cov = torch.FloatTensor([[1, self.rho], [self.rho, 1]])

        cov_prior = self.prior.covariance_matrix
        cov_posterior = torch.linalg.inv(
            torch.linalg.inv(cov) + torch.linalg.inv(cov_prior)
        )
        loc_posterior = cov_posterior @ (
            torch.linalg.inv(cov) @ x_obs
            + torch.linalg.inv(cov_prior) @ self.prior.loc
        )

        return MultivariateNormal(
            loc=loc_posterior, covariance_matrix=cov_posterior, validate_args=False
        )

    def true_tall_posterior(self, x_obs):
        """Gets posterior for the case with multiple observations

        Args:
            x_obs: observations to condition on

        Returns:
            Posterior distribution
        """

        N = len(x_obs)

        covariance_matrix = torch.linalg.inv(
            self.prior.precision_matrix
            + N * self.simulator_precision
        )
        loc = covariance_matrix @ (
            N * self.simulator_precision @ torch.mean(x_obs, dim=0).reshape(-1)
            + self.prior.precision_matrix @ self.prior.loc
        )

        posterior = torch.distributions.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior

