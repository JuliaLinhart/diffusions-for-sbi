import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .prior import GaussianPrior, UniformPrior


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
        samples_x = MultivariateNormal(
            loc=theta, covariance_matrix=self.simulator_cov
        ).sample()
        return samples_x

    def true_posterior(self, x_obs):
        cov = self.simulator_cov

        cov_prior = self.prior.covariance_matrix
        cov_posterior = torch.linalg.inv(
            torch.linalg.inv(cov) + torch.linalg.inv(cov_prior)
        )
        loc_posterior = (
            cov_posterior
            @ (
                (torch.linalg.inv(cov) @ x_obs.mT).mT
                + torch.linalg.inv(cov_prior) @ self.prior.loc
            ).mT
        ).mT

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
            self.prior.precision_matrix + N * self.simulator_precision
        )
        loc = covariance_matrix @ (
            N * self.simulator_precision @ torch.mean(x_obs, dim=0).reshape(-1)
            + self.prior.precision_matrix @ self.prior.loc
        )

        posterior = torch.distributions.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior


class Gaussian_MixtGaussian_mD:
    def __init__(self, dim, rho_min=0.6, rho_max=1.4, device="cuda:0") -> None:
        """
        Prior: mD Gaussian: theta ~ N(means, diag(scales)).
        Simulator: mD Gaussian: x ~ N(theta, rho * I_m).
        SBI task: infer theta from x.
        """
        self.dim = dim
        self.simulator_base_std = (
            torch.linspace(rho_min, rho_max, dim).to(device) ** 0.5
        )
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dim).to(device), covariance_matrix=torch.eye(dim).to(device)
        )
        self.device = device

    def simulator(self, theta):
        cp_dist = MultivariateNormal(
            loc=theta[None].repeat(2, 1),
            scale_tril=torch.stack(
                (
                    torch.diag(self.simulator_base_std * (2.25) ** 0.5),
                    torch.diag(self.simulator_base_std / 3),
                ),
                dim=0,
            ),
        )
        samples_x = torch.distributions.MixtureSameFamily(
            component_distribution=cp_dist,
            mixture_distribution=torch.distributions.Categorical(
                probs=torch.ones(2).to(self.device) / 2
            ),
        ).sample((1,))[0]
        return samples_x

    def true_posterior(self, x_obs):
        equivalent_post_diag_cov_1 = 1 / (1 / (2.25 * self.simulator_base_std**2) + 1)
        equivalent_post_diag_cov_2 = 1 / (9 / self.simulator_base_std**2 + 1)
        log_weights = (
            torch.distributions.Normal(
                loc=torch.zeros_like(x_obs),
                scale=(2.25**0.5) * self.simulator_base_std + 1,
            )
            .log_prob(x_obs)
            .sum(dim=-1),
            torch.distributions.Normal(
                loc=torch.zeros_like(x_obs), scale=(1 / 3) * self.simulator_base_std + 1
            )
            .log_prob(x_obs)
            .sum(dim=-1),
        )

        base_comp = MultivariateNormal(
            loc=torch.stack(
                (
                    equivalent_post_diag_cov_1
                    * x_obs
                    / ((self.simulator_base_std**2) * 2.25),
                    equivalent_post_diag_cov_2
                    * x_obs
                    / ((self.simulator_base_std**2) * (1 / 9)),
                ),
                dim=0,
            ),
            scale_tril=torch.stack(
                (
                    torch.diag(equivalent_post_diag_cov_1**0.5),
                    torch.diag(equivalent_post_diag_cov_2**0.5),
                ),
                dim=0,
            ),
        )
        return torch.distributions.MixtureSameFamily(
            component_distribution=base_comp,
            mixture_distribution=torch.distributions.Categorical(
                logits=torch.stack(log_weights)
            ),
        )

    def diffused_posterior(self, x_obs, alpha_t):
        posterior_0 = self.true_posterior(x_obs)
        cov = posterior_0.component_distribution.covariance_matrix
        return torch.distributions.MixtureSameFamily(
            component_distribution=MultivariateNormal(
                loc=posterior_0.component_distribution.loc * (alpha_t**0.5),
                covariance_matrix=cov * alpha_t
                + (1 - alpha_t) * torch.eye(cov.shape[-1])[None].to(cov.device),
            ),
            mixture_distribution=posterior_0.mixture_distribution,
        )


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
        samples_x = MultivariateNormal(
            loc=theta, covariance_matrix=self.simulator_cov
        ).sample()
        return samples_x

    def true_posterior(self, x_obs, return_loc_cov=False):
        cov = torch.FloatTensor([[1, self.rho], [self.rho, 1]])
        if self.prior_type == "uniform":
            return MultivariateNormal(
                loc=x_obs, covariance_matrix=cov, validate_args=False
            )
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
                    loc=loc_posterior,
                    covariance_matrix=cov_posterior,
                    validate_args=False,
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
            self.prior.prior.precision_matrix + N * self.simulator_precision
        )
        loc = covariance_matrix @ (
            N * self.simulator_precision @ torch.mean(x_obs, dim=0).reshape(-1)
            + self.prior.prior.precision_matrix @ self.prior.prior.loc
        )

        posterior = torch.distributions.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior
