import torch
from sbibm.tasks.gaussian_linear.task import GaussianLinear
from pyro import distributions as pdist
import matplotlib.pyplot as plt
import seaborn as sns


class GaussianLinear_multiobs(GaussianLinear):
    def __init__(
        self,
        dim=10,
        prior_scale=1,
        simulator_scale=torch.linspace(0.6, 1.4, 10)):

        super().__init__(dim, prior_scale, simulator_scale)

    def _get_reference_posterior_multiobs(self, x_obs):
        """Gets posterior for the case with multiple observations

        Args:
            x_obs: list of observations to condition on

        Returns:
            Posterior distribution
        """

        N = len(x_obs)

        covariance_matrix = torch.inverse(
            self.prior_params["precision_matrix"]
            + N * self.simulator_params["precision_matrix"]
        )
        loc = torch.matmul(
            covariance_matrix,
            (
                N
                * torch.matmul(
                    self.simulator_params["precision_matrix"],
                    torch.mean(x_obs, dim=0).reshape(-1)
                )
                + torch.matmul(
                    self.prior_params["precision_matrix"],
                    self.prior_params["loc"],
                )
            ),
        )

        posterior = pdist.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior
    
    def _sample_reference_posterior_multiobs(
            self,
            num_samples,
            x_obs):
        posterior = self._get_reference_posterior_multiobs(x_obs)
        return posterior.sample((num_samples,))


if __name__ == "__main__":

    torch.manual_seed(42)

    # example following the exact setup from Geffner et al.
    task = GaussianLinear_multiobs(dim=10,
                                  prior_scale=torch.ones(10),
                                  simulator_scale=torch.linspace(0.6, 1.4, 10))
    prior = task.get_prior()
    simulator = task.get_simulator()
    thetas = prior(num_samples=1)

    posterior_samples = {}
    for n_repetitions in [1, 10, 50]:
        xs = simulator(torch.cat(n_repetitions*[thetas]))
        print(xs.shape)
        posterior_samples[n_repetitions] = task._sample_reference_posterior_multiobs(
            num_samples=10_000,
            observation_list=xs)

    fig, ax = plt.subplots(figsize=(8, 6), ncols=1)
    for n_repetitions in [1, 10, 50]:
        ax.axvline(x=thetas[0, 0], c='k', ls='--')
        sns.kdeplot(posterior_samples[n_repetitions].numpy()[:, 0],
                    bw_method=0.5,
                    lw=2.0,
                    ax=ax,
                    label=n_repetitions)
    ax.legend()
    # fig.show()
    fig.savefig('gaussianlinear-multiobs.png')
