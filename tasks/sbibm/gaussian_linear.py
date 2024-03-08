import torch

from tasks.sbibm.task import Task


class GaussianLinear(Task):
    def __init__(
        self,
        prior_params={"loc": 0, "scale": 1},
        simulator_scale=(0.6, 1.4),
        dim=10,
        **kwargs,
    ):
        super().__init__(name="gaussian_linear", prior_params=prior_params, **kwargs)

        self.dim = dim
        self.prior_params["loc"] = torch.tensor(
            [self.prior_params["loc"] for _ in range(self.dim)]
        ).float()
        self.prior_params["precision_matrix"] = torch.inverse(
            self.prior_params["scale"] * torch.eye(self.dim)
        )

        self.simulator_scale = torch.linspace(
            simulator_scale[0], simulator_scale[1], dim
        ).float()
        self.simulator_params = {
            "precision_matrix": torch.inverse(
                self.simulator_scale * torch.eye(self.dim)
            ),
        }

    def prior(self):
        return torch.distributions.MultivariateNormal(
            loc=self.prior_params["loc"],
            precision_matrix=self.prior_params["precision_matrix"],
        )

    def simulator(self, theta, n_obs=1):
        # theta shape must be (10,) for correct sample shape
        if len(theta.shape) > 1:
            theta = theta[0]
        assert theta.shape[0] == 10

        return torch.distributions.MultivariateNormal(
            loc=theta, precision_matrix=self.simulator_params["precision_matrix"]
        ).sample((n_obs,))

    def _posterior_dist(self, x_star, n_obs):
        covariance_matrix = torch.inverse(
            self.prior_params["precision_matrix"]
            + n_obs * self.simulator_params["precision_matrix"]
        )

        loc = torch.matmul(
            covariance_matrix,
            (
                n_obs
                * torch.matmul(
                    self.simulator_params["precision_matrix"],
                    torch.mean(x_star, dim=0).reshape(-1),
                )
                + torch.matmul(
                    self.prior_params["precision_matrix"],
                    self.prior_params["loc"],
                )
            ),
        )

        posterior = torch.distributions.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix
        )

        return posterior

    def _posterior_sampler(self, x_star, theta_star, n_obs, num_samples):
        return self._posterior_dist(x_star, n_obs).sample((num_samples,))

    def sample_reference_posterior(self, x_star, n_obs=1, num_samples=1000, **kwargs):
        return super().sample_reference_posterior(
            x_star=x_star,
            n_obs=n_obs,
            num_samples=num_samples,
            theta_star=None,
            **kwargs,
        )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate data for the Lotka-Volterra example"
    )
    parser.add_argument(
        "--train_data", action="store_true", help="Generate training data"
    )
    parser.add_argument(
        "--reference_data", action="store_true", help="Generate reference data"
    )
    parser.add_argument(
        "--reference_posterior",
        action="store_true",
        help="Generate reference posterior samples",
    )
    parser.add_argument(
        "--save_path", type=str, default="data/", help="Path to save the data"
    )

    args = parser.parse_args()

    gl = GaussianLinear(save_path=args.save_path)
    os.makedirs(gl.save_path, exist_ok=True)

    if args.train_data:
        data = gl.generate_training_data(n_simulations=50_000)
        print("Training data:", data["x"].shape, data["theta"].shape)

    if args.reference_data:
        ref_data = gl.generate_reference_data()
        print(ref_data[0].shape, ref_data[1][1].shape)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                samples = gl.generate_reference_posterior_samples(
                    num_obs=num_obs, n_obs=n_obs, num_samples=1_000
                )
                print(samples.shape)

    # # simulate one check
    # theta = gl.prior().sample((1,))
    # x = gl.simulator(theta, n_obs=1)
    # print(x.shape, theta.shape)

    # x_star = torch.load('/data/parietal/store3/work/jlinhart/git_repos/diffusions-for-sbi/results/sbibm/gaussian_linear/x_obs_100_num_1_new.pkl')
    # samples_sbibm = torch.load('/data/parietal/store3/work/jlinhart/git_repos/diffusions-for-sbi/results/sbibm/gaussian_linear/reference_posterior_samples/true_posterior_samples_num_1_n_obs_8.pkl')
    # samples_jl = gl.sample_reference_posterior(x_star=x_star, n_obs=8, num_samples=1000)

    # import matplotlib.pyplot as plt
    # plt.scatter(samples_sbibm[:,1], samples_sbibm[:,2], label='sbibm')
    # plt.scatter(samples_jl[:,1], samples_jl[:,2], label='jl')
    # plt.savefig('gaussian_linear_comparison.png')
    # plt.clf()
