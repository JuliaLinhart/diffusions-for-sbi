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

        self.dim_theta = dim
        self.dim_x = dim
        self.prior_params["loc"] = torch.tensor(
            [self.prior_params["loc"] for _ in range(self.dim_theta)]
        ).float()
        self.prior_params["precision_matrix"] = torch.inverse(
            self.prior_params["scale"] * torch.eye(self.dim_theta)
        )

        self.simulator_scale = torch.linspace(
            simulator_scale[0], simulator_scale[1], self.dim_x
        ).float()
        self.simulator_params = {
            "precision_matrix": torch.inverse(
                self.simulator_scale * torch.eye(self.dim_x)
            ),
        }

    def prior(self):
        return torch.distributions.MultivariateNormal(
            loc=self.prior_params["loc"],
            precision_matrix=self.prior_params["precision_matrix"],
            validate_args=False,
        )

    def simulator(self, theta, n_obs=1):
        # theta shape must be (10,) for correct sample shape
        if len(theta.shape) > 1:
            theta = theta[0]
        assert theta.shape[0] == self.dim_theta

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

    def _posterior_sampler(self, x_star, n_obs, num_samples, **kwargs):
        return self._posterior_dist(x_star, n_obs).sample((num_samples,))

    def sample_reference_posterior(self, x_star, n_obs=1, num_samples=1000, **kwargs):
        return super().sample_reference_posterior(
            x_star=x_star,
            n_obs=n_obs,
            num_samples=num_samples,
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

    # # simulator distribution check
    # from sbibm.tasks.gaussian_linear.task import GaussianLinear as GLSBIBM
    # gl_sbibm = GLSBIBM(simulator_scale=torch.linspace(0.6, 1.4, 10), prior_scale=1, dim=10)
    # theta = gl.prior().sample((1,))
    # x_sbibm = [gl_sbibm.get_simulator()(theta) for _ in range(1000)]
    # x_sbibm = torch.concatenate(x_sbibm, axis=0)
    # x_jl = gl.simulator(rng_key, theta, n_obs=1000)
    # print(x_sbibm.shape, x_jl.shape)

    # import matplotlib.pyplot as plt
    # plt.scatter(x_sbibm[:,0], x_sbibm[:,1], label='sbibm')
    # plt.scatter(x_jl[:,0], x_jl[:,1], label='jl')
    # plt.legend()
    # plt.savefig('gl_sim_check.png')
    # plt.clf()

    # # reference posterior check
    # import sbibm
    # gl_sbibm = sbibm.get_task("gaussian_linear")

    # x_star = gl.get_reference_observation(num_obs=2)
    # theta_star = gl.get_reference_parameters()[1]
    # samples_sbibm = gl_sbibm._sample_reference_posterior(num_samples=1000, observation=x_star, num_observation=2)
    # samples_jl = gl.sample_reference_posterior(rng_key=rng_key, x_star=x_star, theta_star=theta_star, n_obs=1, num_samples=1000)
    # # samples_jl_30 = gl.sample_reference_posterior(rng_key=rng_key, x_star=x_star, theta_star=theta_star, n_obs=30, num_samples=1000)

    # print(samples_sbibm.shape, samples_jl.shape)
    # import matplotlib.pyplot as plt
    # plt.scatter(samples_sbibm[:,0], samples_sbibm[:,1], label='sbibm')
    # plt.scatter(samples_jl[:,0], samples_jl[:,1], label='jl')
    # # plt.scatter(samples_jl_30[:,0], samples_jl_30[:,1], label='jl_30')
    # plt.scatter(theta_star[0], theta_star[1], label='theta_star')
    # plt.scatter(x_star[0,0], x_star[0,1], label='x_star')
    # plt.legend()
    # plt.savefig('gl_post_check.png')
    # plt.clf()
