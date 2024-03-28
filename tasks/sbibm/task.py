import torch
import os

from tqdm import tqdm


class Task:
    # general task class for SBIBM
    def __init__(self, name, prior_params, save_path="data/"):
        self.name = name
        self.prior_params = prior_params
        self.save_path = save_path + name + "/"

    def prior(self):
        return NotImplementedError

    def simulator(self, theta, n_obs=1):
        return NotImplementedError

    def _posterior_sampler(self, x_star, theta_star, n_obs, num_sample, **kwargs):
        return NotImplementedError

    def sample_reference_posterior(
        self, x_star, theta_star, n_obs=1, num_samples=1000, **kwargs
    ):
        # ensure x_star contains n_obs observations
        if x_star.shape[0] > n_obs:
            x_star = x_star[:n_obs]
        assert x_star.shape[0] == n_obs

        return self._posterior_sampler(
            x_star=x_star,
            theta_star=theta_star,
            n_obs=n_obs,
            num_samples=num_samples,
            **kwargs,
        )

    def generate_training_data(self, n_simulations, save=True, n_obs=1, **kwargs):
        print(
            "Generating training data for",
            n_simulations,
            "simulations and",
            n_obs,
            "observations.",
        )
        # prior samples
        prior = self.prior()
        theta_train = prior.sample((n_simulations,))

        # simulator samples
        x_train = []
        for theta_i in tqdm(theta_train):
            x_i = self.simulator(theta=theta_i, n_obs=n_obs, **kwargs)
            x_train.append(x_i)
        x_train = torch.stack(x_train)
        if n_obs == 1:
            x_train = x_train[:, 0, :]

        dataset_train = {"theta": theta_train, "x": x_train}
        if save:
            filename = f"{self.save_path}dataset_n_train_{n_simulations}.pkl"
            if n_obs > 1:
                filename = (
                    f"{self.save_path}dataset_n_train_{n_simulations}_n_obs_{n_obs}.pkl"
                )
            print("Saving at", filename)
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(dataset_train, filename)

        return dataset_train

    def generate_reference_data(
        self, nb_obs=25, n_repeat=100, save=True, load_theta=False, **kwargs
    ):
        print(
            f"Generating reference data for {nb_obs} observations and {n_repeat} repetitions."
        )

        # reference parameters
        filename = f"{self.save_path}theta_true_list.pkl"
        if not load_theta:
            prior = self.prior()
            theta_star = prior.sample((nb_obs,))
            if save:
                print("Saving at", filename)
                os.makedirs(self.save_path, exist_ok=True)
                torch.save(theta_star, filename)
        else:
            theta_star = torch.load(filename)

        # reference observations
        x_star = {}
        for num_obs in range(1, nb_obs + 1):
            theta_true = theta_star[num_obs - 1]
            x_star[num_obs] = self.simulator(theta=theta_true, n_obs=n_repeat, **kwargs)
            if save:
                path = f"{self.save_path}reference_observations/"
                os.makedirs(path, exist_ok=True)
                filename = f"{path}x_obs_{n_repeat}_num_{num_obs}.pkl"
                print("Saving at", filename)
                torch.save(x_star[num_obs], filename)

        return theta_star, x_star

    def generate_reference_posterior_samples(
        self, num_obs, n_obs, num_samples=1000, save=True, **kwargs
    ):
        print(
            "Generating reference posterior samples for num_obs =",
            num_obs,
            "and n_obs =",
            n_obs,
        )

        # reference data for num_obs
        theta_star = self.get_reference_parameters(verbose=False)[num_obs - 1]
        x_star = self.get_reference_observation(num_obs=num_obs, verbose=False)

        # sample from the posterior
        samples = self.sample_reference_posterior(
            x_star=x_star,
            theta_star=theta_star,
            n_obs=n_obs,
            num_samples=num_samples,
            **kwargs,
        )
        if save:
            path = f"{self.save_path}reference_posterior_samples/"
            os.makedirs(path, exist_ok=True)
            filename = f"{path}true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
            print("Saving at", filename)
            torch.save(samples, filename)
        return samples

    def get_training_data(self, n_simulations, n_obs=1):
        filename = f"{self.save_path}dataset_n_train_{n_simulations}.pkl"
        if n_obs > 1:
            filename = (
                f"{self.save_path}dataset_n_train_{n_simulations}_n_obs_{n_obs}.pkl"
            )
        try:
            print(f"Loading training data from {filename}")
            dataset_train = torch.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Training data not found {self.save_path}. Please run `generate_training_data` first."
            )
        return dataset_train

    def get_reference_parameters(self, verbose=True):
        filename = f"{self.save_path}theta_true_list.pkl"
        try:
            if verbose:
                print(f"Loading reference parameters from {filename}")
            theta_star = torch.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Reference parameters not found at {self.save_path}. Please run `generate_reference_data` first."
            )
        return theta_star

    def get_reference_observation(self, num_obs, n_repeat=100, verbose=True):
        filename = (
            f"{self.save_path}reference_observations/x_obs_{n_repeat}_num_{num_obs}.pkl"
        )
        try:
            if verbose:
                print(f"Loading reference observations from {filename}")
            x_star = torch.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Reference observations not found at {self.save_path}. Please run `generate_reference_data` first."
            )
        return x_star

    def get_reference_posterior_samples(self, num_obs, n_obs, verbose=True):
        filename = f"{self.save_path}reference_posterior_samples/true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl"
        try:
            if verbose:
                print(f"Loading reference posterior samples from {filename}")
            samples = torch.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Reference posterior samples not found at {self.save_path}. Please run `generate_reference_posterior_samples` first."
            )
        return samples
