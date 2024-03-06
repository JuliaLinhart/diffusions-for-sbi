import numpy as np
import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
from jax import random
import torch
from tqdm import tqdm
import argparse
import os
from functools import partial


def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u = z[0]
    v = z[1]
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])


def model(
        prior_params={"loc": [-0.125, -3.0, -0.125, -3.0], "scale": [0.50, 0.50, 0.50, 0.50]}, 
        x_obs=None, 
        n_obs=1, 
        summary=None):
    """
    :param int N: number of measurement times in the output
    :param numpy.ndarray x: measured populations with shape (N, 2)
    :param int factor: factor for the number of actual simulated time samples
    """
    # initial population
    z_init = np.array([30.0, 1.0])
    # measurement times
    ts = jnp.arange(0, 20+0.1, 0.1)
    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.LogNormal(loc=jnp.array(prior_params["loc"]),
                       scale=jnp.array(prior_params["scale"]))
    )

    # integrate dz/dt, the result will have shape N x 2
    z = odeint(
        dz_dt,
        z_init,
        ts,
        theta,
        rtol=1e-8,
        atol=1e-8,
        mxstep=1000
    )

    if summary is None:
        z_sub = z
        sigma_i = 0.001
    elif summary == 'subsample':
        z_sub = z[::21, :].T.reshape(-1)
        sigma_i = 0.1

    for i in range(n_obs):
        if x_obs is None:
            numpyro.sample(
                f"obs_{i}",
                dist.LogNormal(
                    jnp.log(jnp.clip(z_sub, 1e-10, 10000)),
                    scale=sigma_i
                ) # noqa
            )
        else:
            numpyro.sample(
                f"obs_{i}",
                dist.LogNormal(
                    jnp.log(jnp.clip(z_sub, 1e-10, 10000)),
                    scale=sigma_i
                ),
                obs=x_obs[i]
            )


class LotkaVolterra():
    def __init__(
            self, 
            prior_params={"loc": [-0.125, -3.0, -0.125, -3.0],
                                "scale": [0.50, 0.50, 0.50, 0.50]},
            summary="subsample",
            torch_version=False,
            save_path='data/lotka_volterra/',
        ):
        self.prior_params = prior_params
        self.model = partial(model, prior_params=prior_params)
        self.summary = summary
        self.torch_version = torch_version
        self.save_path = save_path
            

    def prior(self, torch_version=None):
        if torch_version is None:
            torch_version = self.torch_version
        if torch_version:
            return torch.distributions.LogNormal(
                loc=torch.tensor(self.prior_params["loc"]),
                scale=torch.tensor(self.prior_params["scale"])
            )
        else:
            return dist.LogNormal(
                loc=jnp.array(self.prior_params["loc"]),
                scale=jnp.array(self.prior_params["scale"])
            )
    
    def simulator(self, rng_key, theta, n_obs=1, torch_version=None):

        if torch_version is None:
            torch_version = self.torch_version

        # ensure theta is a jax array
        if isinstance(theta, torch.Tensor):
            theta = theta.numpy()
        theta = jnp.array(theta)

        # simulate
        predictive = Predictive(
            condition(self.model, {"theta": theta}),
            num_samples=1)
        rng_key, subkey = random.split(rng_key)
        data = predictive(
            subkey,
            n_obs=n_obs,
            summary=self.summary
        )
        # concatenate the observations
        x = jnp.concatenate([data[f'obs_{i}'] for i in range(n_obs)])
        
        if torch_version:
            x = torch.from_numpy(np.asarray(x)).float()
        return x # shape (n_obs, dim_x)
    
    def sample_reference_posterior(self, rng_key, x_star, theta_star, n_obs, num_samples=1000):

        # ensure theta_star is a jax array
        if isinstance(theta_star, torch.Tensor):
            theta_star = theta_star.numpy()
        theta_star = jnp.array(theta_star)

        # ensure x_star is a jax array
        if isinstance(x_star, torch.Tensor):
            x_star = x_star.numpy()
        x_star = jnp.array(x_star)

        # ensure x_star contains n_obs observations
        if x_star.shape[0] > n_obs:
            x_star = x_star[:n_obs]
        assert(x_star.shape[0] == n_obs)

        # sample from the posterior
        kernel = NUTS(
            self.model,
            init_strategy=init_to_value(
                None, values={'theta': theta_star}))
        mcmc = MCMC(
            kernel,
            num_warmup=100,
            num_samples=num_samples,
            num_chains=1
        )
        rng_key, subkey = random.split(rng_key)
        mcmc.run(
            rng_key=subkey,
            x_obs=x_star,
            n_obs=n_obs,
            summary='subsample')
        
        samples = mcmc.get_samples()['theta']

        if self.torch_version:
            samples = np.asarray(samples)
            samples = torch.from_numpy(samples).float()
        return samples

    def generate_training_data(self, rng_key, n_simulations, save=True):
        print('Generating training data')
        # prior
        prior = self.prior(torch_version=False)
        rng_key, subkey = random.split(rng_key)
        theta_train = prior.sample(subkey, (n_simulations,))

        # simulator
        x_train = []
        for theta_i in tqdm(theta_train):
            x_i = self.simulator(rng_key=rng_key, theta=theta_i, n_obs=1, torch_version=False)
            x_train.append(x_i)
        x_train = jnp.stack(x_train)

        if self.torch_version:
            x_train = torch.from_numpy(np.asarray(x_train)).float()
            theta_train = torch.tensor(np.asarray(theta_train)).float()

        dataset_train = {"theta": theta_train, "x": x_train}
        if save:
            filename = f'{self.save_path}dataset_n_train_{n_simulations}.pkl'
            print('Saving at', filename)
            torch.save(dataset_train, filename)
        
        return dataset_train

    def generate_reference_data(self, rng_key, nb_obs=25, n_repeat=100, save=True, load_theta=False):
        version = 'torch' if self.torch_version else 'jax'
        print(f'Generating reference data ({version} version) for {nb_obs} observations and {n_repeat} repetitions.')
        
        # reference parameters
        filename = f'{self.save_path}theta_true_list.pkl'
        if not load_theta:
            rng_key, subkey = random.split(rng_key)
            prior = self.prior(torch_version=False)
            theta_star = prior.sample(subkey, (nb_obs,))
            if self.torch_version:
                theta_star = torch.tensor(np.asarray(theta_star)).float()
            if save:
                print('Saving at', filename)
                torch.save(theta_star, filename)
        else:
            theta_star = torch.load(filename)
        
        # reference observations
        x_star = {}
        for num_obs in range(1, nb_obs+1):
            theta_true = theta_star[num_obs-1]
            x_star[num_obs] = self.simulator(rng_key=rng_key, theta=theta_true, n_obs=n_repeat)
            if save:
                path = f'{self.save_path}reference_observations/'
                os.makedirs(path, exist_ok=True)
                filename = f'{path}x_obs_{n_repeat}_num_{num_obs}.pkl'
                print('Saving at', filename)
                torch.save(x_star[num_obs], filename)
        
        return theta_star, x_star

    def get_reference_parameters(self):
        filename = f'{self.save_path}theta_true_list.pkl'
        theta_star = torch.load(filename)
        if self.torch_version and not isinstance(theta_star, torch.Tensor):
            theta_star = torch.tensor(np.asarray(theta_star)).float()
        return theta_star
    
    def get_reference_observation(self, num_obs, n_repeat=100):
        filename = f'{self.save_path}reference_observations/x_obs_{n_repeat}_num_{num_obs}.pkl'
        x_star = torch.load(filename)
        if self.torch_version and not isinstance(x_star, torch.Tensor):
            x_star = torch.tensor(np.asarray(x_star)).float()
        return x_star
    
    def generate_reference_posterior_samples(self, rng_key, num_obs, n_obs, num_samples=1000, save=True):
        print('Generating reference posterior samples for num_obs =', num_obs, 'and n_obs =', n_obs)

        # reference data for num_obs
        theta_star = self.get_reference_parameters()[num_obs-1]
        x_star = self.get_reference_observation(num_obs)

        # sample from the posterior
        samples = self.sample_reference_posterior(rng_key, x_star, theta_star, n_obs, num_samples=num_samples)
        if save:
            path = f'{self.save_path}reference_posterior_samples/'
            os.makedirs(path, exist_ok=True)
            filename = f'{path}true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl'
            print('Saving at', filename)
            torch.save(samples, filename)
        return samples
    
    def get_reference_posterior_samples(self, num_obs, n_obs):
        filename = f'{self.save_path}reference_posterior_samples/true_posterior_samples_num_{num_obs}_n_obs_{n_obs}.pkl'
        samples = torch.load(filename)

        if self.torch_version and not isinstance(samples, torch.Tensor):
            samples = torch.tensor(np.asarray(samples)).float()
            # shuffle the samples to get rid of the MCMC chain structure
            samples = samples[torch.randperm(samples.shape[0])]
        else:
=            samples = samples[np.random.permutation(samples.shape[0])]
        return samples


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description='Generate data for the Lotka-Volterra example'
    )
    parser.add_argument('--train_data', action='store_true', help='Generate training data')
    parser.add_argument('--reference_data', action='store_true', help='Generate reference data')
    parser.add_argument('--reference_posterior', action='store_true', help='Generate reference posterior samples')
    parser.add_argument('--save_path', type=str, default='data/lotka_volterra/', help='Path to save the data')

    args = parser.parse_args()

    rng_key = random.PRNGKey(1)

    lv = LotkaVolterra(torch_version=True, save_path=args.save_path)
    os.makedirs(lv.save_path, exist_ok=True)

    if args.train_data:
        lv.generate_training_data(rng_key, n_simulations=50_000)

    if args.reference_data:
        lv.generate_reference_data(rng_key)

    if args.reference_posterior:
        num_obs_list = range(1, 26)
        n_obs_list = [1, 8, 14, 22, 30]
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                lv.generate_reference_posterior_samples(rng_key, num_obs, n_obs, num_samples=1000)
