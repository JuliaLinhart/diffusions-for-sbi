import math
import torch
import torch.nn as nn

from torch import Tensor, Size
from torch.distributions import Distribution
from tqdm import tqdm
from typing import *

from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.nn import MLP
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast
from torch.autograd.functional import jacobian
from functools import partialmethod, partial
from torch.func import jacrev, vmap



def assure_positive_definitness(m):
    U, S, Vh = torch.linalg.svd(.5 * (m + m.mT), full_matrices=False)
    return U @ torch.diag_embed(S.clip(1e-10, 1e10)) @ Vh


class NSE(nn.Module):
    r"""Creates a neural score estimation (NSE) network.

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        freqs: The number of time embedding frequencies.
        build_net: The network constructor. It takes the
            number of input and output features as positional arguments.
        embedding_nn_theta: The embedding network for the parameters :math:`\theta`.
        embedding_nn_x: The embedding network for the observations :math:`x`.
        kwargs: Keyword arguments passed to the network constructor `build_net`.
    """

    def __init__(
            self,
            theta_dim: int,
            x_dim: int,
            freqs: int = 3,
            build_net: Callable[[int, int], nn.Module] = MLP,
            embedding_nn_theta: nn.Module = nn.Identity(),
            embedding_nn_x: nn.Module = nn.Identity(),
            **kwargs,
    ):
        super().__init__()

        self.embedding_nn_theta = embedding_nn_theta
        self.embedding_nn_x = embedding_nn_x
        self.theta_emb_dim, self.x_emb_dim = self.get_theta_x_embedding_dim(
            theta_dim, x_dim
        )

        self.net = build_net(
            self.theta_emb_dim + self.x_emb_dim + 2 * freqs, theta_dim, **kwargs
        )

        self.register_buffer("freqs", torch.arange(1, freqs + 1) * math.pi)
        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def get_theta_x_embedding_dim(self, theta_dim, x_dim) -> int:
        r"""Returns the dimensionality of the embeddings for :math:`\theta` and :math:`x`."""
        theta, x = torch.ones((1, theta_dim)), torch.ones((1, x_dim))
        return (
            self.embedding_nn_theta(theta).shape[-1],
            self.embedding_nn_x(x).shape[-1],
        )

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            t: The time :math:`t`, with shape :math:`(*,).`

        Returns:
            The estimated noise :math:`\epsilon_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """

        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        theta = self.embedding_nn_theta(theta)
        x = self.embedding_nn_x(x)

        theta, x, t = broadcast(theta, x, t, ignore=1)

        return self.net(torch.cat((theta, x, t), dim=-1))

    # The following function define the VP SDE with linear noise schedule beta(t):
    # dtheta = f(t) theta dt + g(t) dW = -0.5 * beta(t) theta dt + sqrt(beta(t)) dW

    def score(self, theta, x, t):
        return -self(theta, x, t) / self.sigma(t)

    def beta(self, t: Tensor) -> Tensor:
        r"""Linear noise schedule of the VP SDE:
        .. math:: \beta(t) = 32 t .
        """
        return 32 * t

    def f(self, t: Tensor) -> Tensor:
        """Drift of the VP SDE:
        .. math:: f(t) = -0.5 * \beta(t) .
        """
        return -0.5 * self.beta(t)

    def g(self, t: Tensor) -> Tensor:
        """
        .. math:: g(t) = \sqrt{\beta(t)} .
        """
        return torch.sqrt(self.beta(t))

    def alpha(self, t: Tensor) -> Tensor:
        r"""Mean of the transition kernel of the VP SDE:
        .. math: `alpha(t) = \exp ( -0.5 \int_0^t \beta(s)ds)`.
        """
        return torch.exp(-16 * t**2)

    def sigma(self, t: Tensor) -> Tensor:
        r"""Standard deviation of the transition kernel of the VP SDE:
        .. math:: \sigma^2(t) = 1 - \exp( - \int_0^t \beta(s)ds) + C
        where C is such that :math: `\sigma^2(1) = 1, \sigma^2(0)  = \epsilon \approx 1e-4`.
        """
        return torch.sqrt(1 - self.alpha(t) + math.exp(-16))

    def bridge_mean(self, alpha_t: Tensor, alpha_t_1: Tensor, theta_t: Tensor, theta_0: Tensor, bridge_std: float) -> Tensor:
        est_noise = (theta_t - (alpha_t**.5) * theta_0) / ((1 - alpha_t)**.5)
        return (alpha_t_1**.5)*theta_0 + ((1 - alpha_t_1 - bridge_std**2)**.5) * est_noise

    def ode(self, theta: Tensor, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        return self.f(t) * theta + self.g(t) ** 2 / 2 * self(
            theta, x, t, **kwargs
        ) / self.sigma(t)

    def flow(self, x: Tensor, **kwargs) -> Distribution:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            kwargs: additional args for the forward method.

        Returns:
            The normalizing flow :math:`p_\phi(\theta | x)` induced by the
            probability flow ODE.
        """

        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=lambda t, theta: self.ode(theta, x, t, **kwargs),
                t0=x.new_tensor(0.0),
                t1=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1]),
        )

    def ddim(
            self, shape: Size, x: Tensor, steps: int = 64, verbose: bool = False, eta: float = 1., **kwargs
    ):
        if len(x.shape) == 1:
            score_fun = self.score
        else:
            score_fun = partial(self.factorized_score, **kwargs)
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            theta = self.ddim_step(theta, x, t, score_fun, dt, eta)
        return theta

    def ddim_step(self, theta, x, t, score_fun, dt, eta, **kwargs):
        alpha_t = self.alpha(t)
        alpha_t_1 = self.alpha(t - dt)
        bridge_std = eta * ((((1 - alpha_t_1) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_1)) ** .5)
        score = score_fun(theta, x, t).detach()
        pred_theta_0 = self.mean_pred(theta=theta, score=score, alpha_t=alpha_t)
        theta_mean = self.bridge_mean(alpha_t=alpha_t,
                                      alpha_t_1=alpha_t_1,
                                      theta_0=pred_theta_0,
                                      theta_t=theta,
                                      bridge_std=bridge_std)
        # print(theta_mean.mean(axis=0))
        theta = theta_mean + torch.randn_like(theta_mean) * bridge_std
        return theta

    def langevin_corrector(self, theta, x, t, score_fun, n_steps, r, **kwargs):
        alpha_t = self.alpha(t)
        for i in range(n_steps):
            z = torch.randn_like(theta)
            g = score_fun(theta, x, t).detach()
            # eps = 2*alpha_t*(r*torch.linalg.norm(z, axis=-1).mean(axis=0)/torch.linalg.norm(g, axis=-1).mean(axis=0))**2
            eps = (
                    r
                    * (self.alpha(t)**.5)
                    * min(self.sigma(t) ** 2, 1 / g.square().mean())
            )
            theta = theta + eps*g + ((2*eps)**.5)*z
        return theta

    def predictor_corrector(self,
                            shape: Size,
                            x: Tensor,
                            steps: int = 64,
                            verbose: bool = False,
                            predictor_type='ddim',
                            corrector_type='langevin',
                            **kwargs
    ):
        if len(x.shape) == 1:
            score_fun = self.score
        else:
            score_fun = partial(self.factorized_score, **kwargs)

        if predictor_type == 'ddim':
            predictor_fun = partial(self.ddim_step, **kwargs)
        elif predictor_type == 'id':
            predictor_fun = lambda theta, x, t, score_fun, dt: theta
        else:
            raise NotImplemented("")
        if corrector_type == 'langevin':
            corrector_fun = partial(self.langevin_corrector, **kwargs)
        elif corrector_type == 'id':
            corrector_fun = lambda theta, x, t, score_fun: theta
        else:
            raise NotImplemented("")
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            theta_pred = predictor_fun(theta=theta, x=x, t=t, score_fun=score_fun, dt=dt)
            theta = corrector_fun(theta=theta_pred, x=x, t=t-dt, score_fun=score_fun)
        return theta

    def euler(
            self, shape: Size, x: Tensor, steps: int = 64, verbose: bool = False, **kwargs
    ):
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            z = torch.randn_like(theta)

            score = self(theta, x, t, **kwargs) / (-self.sigma(t))

            drift = self.f(t) * theta - self.g(t) ** 2 * score
            diffusion = self.g(t)

            theta = theta + drift * (-dt) + diffusion * z * dt**0.5

        return theta

    def annealed_langevin(
            self,
            shape: Size,
            x: Tensor,
            steps: int = 64,
            lsteps: int = 1000,
            tau: float = 1,
            verbose: bool = False,
            **kwargs,
    ):
        time = torch.linspace(1, 0, steps + 1).to(x)

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            for _ in range(lsteps):
                z = torch.randn_like(theta)
                score = self(theta, x, t, **kwargs) / -self.sigma(t)
                # delta = tau * self.alpha(t) / score.square().mean()
                delta = (
                        tau
                        * (self.alpha(t)**.5)
                        * min(self.sigma(t) ** 2, 1 / score.square().mean())
                )
                theta = theta + delta * score + torch.sqrt(2 * delta) * z

        return theta

    def mean_pred(self, theta: Tensor, score: Tensor, alpha_t: Tensor, **kwargs) -> Tensor:
        '''
        Parameters
        ----------
        theta
        x
        t
        kwargs

        Returns
        -------

        '''
        upsilon = 1 - alpha_t
        mean = (alpha_t ** (-.5)) * (theta + upsilon*score)
        return mean

    def gaussian_approximation(self, x: Tensor, t: Tensor, theta: Tensor, **kwargs) -> Tuple[Tensor]:
        '''
        Gaussian approximation from https://arxiv.org/pdf/2310.06721.pdf
        Parameters
        ----------
        x: Conditioning variable for the score network (n_samples_theta, n_samples_x, dim)
        t: diffusion "time": (1,)
        theta: Current state of the diffusion process (n_samples_theta, n_samples_x, dim)

        Returns mean (n_samples_theta, n_samples_x, dim) and covariance
        -------
        '''
        alpha_t = self.alpha(t)
        upsilon = 1 - alpha_t
        def mean_to_jac(theta, x):
            score = self.score(theta, x, t)
            mu = self.mean_pred(theta=theta, score=score, alpha_t=alpha_t, **kwargs)
            return mu, (mu, score)

        grad_mean, out = vmap(vmap(jacrev(mean_to_jac, has_aux=True)))(theta, x)
        mean, score = out
        return mean, (upsilon / (alpha_t ** .5))*grad_mean, score

    def log_L(self,
              means_posterior: Tensor,
              covar_posteriors: Tensor,
              mean_prior: Tensor,
              covar_prior: Tensor):
        '''
        Calculates all the factors dependent of theta of log L_theta as defined in (slack document for now...) Following http://www.lucamartino.altervista.org/2003-003.pdf
        Parameters
        ----------
        pred_means
        pred_covs

        Returns
        -------

        '''
        def from_canonical_to_sufficient(mean, covar):
            lda = torch.linalg.inv(covar)
            eta = (lda @ mean[..., None])[..., 0]
            return lda, eta, -.5 * (-torch.linalg.slogdet(lda).logabsdet + (mean[..., None].mT @ lda @ mean[..., None])[...,0, 0])

        n_observations = means_posterior.shape[-2]
        lambdas_posterior, etas_posterior, zetas_posterior = from_canonical_to_sufficient(means_posterior, covar_posteriors)
        lda_prior, eta_prior, zeta_prior = from_canonical_to_sufficient(mean_prior, covar_prior)

        sum_zetas = zetas_posterior.sum(axis=1) + (1 - n_observations)*zeta_prior

        final_gaussian_etas = (1 - n_observations)*eta_prior + etas_posterior.sum(axis=1) 
        final_gaussian_ldas = (1 - n_observations)*lda_prior + lambdas_posterior.sum(axis=1)
        final_gaussian_zeta = -.5 * (-torch.linalg.slogdet(final_gaussian_ldas).logabsdet
                                     + (final_gaussian_etas[..., None].mT @ torch.linalg.inv(final_gaussian_ldas) @final_gaussian_etas[..., None])[..., 0, 0])
        return sum_zetas - final_gaussian_zeta

    def factorized_score(self, theta, x, t, prior_score_fun, corrector_lda=0, **kwargs):
        # Defining stuff
        n_observations = x.shape[0]
        n_samples = theta.shape[0]
        alpha_t = self.alpha(t)
        upsilon = 1 - alpha_t
        theta_ = theta.clone().requires_grad_(True)

        # Calculating m, Sigma and scores for the posteriors
        predicted_mean, predicted_covar, scores = self.gaussian_approximation(theta=theta_[:, None].repeat(1, n_observations, 1),
                                                                              x=x[None, :].repeat(n_samples, 1, 1),
                                                                              t=t)

        # Calculating m, Sigma and score of the prior
        def pred_mean_prior(theta):
            prior_score = prior_score_fun(theta[None], t)[0]
            prior_predicted_mean = self.mean_pred(theta, prior_score, alpha_t)
            return prior_predicted_mean, (prior_predicted_mean, prior_score)

        grad_prior_predicted_mean, out = vmap(jacrev(pred_mean_prior, has_aux=True))(theta_)
        prior_predicted_mean, prior_score = out
        prior_predicted_covar = (upsilon / (alpha_t ** .5)) * grad_prior_predicted_mean
        prior_predicted_covar = prior_predicted_covar
        predicted_covar = assure_positive_definitness(predicted_covar.detach())
        prior_predicted_covar = assure_positive_definitness(prior_predicted_covar.detach())
        # Calculating correction term
        log_L = self.log_L(predicted_mean,
                           predicted_covar,
                           prior_predicted_mean,
                           prior_predicted_covar)

        log_L.sum().backward()
        langevin_grad = (1 - n_observations) * prior_score + scores.sum(axis=1)
        correction = theta_.grad * (n_observations > 1)
        #print(torch.linalg.norm(langevin_grad + correction, axis=-1).mean() / torch.linalg.norm(correction, axis=-1).mean() )
        aggregated_score = langevin_grad + corrector_lda*correction
        theta_.detach()
        # real_score = self.score(theta, x, t)
        # res = torch.linalg.norm(aggregated_score - real_score, axis=-1)
        # print((res/torch.linalg.norm(real_score, axis=-1)).max().item())
        return aggregated_score


class NSELoss(nn.Module):
    r"""Calculates the *noise parametrized* denoising score matching (DSM) loss for NSE.
    Minimizing this loss estimates the noise :math: `\eplison_phi`, from which the score function
    can be calculated as

        .. math: `s_\phi(\theta, x, t) = - \sigma(t) * \epsilon_\phi(\theta, x, t)`.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N\|
            \epsilon_\phi(\alpha(t_i) \theta_i + \sigma(t_i) \epsilon_i, x_i, t_i)
            - \epsilon_i
        \|_2^2

    where :math:`t_i \sim \mathcal{U}(0, 1)` and :math:`\epsilon_i \sim \mathcal{N}(0, I)`.

    Arguments:
        estimator: A regression network :math:`\epsilon_\phi(\theta, x, t)`.
    """

    def __init__(self, estimator: NSE):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.
            kwargs: Additional args for the forward method of the estimator.

        Returns:
            The scalar loss :math:`l`.
        """

        t = torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)

        scaling = self.estimator.alpha(t)**.5
        sigma = self.estimator.sigma(t)

        eps = torch.randn_like(theta)
        theta_t = scaling[:, None] * theta + sigma[:, None] * eps

        return (self.estimator(theta_t, x, t, **kwargs) - eps).square().mean()


if __name__ == "__main__":
    theta = torch.randn(128, 2)
    x = torch.randn(10,2)
    t = torch.rand(1)
    nse = NSE(2,2)

    nse.predictor_corrector((128,),
                            x=x,
                            steps=2,
                            prior_score_fun=lambda theta, t: torch.ones_like(theta),
                            eta=0.01,
                            corrector_lda=0.1,
                            n_steps=2,
                            r=.5,
                            predictor_type='ddim',
                            verbose=True).cpu()