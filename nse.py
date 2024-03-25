import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Distribution
from tqdm import tqdm
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.nn import MLP
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from embedding_nets import FNet
from tall_posterior_sampler import (prec_matrix_backward,
                                    tweedies_approximation,
                                    tweedies_approximation_prior)


class NSE(nn.Module):
    r"""Creates a neural score estimation (NSE) network.

    Args:
        theta_dim (int): The dimensionality :math:`m` of the parameter space.
        x_dim (int): The dimensionality :math:`d` of the observation space.
        freqs (int): The number of time embedding frequencies.
        build_net (Callable): The network constructor. It takes the
            number of input and output features as positional arguments.
        net_type (str): The type of final score network. Can be 'default' or 'fnet'.
        embedding_nn_theta (Callable): The embedding network for the parameters :math:`\theta`.
            Default is the identity function.
        embedding_nn_x (Callable): The embedding network for the observations :math:`x`.
            Default is the identity function.
        kwargs (dict): Keyword arguments passed to the network constructor `build_net`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        freqs: int = 3,
        build_net: Callable[[int, int], nn.Module] = MLP,
        net_type: str = "default",
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
        self.net_type = net_type

        if net_type == "default":
            self.net = build_net(
                self.theta_emb_dim + self.x_emb_dim + 2 * freqs, theta_dim, **kwargs
            )
        elif net_type == "fnet":
            self.net = FNet(
                dim_input=theta_dim, dim_cond=x_dim, dim_embedding=128, n_layers=1
            )
        else:
            raise NotImplementedError("Unknown net_type")

        self.register_buffer("freqs", torch.arange(1, freqs + 1) * math.pi)
        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def get_theta_x_embedding_dim(self, theta_dim: int, x_dim: int) -> int:
        r"""Returns the dimensionality of the embeddings for :math:`\theta` and :math:`x`."""
        theta, x = torch.ones((1, theta_dim)), torch.ones((1, x_dim))
        return (
            self.embedding_nn_theta(theta).shape[-1],
            self.embedding_nn_x(x).shape[-1],
        )

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Args:
            theta (torch.Tensor): The parameters :math:`\theta`, with shape :math:`(*, m)`.
            x (torch.Tensor): The observation :math:`x`, with shape :math:`(*, d)`.
            t (torch.Tensor): The time :math:`t`, with shape :math:`(*,).`

        Returns:
            (torch.Tensor): The estimated noise :math:`\epsilon_\phi(\theta, x, t)`, with shape :math:`(*, m)`.
        """

        if self.net_type == "default":
            t = self.freqs * t[..., None]
            t = torch.cat((t.cos(), t.sin()), dim=-1)

            theta = self.embedding_nn_theta(theta)
            x = self.embedding_nn_x(x)

            theta, x, t = broadcast(theta, x, t, ignore=1)

            return self.net(torch.cat((theta, x, t), dim=-1))

        if self.net_type == "fnet":
            return self.net(theta, x, t)

    # The following function define the VP SDE with linear noise schedule beta(t):
    # dtheta = f(t) theta dt + g(t) dW = -0.5 * beta(t) theta dt + sqrt(beta(t)) dW

    def score(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
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

    def ode(self, theta: Tensor, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        r"""The probability flow ODE corresponding to the VP SDE."""
        return self.f(t) * theta + self.g(t) ** 2 / 2 * self(
            theta, x, t, **kwargs
        ) / self.sigma(t)

    def flow(self, x: Tensor, **kwargs) -> Distribution:
        r"""
        Args:
            x (torch.Tensor): observation :math:`x`, with shape :math:`(*, d)`.
            kwargs (dict): additional args for the forward method.

        Returns:
            (zuko.distributions.Distribution): The normalizing flow
                :math:`p_\phi(\theta | x)` induced by the probability flow ODE.
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

    def mean_pred(
        self, theta: Tensor, score: Tensor, alpha_t: Tensor, **kwargs
    ) -> Tensor:
        """Mean predictor of the backward kernel
        (used in DDIM sampler and gaussian approximation).
        """
        upsilon = 1 - alpha_t
        mean = (alpha_t ** (-0.5)) * (theta + upsilon * score)
        return mean

    def bridge_mean(
        self,
        alpha_t: Tensor,
        alpha_t_1: Tensor,
        theta_t: Tensor,
        theta_0: Tensor,
        bridge_std: float,
    ) -> Tensor:
        """Bridge mean for the DDIM sampler."""
        est_noise = (theta_t - (alpha_t**0.5) * theta_0) / ((1 - alpha_t) ** 0.5)
        return (alpha_t_1**0.5) * theta_0 + (
            (1 - alpha_t_1 - bridge_std**2) ** 0.5
        ) * est_noise

    def ddim(
        self,
        shape: Size,
        x: Tensor,
        steps: int = 64,
        eta: float = 1.0,
        verbose: bool = False,
        theta_clipping_range=(None, None),
        **kwargs,
    ) -> Tensor:
        r"""Sampler from Denoising Diffusion Implicit Models (DDIM, Song et al., 2021),
            but adapted to the tall data setting with `n` context observations `x`.

        Args:
            shape (torch.Size): The shape of the samples.
            x (torch.Tensor): The conditioning variable for the score network, with shape :math:`(n, m)`.
            steps (int): The number of steps in the diffusion process.
            eta (float): The noise level for the bridge process.
            verbose (bool): If True, displays a progress bar.

        Returns:
            (torch.Tensor): The samples from the diffusion process, with shape :math:`(shape[0], m)`.
        """

        if x.shape[0] == shape[0] or len(x.shape) == 1 or x.shape[0] == 1:
            score_fun = self.score
        else:
            score_fun = partial(self.factorized_score, **kwargs)

        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            theta = self.ddim_step(theta, x, t, score_fun, dt, eta)
            if theta_clipping_range[0] is not None:
                theta = theta.clip(*theta_clipping_range)
        return theta

    def ddim_step(
        self,
        theta: Tensor,
        x: Tensor,
        t: Tensor,
        score_fun: Callable[[Tensor, Tensor, Tensor], Tensor],
        dt: float,
        eta: float,
        **kwargs,
    ):
        r"""One step of the DDIM sampler."""

        score = score_fun(theta, x, t).detach()

        alpha_t = self.alpha(t)
        alpha_t_1 = self.alpha(t - dt)
        bridge_std = eta * (
            (((1 - alpha_t_1) / (1 - alpha_t)) * (1 - alpha_t / alpha_t_1)) ** 0.5
        )

        pred_theta_0 = self.mean_pred(theta=theta, score=score, alpha_t=alpha_t)
        theta_mean = self.bridge_mean(
            alpha_t=alpha_t,
            alpha_t_1=alpha_t_1,
            theta_0=pred_theta_0,
            theta_t=theta,
            bridge_std=bridge_std,
        )

        theta = theta_mean + torch.randn_like(theta_mean) * bridge_std
        return theta

    def langevin_corrector(
        self,
        theta: Tensor,
        x: Tensor,
        t: Tensor,
        score_fun: Callable[[Tensor, Tensor, Tensor], Tensor],
        n_steps: int,
        r: float,
        **kwargs,
    ) -> Tensor:
        r"""Langevin corrector for the Predictor-Corrector (PC) sampler.

        Args:
            theta (torch.Tensor): The current state of the diffusion process, with shape :math:`(n, m)`.
            x (torch.Tensor): The conditioning variable for the score network, with shape :math:`(n, m)`.
            t (torch.Tensor): The time :math:`t`, with shape :math:`(n,)`.
            score_fun (Callable): The score function.
            n_steps (int): The number of Langevin steps.
            r (float): The step size (or signal-to-noise ratio) for the Langevin dynamics.

        Returns:
            (torch.Tensor): The corrected sampple state of the diffusion process, with shape :math:`(n, m)`.
        """

        for _ in range(n_steps):
            z = torch.randn_like(theta)
            g = score_fun(theta, x, t).detach()
            # eps = 2*alpha_t*(r*torch.linalg.norm(z, axis=-1).mean(axis=0)/torch.linalg.norm(g, axis=-1).mean(axis=0))**2
            eps = (
                r
                * (self.alpha(t) ** 0.5)
                * min(self.sigma(t) ** 2, 1 / g.square().mean())
            )
            theta = theta + eps * g + ((2 * eps) ** 0.5) * z
        return theta

    def predictor_corrector(
        self,
        shape: Size,
        x: Tensor,
        steps: int = 64,
        verbose: bool = False,
        predictor_type="ddim",
        corrector_type="langevin",
        theta_clipping_range=(None, None),
        **kwargs,
    ) -> Tensor:
        r"""Predictor-Corrector (PC) sampling algorithm (Song et al., 2021),
        but adapted to the tall data setting with `n` context observations `x`.

        The PC sampler is a generalization of the
        - Langevin Dynamics: predictor_type = 'id', corrector_type = 'langevin'
        - DDIM sampler: predictor_type = 'ddim', corrector_type = 'id'

        Args:
            shape (torch.Size): The shape of the samples.
            x (torch.Tensor): The conditioning variable for the score network, with shape :math:`(n, m)`.
            steps (int): The number of steps in the diffusion process.
            verbose (bool): If True, displays a progress bar.
            predictor_type (str): The type of predictor. Can be 'ddim' or 'id'.
            corrector_type (str): The type of corrector. Can be 'langevin' or 'id'.
            theta_clipping_range (Tuple[float, float]): The range for clipping the samples.
            kwargs (dict): Additional args for the score function, the predictor and corrector.

        Returns:
            (torch.Tensor): The samples from the diffusion process, with shape :math:`(shape[0], m)`.
        """

        # get simple or tall data score function

        if x.shape[0] == shape[0] or len(x.shape) == 1 or x.shape[0] == 1:
            score_fun = self.score
        else:
            if corrector_type == "langevin":
                score_fun = partial(self.factorized_score_geffner, **kwargs)
            else:
                score_fun = partial(self.factorized_score, **kwargs)

        # get predictor and corrector functions
        if predictor_type == "ddim":
            predictor_fun = partial(self.ddim_step, **kwargs)
        elif predictor_type == "id":
            predictor_fun = lambda theta, x, t, score_fun, dt: theta
        else:
            raise NotImplemented("")
        if corrector_type == "langevin":
            corrector_fun = partial(self.langevin_corrector, **kwargs)
        elif corrector_type == "id":
            corrector_fun = lambda theta, x, t, score_fun: theta
        else:
            raise NotImplemented("")

        # run the PC sampler
        time = torch.linspace(1, 0, steps + 1).to(x)
        dt = 1 / steps

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for t in tqdm(time[:-1], disable=not verbose):
            # predictor step
            theta_pred = predictor_fun(
                theta=theta, x=x, t=t, score_fun=score_fun, dt=dt
            )
            # corrector step
            theta = corrector_fun(theta=theta_pred, x=x, t=t - dt, score_fun=score_fun)
            if theta_clipping_range[0] is not None:
                theta = theta.clip(*theta_clipping_range)
        return theta

    # The following functions are samplers for the tall data setting
    # with n context observations x, based on the factorized posterior.

    def annealed_langevin_geffner(
        self,
        shape: Size,
        x: Tensor,
        prior_score_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        prior_type: Optional[str] = None,
        steps: int = 400,
        lsteps: int = 5,
        tau: float = 0.5,
        theta_clipping_range=(None, None),
        verbose: bool = False,
        **kwargs,
    ):
        time = torch.linspace(1, 0, steps + 1).to(x)

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        for i, t in enumerate(tqdm(time[:-1], disable=not verbose)):
            if i < steps - 1:
                gamma_t = self.alpha(t) / self.alpha(t - time[-2])
            else:
                gamma_t = self.alpha(t)
            
            delta = tau * (1 - gamma_t) / (gamma_t**0.5)
            for _ in range(lsteps):
                z = torch.randn_like(theta)

                if len(x.shape) == 1 or x.shape[0] == 1:
                    score = self.score(theta, x, t).detach()
                else:
                    score = self.factorized_score_geffner(
                        theta, x, t, prior_score_fn, prior_type=prior_type, **kwargs
                    ).detach()

                theta = theta + delta * score + ((2 * delta) ** 0.5) * z
            
            if theta_clipping_range[0] is not None:
                theta = theta.clip(*theta_clipping_range)

        return theta

    def factorized_score_geffner(
        self, theta, x, t, prior_score_fun, clf_free_guidance=None, **kwargs
    ):
        r"""Factorized score function for the tall data setting with n context observations x.
        From Geffner et al. (2023).
        """
        # Defining variables
        n_observations = x.shape[0] if len(x.shape) > 1 else 1
        n_samples = theta.shape[0]
        theta_ = theta.clone().requires_grad_(True)

        # Calculating m, Sigma and scores for the posteriors
        if self.net_type == "fnet":
            scores = self(
                theta[:, None, :].repeat(1, n_observations, 1).reshape(n_observations * n_samples, -1),
                x[None, :, :].repeat(n_samples, 1, 1).reshape(n_observations * n_samples, -1),
                t[None, None].repeat(n_samples * n_observations, 1),
                **kwargs,
            ) / -self.sigma(t)
            scores = scores.reshape(n_samples, n_observations, -1)
        else:
            scores = self.score(theta[:,None], x[None, :], t).detach()

        if clf_free_guidance:
            x_ = torch.zeros_like(x[0]) # replace 1 with n_max for multiple context observations ?
            prior_score = self.score(theta[:,None], x_[None, :], t).detach()[:,0,:]
        else:
            prior_score = prior_score_fun(theta[None], t)[0]
        aggregated_score = (1 - n_observations) * prior_score + scores.sum(axis=1)
        theta_.detach()
        return aggregated_score

    def factorized_score(
        self,
        theta,
        x_obs,
        t,
        prior_score_fn,
        prior,
        dist_cov_est=None,
        dist_cov_est_prior=None,
        cov_mode="JAC",
        prior_type="gaussian",
        clf_free_guidance=False,
    ):
        r"""Factorized score function for the tall data setting with n context observations x.
        Our proposition ("GAUSS" and "JAC").
        """
        # device
        n_obs = x_obs.shape[0]
        prec_0_t, _, scores = tweedies_approximation(
            x=x_obs,
            theta=theta,
            nse=self,
            t=t,
            score_fn=self.score,
            dist_cov_est=dist_cov_est,
            mode=cov_mode,
        )

        if clf_free_guidance:
            x_=torch.zeros_like(x_obs[0][None,:]) # replace 1 with n_max for multiple context observations ?
            prec_prior_0_t_cfg, _, prior_score_cfg = tweedies_approximation(
                x=x_,
                theta=theta,
                nse=self,
                t=t,
                score_fn=self.score,
                dist_cov_est=dist_cov_est_prior,
                mode=cov_mode,
            )
            prec_score_prior_cfg = (prec_prior_0_t_cfg @ prior_score_cfg[..., None])[..., 0][:,0,:]
            prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
            lda_cfg = prec_prior_0_t_cfg[:,0,:] * (1 - n_obs) + prec_0_t.sum(dim=1)
            weighted_scores_cfg = prec_score_prior_cfg + (
                prec_score_post - prec_score_prior_cfg[:, None]
            ).sum(dim=1)

            total_score = torch.linalg.solve(A=lda_cfg, B=weighted_scores_cfg)
        
        else:
            if prior_type == "gaussian":
                prior_score = prior_score_fn(theta, t)
                prec_prior_0_t = prec_matrix_backward(t, prior.covariance_matrix, self)
                prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
                prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
                lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1)
                weighted_scores = prec_score_prior + (
                    prec_score_post - prec_score_prior[:, None]
                ).sum(dim=1)

                total_score = torch.linalg.solve(A=lda, B=weighted_scores)

            else:
                prior_score = prior_score_fn(theta, t)
                total_score = (1 - n_obs) * prior_score + scores.sum(dim=1)
                if (self.alpha(t) ** 0.5 > 0.5) and (n_obs > 1):
                    prec_prior_0_t, _, _ = tweedies_approximation_prior(
                        theta, t, prior_score_fn, nse=self
                    )
                    prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
                    prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
                    lda = prec_prior_0_t * (1 - n_obs) + prec_0_t.sum(dim=1)
                    weighted_scores = prec_score_prior + (
                        prec_score_post - prec_score_prior[:, None]
                    ).sum(dim=1)

                    total_score = torch.linalg.solve(A=lda, B=weighted_scores)

        return total_score  # / (1 + (1/n_obs)*torch.abs(total_score))

    # The following functions are the original samplers for the single observation setting.

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
                    * (self.alpha(t) ** 0.5)
                    * min(self.sigma(t) ** 2, 1 / score.square().mean())
                )
                theta = theta + delta * score + torch.sqrt(2 * delta) * z

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

    Args:
        estimator (NSE): A regression network :math:`\epsilon_\phi(\theta, x, t)`.
    """

    def __init__(self, estimator: NSE):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor, **kwargs) -> Tensor:
        r"""
        Args:
            theta (torch.Tensor): The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x (torch.Tensor): The observation :math:`x`, with shape :math:`(N, L)`.
            kwargs (dict): Additional args for the forward method of the estimator.

        Returns:
            (torch.Tensor): The noise parametrized scalar DSM loss :math:`l`.
        """

        t = torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)

        scaling = self.estimator.alpha(t) ** 0.5
        sigma = self.estimator.sigma(t)

        eps = torch.randn_like(theta)
        theta_t = scaling[:, None] * theta + sigma[:, None] * eps

        return (self.estimator(theta_t, x, t, **kwargs) - eps).square().mean()


if __name__ == "__main__":
    theta = torch.randn(128, 2)
    x = torch.randn(10, 2)
    t = torch.rand(1)
    nse = NSE(2, 2)

    nse.predictor_corrector(
        (128,),
        x=x,
        steps=2,
        prior_score_fun=lambda theta, t: torch.ones_like(theta),
        eta=0.01,
        corrector_lda=0.1,
        n_steps=2,
        r=0.5,
        predictor_type="ddim",
        verbose=True,
    ).cpu()