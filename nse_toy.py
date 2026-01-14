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



def mean_backward(theta, t, score_fn, nse, **kwargs):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    return (
            1
            / (alpha_t**0.5)
            * (theta + sigma_t**2 * score_fn(theta=theta, t=t, **kwargs))
    )


def sigma_backward(t, dist_cov, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    eye = torch.eye(dist_cov.shape[-1]).to(alpha_t.device)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    if (alpha_t**.5) < 0.5:
        return torch.linalg.inv(torch.linalg.inv(dist_cov) + (alpha_t / sigma_t ** 2) * eye)
    return (((sigma_t ** 2) / alpha_t) * eye
            - (((sigma_t ** 2)**2 / alpha_t)) * torch.linalg.inv(
                alpha_t * (dist_cov.to(alpha_t.device) - eye)
                + eye))


def prec_matrix_backward(t, dist_cov, nse):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)
    eye = torch.eye(dist_cov.shape[-1]).to(alpha_t.device)
    # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
    #return torch.linalg.inv(dist_cov * alpha_t + (sigma_t**2) * eye)
    return (torch.linalg.inv(dist_cov) + (alpha_t / sigma_t ** 2) * eye)


def tweedies_approximation(theta, x, t, score_fn, nse, dist_cov_est=None, mode='JAC', clip_mean_bounds = (None, None)):
    alpha_t = nse.alpha(t)
    sigma_t = nse.sigma(t)

    def score_jac(theta, x):
        n_theta = theta.shape[0]
        n_x = x.shape[0]
        score = score_fn(theta=theta[:, None, :].repeat(1, n_x, 1).reshape(n_theta * n_x, -1),
                         t=t[None, None].repeat(n_theta * n_x, 1),
                         x=x[None, ...].repeat(n_theta, 1, 1).reshape(n_theta * n_x, -1))
        score = score.reshape(n_theta, n_x, -1)
        return score, score

    if mode == 'JAC':
        def score_jac(theta, x):
            score = score_fn(theta=theta[None, :],
                             t=t[None, None],
                             x=x[None, ...])[0]
            return score, score
        jac_score, score = vmap(lambda theta: vmap(jacrev(score_jac, has_aux=True))(theta[None].repeat(x.shape[0], 1), x))(
            theta)
        cov = (sigma_t**2 / alpha_t) * (torch.eye(theta.shape[-1], device=theta.device) + (sigma_t**2)*jac_score)
        prec = torch.linalg.inv(cov)
    elif mode == 'GAUSS':
        prec = prec_matrix_backward(t=t, dist_cov=dist_cov_est, nse=nse)
        # eye = torch.eye(dist_cov_est.shape[-1]).to(alpha_t.device)
        # # Same as the other but using woodberry. (eq 53 or https://arxiv.org/pdf/2310.06721.pdf)
        # cov = torch.linalg.inv(torch.linalg.inv(dist_cov_est) + (alpha_t / sigma_t ** 2) * eye)
        prec = prec[None].repeat(theta.shape[0], 1, 1, 1)
        score = score_jac(theta, x)[0]
        # score = vmap(lambda theta: vmap(partial(score_fn, t=t),
        #                                 in_dims=(None, 0),
        #                                 randomness='different')(theta, x),
        #              randomness='different')(theta)
    else:
        raise NotImplemented("Available methods are GAUSS, PSEUDO, JAC")
    mean = (1 / (alpha_t**0.5) * (theta[:, None] + sigma_t**2 * score))
    if clip_mean_bounds[0]:
        mean = mean.clip(*clip_mean_bounds)
    return prec, mean, score

class PositionalEncodingVector(nn.Module):

    def __init__(self, d_model: int, M: int):
        super().__init__()
        div_term = 1 / M**(2 * torch.arange(0, d_model, 2) / d_model)
        self.register_buffer('div_term', div_term)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        exp_div_term = self.div_term.reshape(*(1,)*(len(x.shape) - 1), -1)
        return torch.cat((torch.sin(x*exp_div_term), torch.cos(x*exp_div_term)), dim=-1)



class _FBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 emb_channels,
                 dropout=0,
                 eps=1e-5,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.norm0 = torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=eps)
        self.conv0 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.affine = torch.nn.Linear(in_features=emb_channels, out_features=out_channels)
        self.norm1 = torch.nn.GroupNorm(num_groups=16, num_channels=out_channels, eps=eps)
        self.conv1 = torch.nn.Linear(in_features=out_channels, out_features=out_channels)

        self.skip = None
        if out_channels != in_channels:
            self.skip = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.skip_scale = .5 ** .5

    def forward(self, x, emb):
        silu = torch.nn.functional.silu
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb)

        x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        return x


class FNet(torch.nn.Module):

    def     __init__(self,
                     dim_input,
                     dim_cond,
                     dim_embedding=512,
                     n_layers=3):
        super().__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_embedding),
            # torch.nn.SiLU(),
            # torch.nn.Linear
        )
        self.cond_layer = torch.nn.Linear(dim_cond, dim_embedding)

        self.embedding_map = torch.nn.Sequential(
            torch.nn.Linear(dim_embedding, 2*dim_embedding),
            torch.nn.GroupNorm(num_groups=16,
                               num_channels=2*dim_embedding),
            torch.nn.SiLU(),
            torch.nn.Linear(2*dim_embedding, dim_embedding)
        )
        self.res_layers = torch.nn.ModuleList([_FBlock(in_channels=dim_embedding // 2**i,
                                                       out_channels=dim_embedding // 2**(i+1),
                                                       emb_channels=dim_embedding,
                                                       dropout=.1) for i in range(n_layers)])
        self.final_layer = torch.nn.Sequential(torch.nn.GroupNorm(num_groups=16, num_channels=dim_embedding // 2**(n_layers)),
                                               torch.nn.SiLU(),
                                               torch.nn.Linear(dim_embedding // 2**(n_layers),
                                                               dim_input, bias=False))
        self.time_embedding = PositionalEncodingVector(d_model=dim_embedding, M=1000)

    def forward(self, theta, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=theta.device)
        theta_emb = self.input_layer(theta)
        x_emb = self.cond_layer(x)
        t_emb = self.time_embedding(t)
        # theta_emb = theta_emb.reshape(-1, theta_emb.shape[-1])
        # x_emb = x_emb.reshape(-1, x_emb.shape[-1])
        # t_emb = t_emb.reshape(-1, t_emb.shape[-1])
        emb = self.embedding_map(t_emb + x_emb)
        for lr in self.res_layers:
            theta_emb = lr(x=theta_emb, emb=emb)
        return self.final_layer(theta_emb).reshape(*theta.shape) #- theta


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
            beta_min: float = 0.1,
            beta_max: float = 40,
            M: int = 1000,
            eps_t: float = 1e-10,
            sampling_dist: str = 'Uniform',
            **kwargs,
    ):
        super().__init__()
        self.eps_t = eps_t
        self.initialize_sampling_fun(sampling_dist)
        self.beta_min = beta_min
        self.beta_d = (beta_max - beta_min)
        self.M = M
        self.net = FNet(dim_input=theta_dim,
                        dim_cond=x_dim,
                        dim_embedding=128,
                        n_layers=1)

        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def initialize_sampling_fun(self, sampling_dist):
        if sampling_dist == 'Uniform':
            self._dist = torch.distributions.Uniform(low=self.eps_t, high=1)
            def sampling_fun(n_samples):
                return self._dist.sample((n_samples,))
        elif sampling_dist == 'LogNormal':
            self._dist = torch.distributions.LogNormal(loc=-1.2, scale=1)
            def sampling_fun(n_samples):
                samples = self._dist.sample((n_samples,))
                return samples.clip(self.eps_t, 1)
        else:
            raise NotImplemented
        self.sampling_fun = sampling_fun

    def save(self, path, meta_info):
        self.net.eval()
        self.net.cpu()
        torch.save({
            "state_dict_f_net": self.net.state_dict(),
            "meta": meta_info,
        },
            path)

    def load(self, path):
        infos = torch.load(path)
        self.net.load_state_dict(infos["state_dict_f_net"])
        return infos["meta"]

    def forward(self,
                theta: Tensor,
                x: Tensor,
                t: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            t: The time :math:`t`, with shape :math:`(*,).`

        Returns:
            The estimated noise :math:`\epsilon_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """
        return self.net(theta, x, t)

    def loss(self, theta, x, **kwargs):
        inv_std = self.sampling_fun(theta.shape[0]).to(theta.device)
        c_sigma = (self.M - 1)*inv_std
        scaling = 1 / (1 + (1 / inv_std)**2)**.5
        sigma = scaling / inv_std
        c_in = 1 / scaling
        c_out = 1 #/ sigma

        #t = torch.rand(theta.shape[0], dtype=theta.dtype, device=theta.device)

        eps = torch.randn_like(theta)
        theta_t = scaling[:, None] * theta + sigma[:, None] * eps
        eps_est = self.net(c_in[:, None] * theta_t,
                           x,
                           c_sigma[:, None], **kwargs) #/ c_out[:, None, None]
        errors = ((eps_est - eps) / sigma[:, None]).square().sum(dim=-1)
        return errors.mean()


    # The following function define the VP SDE with linear noise schedule beta(t):
    # dtheta = f(t) theta dt + g(t) dW = -0.5 * beta(t) theta dt + sqrt(beta(t)) dW

    def score(self, theta, x, t):
        return -self(theta, x, t) / self.sigma(t)

    def beta(self, t: Tensor) -> Tensor:
        r"""Linear noise schedule of the VP SDE:
        .. math:: \beta(t) = 32 t .
        """
        return self.beta_d * t + self.beta_min

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
        log_alpha = .5 * self.beta_d * (t**2) + self.beta_min*t
        return torch.exp(-.5 * log_alpha)#torch.exp(-16 * t**2)

    def std_karas(self, t):
        log_alpha = .5 * self.beta_d * (t ** 2) + self.beta_min * t
        return (torch.exp(log_alpha) - 1)**.5

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
        if x.shape[0] == shape[0]:
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

    def annealed_langevin_geffner(
            self,
            shape: Size,
            x: Tensor,
            prior_score_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
            steps: int = 64,
            lsteps: int = 5,
            tau: float = .5,
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
            delta = tau * (1 - gamma_t) / (gamma_t ** .5)
            for _ in range(lsteps):
                z = torch.randn_like(theta)
                n_x, n_theta = x.shape[0], theta.shape[0]
                post_score = self(theta[:, None, :].repeat(1, n_x, 1).reshape(n_x*n_theta, -1),
                             x[None, :, :].repeat(n_theta, 1, 1).reshape(n_x*n_theta, -1),
                             t[None, None].repeat(n_theta*n_x, 1),
                             **kwargs) / -self.sigma(t)
                post_score = post_score.reshape(n_theta, n_x, -1)
                prior_score, mean_prior_0_t, prec_prior_0_t = prior_score_fn(theta, t)
                score = post_score.sum(dim=1) + (1 - n_x) * prior_score
                #delta = step_size
                theta = theta + delta * score + ((2 * delta)**.5) * z

        return theta

    def deterministic_gaussian_geffner(
            self,
            shape: Size,
            x: Tensor,
            prior_score_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
            steps: int = 64,
            verbose: bool = False,
            **kwargs,
    ):
        """Deterministic Gaussian NSE sampler from Geffner et al. (2023):
        corresponds to Algorithm 2 in Appendix D but with continuous noise schedule 
        (`self.alpha(t)`) and analytic diffused prior score (`prior_score_fn`).

        N.B.: alpha and gamma roles are inverted wrt Geffner et al. notation.
        """
        time = torch.linspace(1, 0, steps + 1).to(x)

        theta = DiagNormal(self.zeros, self.ones).sample(shape)

        n_x, n_theta = x.shape[0], theta.shape[0]

        for i, t in enumerate(tqdm(time[:-1], disable=not verbose)):
            if i < steps - 1:
                gamma_t = self.alpha(t) / self.alpha(t - time[-2])
            else:
                gamma_t = self.alpha(t)

            # compute posterior score for all x
            post_score = self(theta[:, None, :].repeat(1, n_x, 1).reshape(1*n_x*n_theta, -1),
                         x[None, :, :].repeat(n_theta, 1, 1).reshape(n_x*n_theta, -1),
                         t[None, None].repeat(n_theta*n_x, 1),
                         **kwargs) / -self.sigma(t)
            post_score = post_score.reshape(n_theta, n_x, -1)
            # get prior score
            prior_score, _, _ = prior_score_fn(theta, t)
            # compute transition mean
            mu_t = 1/gamma_t**.5 * (n_x * theta + (1-gamma_t) * post_score.sum(dim=1)) - (n_x - 1) * gamma_t**.5 * theta
            mu_t *= 1 / (n_x - gamma_t * (n_x - 1))
            # compute transition variance
            var_t = ((1 - gamma_t) / (n_x - gamma_t * (n_x - 1)))
            # prior correction term
            mu_t += var_t * (1 - n_x) * prior_score
            # update theta
            theta = mu_t + torch.randn_like(mu_t) * var_t**.5


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

    def factorized_score(self,
                         theta,
                         x_obs,
                         t,
                         prior_score_fn,
                         dist_cov_est=None,
                         cov_mode='JAC',
                         ):
        # device
        n_obs = x_obs.shape[0]

        prec_0_t, mean_0_t, scores = tweedies_approximation(x=x_obs,
                                                            theta=theta,
                                                            nse=self,
                                                            t=t,
                                                            score_fn=self.score,
                                                            dist_cov_est=dist_cov_est,
                                                            mode=cov_mode)

        prior_score, mean_prior_0_t, prec_prior_0_t = prior_score_fn(theta, t)
        prec_score_prior = (prec_prior_0_t @ prior_score[..., None])[..., 0]
        prec_score_post = (prec_0_t @ scores[..., None])[..., 0]
        lda = prec_prior_0_t*(1-n_obs) + prec_0_t.sum(dim=1)
        weighted_scores = prec_score_prior + (prec_score_post - prec_score_prior[:, None]).sum(dim=1)

        total_score = torch.linalg.solve(A=lda, B=weighted_scores)
        return total_score #/ (1 + (1/n_obs)*torch.abs(total_score))


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

    def __init__(self, estimator: NSE, eps_t: float = 1e-5):
        super().__init__()
        self.eps_t = eps_t
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