import torch
import torch.nn as nn
import torch.nn.functional as F

from nse import NSE
from torch import Size, Tensor
from typing import Callable, Optional, Tuple
from zuko.nn import MLP
from zuko.utils import broadcast

from tall_posterior_sampler import tweedies_approximation
from functools import partial


class PF_NSE(NSE):
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        n_max: int,
        freqs: int = 3,
        build_net: Callable[[int, int], nn.Module] = MLP,
        embedding_nn_theta: nn.Module = nn.Identity(),
        embedding_nn_x: nn.Module = nn.Identity(),
        **kwargs: dict,
    ) -> None:
        """Creates a neural score estimatior for partially fatorized
        conditional density estimation (PF-NSE). The model is based on the
        Neural Score Estimator (NSE) and is designed to handle context sets
        of varying sizesbetween 1 and `n_max`.
        At sampling, the factorization is perfomed over batches of context sets
        of size `n_max`. The `annealed_langevin_geffner` method corresponds to
        the `PF-NPSE` method from [Geffner et al, 2023].

        For `n_max=1`, the the class reduces to the NSE class, with full factorization
        over the context set (e.g. `F-NPSE` from [Geffner et al, 2023]).

        Args:
            theta_dim: The dimensionality `m` of the parameter space.
            x_dim: The dimensionality `d` of the observation space.
            freqs: The number of time embedding frequencies.
            n_max: The maximum number context observations.
            build_net: The network constructor. It takes the
                number of input and output features as positional arguments.
            embedding_nn_theta: The embedding network for the parameters `theta`.
                Default is the identity function.
            embedding_nn_x: The embedding network for the observations `x`.
                Default is the identity function.
            kwargs: Keyword arguments passed to the network constructor `build_net`.
        """

        if n_max > 1:
            print()
            print("WARNING: It is recommended to use NSE for n_max=1.")
            print()

        super().__init__(
            theta_dim=theta_dim,
            x_dim=x_dim,
            freqs=freqs,
            build_net=build_net,
            net_type="default",
            embedding_nn_theta=embedding_nn_theta,
            embedding_nn_x=embedding_nn_x,
            **kwargs,
        )

        self.n_max = n_max
        # Adjust the input dimension of the neural network
        self.net = build_net(
            self.theta_emb_dim + self.x_emb_dim + 2 * freqs + n_max, theta_dim, **kwargs
        )
        # Get appropriate implementation of the Tweedie's approximation
        self.tweedies_approximator = partial(
            tweedies_approximation, partial_factorization=True
        )

    def _create_mask(self, n: Tensor) -> Tensor:
        shape = (self.n_max,)
        batch_dim = (n.ndim > 2) * 1
        if batch_dim != 0:
            shape = (batch_dim,) + shape

        mask = ((torch.arange(self.n_max).expand(shape).to(n) < n) * 1).unsqueeze(-1)
        return mask  # (*, n_max, 1)

    def set_mask(self, mask: Tensor) -> None:
        """Sets the masks in context embedding net."""
        if hasattr(self.embedding_nn_x, "set_mask"):
            self.embedding_nn_x.set_mask(mask)
        elif hasattr(self.embedding_nn_x, "__iter__"):
            for layer in self.embedding_nn_x:
                if hasattr(layer, "set_mask"):
                    layer.set_mask(mask)

    def aggregate(self, x: Tensor, n: Tensor) -> Tensor:
        r"""Aggregates context sets of size `n` in `x` by computing
        the mean over set elements `x^j`:

            `x_agg = 1/n sum_j(x^j)`

        Arguments:
            x: (masked) context variables, with shape `(*, n_max, d)`
            n: sizes of the context sets, with shape `(*,)`

        Returns:
            x_agg: aggregated context variable, with shape `(*, d)`
        """
        if n.sum() != 0:
            # variable context sizes
            x_agg = torch.sum(x, dim=-2) / n
        else:
            # fixed context size
            x_agg = torch.mean(x, dim=-2)
        return x_agg

    def forward(
        self,
        theta: Tensor,
        x: Tensor,
        t: Tensor,
        n: Tensor,
    ) -> Tensor:
        r"""
        Arguments:
            theta: The parameters `\theta`, with shape `(*, m)`.
            x: The observation `x`, with shape `(*, n_max, d)`.
            t: The time `t`, with shape `(*,1).`
            n: The sizes of the context sets, with shape `(*,1)`.
                Values either range between 1 and `n_max`, or be all 0 (no contex = prior).

        Returns:
            The estimated noise `epsilon(\theta, x, t)`, with shape `(*, D)`.
        """

        # Define masks to allow variable set sizes
        mask = self._create_mask(n)
        assert (n == torch.sum(mask, dim=-2)).all(), "Mask is incorrect: n != sum(mask)"

        # Mask observations
        x = mask * x
        # Set masks for embedding net
        self.set_mask(mask)

        # Positional embedding for the time variable
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        # Embeddings for parameters and observations
        theta = self.embedding_nn_theta(theta)
        x = self.embedding_nn_x(x)

        # Aggregate observation sets
        x = mask * x  # necessary if x isn't masked within the embedding net
        x = self.aggregate(x, n)  # (*, x_emb_dim)

        # One-hot encode the set sizes n between 0 and n_max
        if not n.eq(0).all():
            n = F.one_hot(n.long().squeeze() - 1, num_classes=self.n_max)
        else:
            n = torch.zeros((n.shape[0], self.n_max))
        # Compute the score
        theta, x, t, n = broadcast(theta, x, t, n, ignore=1)
        return self.net(torch.cat((theta, x, t, n), dim=-1))

    def score(self, theta: Tensor, x: Tensor, t: Tensor, n: Tensor, **kwargs) -> Tensor:
        return -self(theta, x, t, n) / self.sigma(t)

    def _create_pf_data(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # total number of context observations
        n_observations = x.shape[0] if len(x.shape) > 1 else 1
        # number of full subsets of size n_max
        k = n_observations // self.n_max
        # number of remaining context observations
        r = n_observations % self.n_max

        # list of k subsets of size (n_max, dim_x)
        idx_list = [
            torch.arange(i * self.n_max, (i + 1) * self.n_max) for i in range(k)
        ]
        xs = [x[idx, :] for idx in idx_list]
        ns = [torch.ones((1,)).to(x) * self.n_max for _ in range(k)]

        # append remaining subset of size (r, dim_x)
        # (padded with zeros to get shape (n_max, dim_x))
        if r > 0:
            x_remain = x[torch.arange(n_observations - r, n_observations)]
            mask = torch.zeros((self.n_max - r, x.shape[-1])).to(x)
            x_c = torch.cat((x_remain, mask), dim=0)
            xs.append(x_c)
            ns.append(torch.ones((1,)).to(x) * r)

        # get tensor of shape (len(xs), n_max, dim_x), len(xs) = k or k+1
        xs = torch.stack(xs).to(x)
        ns = torch.stack(ns).to(x)

        return xs, ns

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
        """Performs the DDIM algorithm for the PF-NSE model.
        The only difference with the NSE model is that the context sets
        are reshaped to be of size `n_max` before the algorithm is run.
        """

        xs, ns = self._create_pf_data(x)
        print("Reshaped context sets: ", f"x: {xs.shape}, n: {ns.shape}")

        if "n" in kwargs:
            assert (
                ns.shape == kwargs["n"].shape
            ), f"n.shape = {ns.shape} != kwargs['n'].shape = {kwargs['n'].shape}"
            ns = kwargs["n"]
            del kwargs["n"]

        return super().ddim(
            shape=shape,
            x=xs,
            steps=steps,
            eta=eta,
            verbose=verbose,
            theta_clipping_range=theta_clipping_range,
            n=ns,
            **kwargs,
        )

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
        """Performs the Predictor-Corrector algorithm for the PF-NSE model.
        The only difference with the NSE model is that the context sets
        are reshaped to be of size `n_max` before the algorithm is run.
        """
        xs, ns = self._create_pf_data(x)
        print("Reshaped context sets: ", f"x: {xs.shape}, n: {ns.shape}")

        return super().predictor_corrector(
            shape=shape,
            x=xs,
            steps=steps,
            verbose=verbose,
            predictor_type=predictor_type,
            corrector_type=corrector_type,
            theta_clipping_range=theta_clipping_range,
            n=ns,
            **kwargs,
        )

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
    ) -> Tensor:
        """Corresponds to the PF-NPSE method from [Geffner et al, 2023].
        The only differnce with the F-NPSE method implemented in `NSE` is that
        the context sets are reshaped to be of size `n_max` before the algorithm is run.
        """
        xs, ns = self._create_pf_data(x)
        print("Reshaped context sets: ", f"x: {xs.shape}, n: {ns.shape}")

        return super().annealed_langevin_geffner(
            shape=shape,
            x=xs,
            prior_score_fn=prior_score_fn,
            prior_type=prior_type,
            steps=steps,
            lsteps=lsteps,
            tau=tau,
            theta_clipping_range=theta_clipping_range,
            verbose=verbose,
            n=ns,
            **kwargs,
        )


if __name__ == "__main__":
    from torch.func import vmap

    # Test PF-NPSE
    theta_dim = 2
    x_dim = 2
    n_max = 10
    pf_nse = PF_NSE(theta_dim, x_dim, n_max)
    nse = NSE(theta_dim, x_dim)

    x = torch.randn((107, x_dim))
    xs, ns = pf_nse._create_pf_data(x)
    print(xs.shape, ns.shape)

    cov_est = vmap(
        lambda x: pf_nse.ddim(shape=(1000,), x=x, steps=100, eta=0.5),
        randomness="different",
    )(xs)
    cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
    print(cov_est.shape)

    x_ = torch.zeros_like(xs[0][None, :])
    n_ = torch.zeros_like(ns[0][None, :])
    cov_est_prior = vmap(
        lambda x: pf_nse.ddim(shape=(1000,), x=x, steps=100, eta=0.5, n=n_),
        randomness="different",
    )(x_)
    cov_est_prior = vmap(lambda x: torch.cov(x.mT))(cov_est_prior)
    print(cov_est_prior.shape)

    samples_gauss = pf_nse.ddim(
        shape=(1000,),
        x=x,
        eta=1,
        steps=1000,
        prior_score_fn=None,
        prior=None,
        dist_cov_est=cov_est,
        dist_cov_est_prior=cov_est_prior,
        cov_mode="GAUSS",
        clf_free_guidance=True,
        verbose=True,
    )
    print(samples_gauss.shape)

    samples_pc = pf_nse.predictor_corrector(
        shape=(1000,),
        x=x,
        steps=400,
        verbose=True,
        prior_score_fun=None,
        eta=1,
        lsteps=5,
        theta_clipping_range=(None, None),
        clf_free_guidance=True,
        r=0.5,
        predictor_type="id",
    )
    print(samples_pc.shape)

    samples_geffner = pf_nse.annealed_langevin_geffner(
        shape=(1000,), x=x, prior_score_fn=None, clf_free_guidance=True, verbose=True
    )
    print(samples_geffner.shape)
