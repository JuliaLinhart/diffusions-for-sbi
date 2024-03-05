import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncodingVector(nn.Module):
    def __init__(self, d_model: int, M: int):
        super().__init__()
        div_term = 1 / M ** (2 * torch.arange(0, d_model, 2) / d_model)
        self.register_buffer("div_term", div_term)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        exp_div_term = self.div_term.reshape(*(1,) * (len(x.shape) - 1), -1)
        return torch.cat(
            (torch.sin(x * exp_div_term), torch.cos(x * exp_div_term)), dim=-1
        )


class _FBlock(torch.nn.Module):
    def __init__(
        self,
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

        self.norm0 = torch.nn.GroupNorm(
            num_groups=16, num_channels=in_channels, eps=eps
        )
        self.conv0 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.affine = torch.nn.Linear(
            in_features=emb_channels, out_features=out_channels
        )
        self.norm1 = torch.nn.GroupNorm(
            num_groups=16, num_channels=out_channels, eps=eps
        )
        self.conv1 = torch.nn.Linear(
            in_features=out_channels, out_features=out_channels
        )

        self.skip = None
        if out_channels != in_channels:
            self.skip = torch.nn.Linear(
                in_features=in_channels, out_features=out_channels
            )
        self.skip_scale = 0.5**0.5

    def forward(self, x, emb):
        silu = torch.nn.functional.silu
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb)

        x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        return x


class FNet(torch.nn.Module):
    def __init__(self, dim_input, dim_cond, dim_embedding=512, n_layers=3):
        super().__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_embedding),
            # torch.nn.SiLU(),
            # torch.nn.Linear
        )
        self.cond_layer = torch.nn.Linear(dim_cond, dim_embedding)

        self.embedding_map = torch.nn.Sequential(
            torch.nn.Linear(dim_embedding, 2 * dim_embedding),
            torch.nn.GroupNorm(num_groups=16, num_channels=2 * dim_embedding),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * dim_embedding, dim_embedding),
        )
        self.res_layers = torch.nn.ModuleList(
            [
                _FBlock(
                    in_channels=dim_embedding // 2**i,
                    out_channels=dim_embedding // 2 ** (i + 1),
                    emb_channels=dim_embedding,
                    dropout=0.1,
                )
                for i in range(n_layers)
            ]
        )
        self.final_layer = torch.nn.Sequential(
            torch.nn.GroupNorm(
                num_groups=16, num_channels=dim_embedding // 2 ** (n_layers)
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_embedding // 2 ** (n_layers), dim_input, bias=False),
        )
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
        return self.final_layer(theta_emb).reshape(*theta.shape)  # - theta


# Noisy network for experiments of section 4.2


class EpsilonNet(torch.nn.Module):
    def __init__(self, DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * DIM + 1, 5 * DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(5 * DIM, DIM),
            torch.nn.Tanh(),
        )

    def forward(self, theta, x, t):
        return self.net(torch.cat((theta, x, t), dim=-1))


class FakeFNet(torch.nn.Module):
    def __init__(self, real_eps_fun, eps_net, eps_net_max):
        r"""Fake score network that returns the real score plus a perturbation

        Args:
            real_eps_fun (callable): function that returns the real score network (analytic or trained)
            eps_net (torch.nn.Module): perturbation network (randomly initialized, not trained)
            eps_net_max (float): scaling factor for the perturbation
        """
        super().__init__()
        self.real_eps_fun = real_eps_fun
        self.eps_net = eps_net
        self.eps_net_max = eps_net_max

    def forward(self, theta, x, t):
        if len(t.shape) == 0:
            t = t[None, None].repeat(theta.shape[0], 1)
        real_eps = self.real_eps_fun(theta, x, t)
        perturb = self.eps_net(theta, x, t)
        return real_eps + self.eps_net_max * perturb