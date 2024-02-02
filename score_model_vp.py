import math
import torch
from dataclasses import dataclass
from math import log
from typing import List, Tuple

import torch
import tqdm
from torch import device


class ScoreModel(torch.nn.Module):

    def __init__(self, net, alphas_cumprod):
        super(ScoreModel, self).__init__()
        self.net = net
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    def score(self, theta, x, t):
        return - self.net(theta, x, t) / ((1 - self.alphas_cumprod[t])**.5)

    def forward(self, theta, x, t):
        rescaled_input = (theta/self.alphas_cumprod[t]**.5)
        return self.net(rescaled_input, x, t)


def generate_coefficients_ddim(
        alphas_cumprod,
        time_step,
        prev_time_step,
        eta):
    alphas_cumprod_t_1 = alphas_cumprod[prev_time_step] if prev_time_step >= 0 else 1
    alphas_cumprod_t = alphas_cumprod[time_step]

    noise = eta * (((1 - alphas_cumprod_t_1) / (1 - alphas_cumprod_t)) * (1 - alphas_cumprod_t / alphas_cumprod_t_1)) ** .5

    coeff_sample = (alphas_cumprod_t_1 / alphas_cumprod_t) ** .5
    coeff_score = ((1 - alphas_cumprod_t_1 - noise ** 2) ** .5) - coeff_sample * ((1 - alphas_cumprod_t)**.5)

    return noise, coeff_sample, coeff_score


def ddim_marginal_logprob(
                  x0: torch.Tensor,
                  alphas_cumprod: List[float],
                  timesteps: List[int],
                  score_model: ScoreModel,
                  n_samples: int,
                  eta: float = 1) -> torch.Tensor:
    """
    Computes the log marginal of x0 sampled from ddim.

    steps: 1- sample a path from the real backward process
    conditionned on x0, see eq. (7)
    and compute its logprob
    2- compute the logprob of the same path under the ddim path log_prob

    output:
    :log_weights: log ratio, which corresponds to the estimate of the log marginal when
    one sample is used
    :bwd: forward samples of DDIM, conditionned on the real x0
    """
    dim_range = tuple(range(2,x0.dim() + 1))
    alpha_T = alphas_cumprod[-1]
    noise_sample = torch.randn((n_samples, *x0.shape))
    x = (alpha_T ** .5) * x0 + (1 - alpha_T) ** .5 * noise_sample
    log_weights = ((noise_sample ** 2).sum(dim_range) / 2) - (x**2).sum(dim_range) / 2
    for prev_time_step, time_step in tqdm.tqdm(zip(timesteps[1:],
                                                   timesteps[:-1])):
        alphas_cumprod_t_1 = alphas_cumprod[prev_time_step] if prev_time_step >= 0 else 1
        alphas_cumprod_t = alphas_cumprod[time_step]
        noise_std, coeff_sample, coeff_score = generate_coefficients_ddim(
            alphas_cumprod=score_model.alphas_cumprod,
            time_step=time_step,
            prev_time_step=prev_time_step,
            eta=eta
        )
        epsilon_predicted = score_model.net(x, time_step)
        mean = coeff_sample * x + coeff_score * epsilon_predicted
        if prev_time_step != 0:
            x = (alphas_cumprod_t_1 ** .5) * x0 \
                + (1 - alphas_cumprod_t_1 - noise_std ** 2)**.5 \
                * (x - (alphas_cumprod_t ** .5) * x0) / ((1 - alphas_cumprod_t) ** .5)
            noise_sample = torch.randn_like(x)
            x += noise_std * noise_sample
            log_prob_ddim = - ((x - mean)**2).sum(dim_range) / (2 * noise_std**2)
            log_prob_fwd_ddim = - (noise_sample ** 2).sum(dim_range) / 2
            log_weights += log_prob_ddim - log_prob_fwd_ddim
        else:
            log_prob_ddim = - ((x0 - mean)**2).sum(dim_range) / (2 * noise_std**2)
            log_weights += log_prob_ddim
    return log_weights.logsumexp(0) - log(n_samples)


def ddim_parameters(x: torch.Tensor,
                    score_model: ScoreModel,
                    t: float,
                    t_prev: float,
                    eta: float,) -> Tuple[torch.Tensor, torch.Tensor]:
    noise, coeff_sample, coeff_score = generate_coefficients_ddim(
        alphas_cumprod=score_model.alphas_cumprod.to(x.device),
        time_step=t,
        prev_time_step=t_prev,
        eta=eta
    )
    epsilon_predicted = score_model.net(x, t)
    mean = coeff_sample * x + coeff_score * epsilon_predicted.to(x.device)

    return mean, noise

def ddim_sampling(initial_noise_sample: torch.Tensor,
                  timesteps: List[int],
                  score_model: ScoreModel,
                  eta: float = 1,
                  disable: bool = True) -> torch.Tensor:
    '''
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    '''
    sample = initial_noise_sample
    for prev_time_step, time_step in tqdm.tqdm(zip(timesteps[1:],
                                                   timesteps[:-1]),
                                               disable=disable):
        mean, noise = ddim_parameters(x=sample,
                                      score_model=score_model,
                                      t=time_step,
                                      t_prev=prev_time_step,
                                      eta=eta)
        sample = mean + noise * torch.randn_like(mean)
    return sample

def ddim_trajectory(initial_noise_sample: torch.Tensor,
                  timesteps: List[int],
                  score_model: ScoreModel,
                  eta: float = 1) -> torch.Tensor:
    '''
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    '''
    sample = initial_noise_sample
    samples = sample.unsqueeze(0)
    for prev_time_step, time_step in tqdm.tqdm(zip(timesteps[1:],
                                                   timesteps[:-1])):
        mean, noise = ddim_parameters(x=sample,
                                      score_model=score_model,
                                      t=time_step,
                                      t_prev=prev_time_step,
                                      eta=eta)
        sample = mean + noise * torch.randn_like(mean)
        samples = torch.cat([samples, sample.unsqueeze(0)])
    return samples


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[x]


class FNet(torch.nn.Module):

    def     __init__(self,
                     dim_input,
                     dim_cond,
                     dim_embedding=512,
                     n_layers=3):
        super().__init__()
        self.input_layer = torch.nn.Sequential(torch.nn.Linear(dim_input, dim_embedding),
                                               torch.nn.BatchNorm1d(dim_embedding),
                                               torch.nn.LeakyReLU(.2))
        self.cond_layer = torch.nn.Sequential(torch.nn.Linear(dim_cond, dim_embedding),
                                              torch.nn.BatchNorm1d(dim_embedding),
                                              torch.nn.LeakyReLU(.2))
        def res_layer_maker(dim_in, dim_out):
            return torch.nn.Sequential(torch.nn.Linear(dim_in, 2*dim_out),
                                       torch.nn.BatchNorm1d(2*dim_out),
                                       torch.nn.LeakyReLU(.2),
                                       torch.nn.Linear(2 * dim_out, 2 * dim_out),
                                       torch.nn.BatchNorm1d(2 * dim_out),
                                       torch.nn.LeakyReLU(.2),
                                       torch.nn.Linear(2 * dim_out, dim_out),
                                       torch.nn.BatchNorm1d(dim_out),
                                       torch.nn.LeakyReLU(.2),
                                       )
        self.res_layers = torch.nn.ModuleList([res_layer_maker(dim_embedding, dim_embedding, emb_chan) for i in range(n_layers)])
        self.final_layer = torch.nn.Linear(dim_embedding, dim_input)
        self.time_embedding = PositionalEncoding(d_model=dim_embedding)

    def forward(self, theta, x, t):
        if isinstance(t, int):
            t = torch.tensor([t], device=theta.device)
        theta_emb = self.input_layer(theta)
        x_emb = self.cond_layer(x)
        t_emb = self.time_embedding(t.long())
        theta_emb = theta_emb.reshape(-1, theta_emb.shape[-1])
        x_emb = x_emb.reshape(-1, x_emb.shape[-1])
        t_emb = t_emb.reshape(-1, t_emb.shape[-1])
        emb = t_emb + theta_emb + x_emb
        for lr in self.res_layers:
            emb = lr(emb) + emb
        return self.final_layer(emb).reshape(*theta.shape) #- theta
