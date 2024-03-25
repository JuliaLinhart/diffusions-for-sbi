from functools import partial

import torch
from scipy.signal import welch
from torch import nn

torch.autograd.set_detect_anomaly(True)


# Welch's method for power spectral density estimation
#
# TODO: clean this up with a full-pytorch implementation
class PowerSpecDens(nn.Module):
    def __init__(self, nbins=128, logscale=False):
        super().__init__()
        self.nbins = nbins
        self.logscale = logscale

    def forward(self, X):
        if X.ndim == 1:
            X = X.view(1, -1)
        else:
            X = X.view(len(X), -1)

        ntrials = X.shape[0]
        ns = X.shape[1]
        nfreqs = 2 * (self.nbins - 1) + 1
        windows = [
            wi + torch.arange(nfreqs)
            for wi in range(0, ns - nfreqs + 1, int(nfreqs / 2))
        ]
        S = []
        for i in range(ntrials):
            Xi = X[i, :].view(-1)
            Si = []
            for w in windows:
                Si.append(torch.abs(torch.fft.rfft(Xi[w])) ** 2)
            Si = torch.mean(torch.stack(Si), axis=0)
            if self.logscale:
                Si = torch.log10(Si)
            S.append(Si)
        return torch.stack(S)


class FourierLayer(nn.Module):
    def __init__(self, nfreqs=129):
        super().__init__()
        self.nfreqs = nfreqs

    def forward(self, X):
        if X.ndim == 1:
            X = X.view(1, -1)
        else:
            X = X.view(len(X), -1)
        Xfourier = []
        Xnumpy = X.clone().detach().to("cpu").numpy()
        _, Xfourier = welch(Xnumpy, nperseg=2 * (self.nfreqs - 1))
        return X.new_tensor(Xfourier)


class fourier_embedding(nn.Module):
    def __init__(self, d_out=1, n_time_samples=1024, plcr=True, logscale=False):
        super().__init__()
        self.d_out = d_out
        if plcr:  # use my implementation for power spectral density
            self.net = PowerSpecDens(nbins=d_out, logscale=logscale)
        else:  # use implementation from scipy.signal.welch (much slower)
            self.net = FourierLayer(nfreqs=d_out)

    def forward(self, x):
        y = self.net(x.view(1, -1))
        return y


class identity(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

    def forward(self, x):
        return x


class debug_embedding(nn.Module):
    def __init__(self, d_out=1, n_time_samples=1024):
        super().__init__()
        self.net = nn.Linear(in_features=n_time_samples, out_features=d_out)

    def forward(self, x):
        y = self.net(x.view(1, -1))
        # A = torch.ones(33, 1024)
        # y = A * x
        return y


emb_dict = {}
emb_dict["Fourier"] = partial(fourier_embedding, plcr=True)
emb_dict["Debug"] = debug_embedding


class summary_JRNMM(nn.Module):
    def __init__(
        self, n_extra=0, d_embedding=33, n_time_samples=1024, type_embedding="Fourier"
    ):
        super().__init__()
        self.n_extra = n_extra
        self.n_time_samples = n_time_samples
        self.d_embedding = d_embedding
        self.embedding = emb_dict[type_embedding](
            d_out=d_embedding, n_time_samples=n_time_samples
        )

    def forward(self, x, n_extra=None):
        self.embedding.eval()

        # set n_extra to default if not specified
        if n_extra is None:
            n_extra = self.n_extra

        # add batch dimension if needed
        if x.ndim == 2:
            x = x[None, :, :]  # (1, 1024, n_extra + 1)

        # compute summary statistics
        y = []
        for xi in x:
            yi = [self.embedding(xi[:, 0])]
            for j in range(n_extra):
                yi.append(self.embedding(xi[:, j + 1]))
            yi = torch.cat(yi).T
            y.append(yi)
        y = torch.stack(y)
        return y


if __name__ == "__main__":
    n_extra = 9
    d_embedding = 33
    x = torch.randn(3, 1024, 10)
    net = summary_JRNMM(
        n_extra=n_extra, d_embedding=d_embedding, type_embedding="Fourier"
    )
    y = net(x)
    print(y.shape)
