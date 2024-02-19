import torch
import numpy as np

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from tqdm import tqdm


# setup all parameters
JRNMM_parameters = {}
JRNMM_parameters["A"] = 3.25
JRNMM_parameters["B"] = 22.0
JRNMM_parameters["a"] = 100.0
JRNMM_parameters["b"] = 50.0
JRNMM_parameters["C"] = 135.0
JRNMM_parameters["C1"] = 1.00 * JRNMM_parameters["C"]
JRNMM_parameters["C2"] = 0.80 * JRNMM_parameters["C"]
JRNMM_parameters["C3"] = 0.25 * JRNMM_parameters["C"]
JRNMM_parameters["C4"] = 0.25 * JRNMM_parameters["C"]
JRNMM_parameters["vmax"] = 5
JRNMM_parameters["v0"] = 6
JRNMM_parameters["r"] = 0.56
JRNMM_parameters["mu"] = 220.0
JRNMM_parameters["s4"] = 0.01
JRNMM_parameters["sigma"] = 2000.0
JRNMM_parameters["s6"] = 1.00
JRNMM_parameters["gain"] = 0.00  # in db


def simulator_JRNMM(
    theta, input_parameters, t_recording=8, n_extra=0, p_gain=None, n_time_samples=1024
):
    """Define the simulator function

    Parameters
    ----------
    theta : torchtensor, shape (n_trials, dim_theta)
        ndarray of trials.
    n_extra : int
        how many extra observations sharing the same beta should we simulate.
        the minimum is 0, for which the output is simply that with theta.
        n_instances > 0 will generate other outputs with other [C, mu, sigma]
        but the same gain. the first coordinate of the sampled observation is
        the one corresponding to the input theta
    p_gain : torch.distribution
        probability distribution from which to sample the different values of
        [C, mu, sigma] for a given gain (only used when n_extra > 0)

    Returns
    -------
    x : torchtensor shape (n_trials, n_time_samples, 1+n_extra)
        observations for the model with different input parameters

    """

    if theta.ndim == 1:
        return simulator_JRNMM(
            theta.view(1, -1),
            input_parameters,
            t_recording,
            n_extra,
            p_gain,
            n_time_samples,
        )

    x = []
    xextra = []

    # loop over trial samples
    for thetai in tqdm(theta):
        # get parameter dictionary
        JRNMM_parameters_i = JRNMM_parameters.copy()
        # convert to numpy on cpu
        thetai = thetai.detach().clone().to("cpu").numpy().astype(np.float64)

        # get parameters for main observation xi
        for i, p in enumerate(input_parameters):
            JRNMM_parameters_i[p] = thetai[i]
            # save global parameter: gain
            if p == "gain":
                gaini = thetai[i]

        # simulate main observation xi
        xi = simulate_jansen_rit_StrangSplitting(
            t_recording, JRNMM_parameters_i, n_time_samples
        )  # (n_time_samples)
        # center
        xi = xi - np.mean(xi)
        x.append(xi)

        # simulate extra observations xextra
        xextrai = []
        for _ in range(n_extra):
            JRNMM_parameters_j = JRNMM_parameters.copy()
            # sample local parameters for fixed gain
            thetaj = p_gain.sample((1,)).view(-1)
            # convert to numpy on cpu
            thetaj = thetaj.detach().clone().cpu().numpy().astype(np.float64)
            # get parameters for extra observation xj
            for k, p in enumerate(input_parameters):
                # global parameter: gain
                if p == "gain":
                    JRNMM_parameters_j[p] = gaini
                # local parameters
                else:
                    JRNMM_parameters_j[p] = thetaj[k]

            # simulate extra observation xj
            xj = simulate_jansen_rit_StrangSplitting(
                t_recording, JRNMM_parameters_j, n_time_samples
            )  # (n_time_samples)
            # center
            xj = xj - np.mean(xj)

            xextrai.append(xj)
        # stack extra observations into a tensor
        xextrai = theta.new(xextrai).T  # (n_time_samples, n_extra)

        xextra.append(xextrai)

    x = theta.new(x).unsqueeze(-1)  # (n_trials, n_time_samples, 1)
    if n_extra > 0:
        # stack extra observations into a tensor
        xextra = torch.stack(xextra)  # (n_trials, n_time_samples, n_extra)
        # concatenate main and extra observations
        x = torch.cat([x, xextra], dim=-1)  # (n_trials, n_time_samples, 1+n_extra)
    return x  # (n_trials, 1+n_extra, n_time_samples)


def simulate_jansen_rit_StrangSplitting(
    trecording, parameters, n_time_samples, x0=None
):
    if x0 is None:
        x0 = np.random.randn(6)

    # import the sdbmpABC package which had previously been installed on R
    sdbmp = importr("sdbmsABC")
    rchol = robjects.r["chol"]
    rt = robjects.r["t"]

    burnin = 2.0
    T = trecording  # time interval for the datasets
    h = 1 / n_time_samples  # time step (corresponds to Delta)  ##changed

    # theta_true: parameters used to simulate the reference data
    sigma = float(parameters["sigma"])
    mu = float(parameters["mu"])
    C = float(parameters["C"])

    # fixed model coefficients
    A = float(parameters["A"])
    B = float(parameters["B"])
    a = float(parameters["a"])
    b = float(parameters["b"])
    v0 = float(parameters["v0"])
    vmax = float(parameters["vmax"])
    r = float(parameters["r"])
    s4 = float(parameters["s4"])
    s6 = float(parameters["s6"])
    gain_db = float(parameters["gain"])
    gain_abs = 10 ** (gain_db / 10)

    startv = robjects.FloatVector(list(x0))
    grid = robjects.FloatVector(list(np.arange(0, T + burnin, h)))
    M1 = sdbmp.exp_matJR(h, a, b)
    M2 = rt(
        rchol(sdbmp.cov_matJR(h, robjects.FloatVector([0, 0, 0, s4, sigma, s6]), a, b))
    )
    X = gain_abs * np.array(
        sdbmp.Splitting_JRNMM_output_Cpp(
            h, startv, grid, M1, M2, mu, C, A, B, a, b, v0, r, vmax
        )
    )
    X = X[int(burnin / h) :]
    X = X[::8]

    return X


def get_ground_truth(meta_parameters, input_parameters, p_gain=None):
    "Take the parameters dict as input and output the observed data."

    # ground truth observation
    signal = simulator_JRNMM(
        theta=meta_parameters["theta"],
        input_parameters=input_parameters,
        t_recording=meta_parameters["t_recording"],
        n_extra=meta_parameters["n_extra"],
        p_gain=p_gain,
    )

    # get the ground_truth observation data
    ground_truth = {}
    ground_truth["theta"] = meta_parameters["theta"].clone().detach()
    ground_truth["observation"] = signal

    return ground_truth


if __name__ == "__main__":
    from prior import prior_JRNMM

    meta_parameters = {}
    prior = prior_JRNMM()
    theta = prior.sample((128,))
    x = simulator_JRNMM(
        theta,
        input_parameters=["C", "sigma", "mu", "gain"],
    )

    ## try aggregating before feeding to snpe to try different nextra between training and inference
    from summary import summary_JRNMM

    summary_extractor = summary_JRNMM()

    # let's use the log power spectral density instead
    summary_extractor.embedding.net.logscale = True
    x = summary_extractor(x)  # n_batch, n_embed, 1+nextra
    print(x.shape)
    # xobs = x[:, :, 0].unsqueeze(-1)  # n_batch, n_embed, 1
    # print(xobs.shape)
    # xagg = x[:, :, 1:].mean(dim=-1).unsqueeze(-1)  # n_batch, n_embed, 1
    # print(xagg.shape)
    # x = torch.cat([xobs, xagg], dim=2)  # n_batch, n_embed, 2
    x = x.permute(0, 2, 1)  # n_batch, 2, n_embed

    print(x.shape)
