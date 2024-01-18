import sys

sys.path.append("tasks/jrnnm/")

import argparse
import os
import torch

from functools import partial
from nse import NSE, NSELoss
from sm_utils import train
from tasks.jrnnm.summary import summary_JRNMM
from tasks.jrnnm.prior import prior_JRNMM
from tasks.jrnnm.simulator import simulator_JRNMM
from tqdm import tqdm
from zuko.nn import MLP

from debug_learned_uniform import diffused_tall_posterior_score, euler_sde_sampler
from vp_diffused_priors import get_vpdiff_uniform_score

PATH_EXPERIMENT = "results/jrnnm/"
THETA_DIM = 4  # C, mu, sigma, gain


def run_train_sgm(
    data_train,
    n_epochs,
    batch_size,
    lr,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Prepare training data
    theta_train, x_train = data_train["theta"], data_train["x"]
    # normalize theta
    theta_train_norm = (theta_train - theta_train.mean(dim=0)) / theta_train.std(dim=0)
    # normalize x
    x_train = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
    # dataset for dataloader
    data_train = torch.utils.data.TensorDataset(theta_train_norm.to(device), x_train.to(device))

    # Score network
    score_network = NSE(
        theta_dim=THETA_DIM,
        x_dim=x_train.shape[-1],
        hidden_features=[128, 256, 128],
    ).to(device)

    # Train score network
    print(
        "=============================================================================="
    )
    print(
        f"Training score network: n_train = {theta_train.shape[0]}, n_epochs = {n_epochs}."
    )
    # print()
    # print(f"n_max: {n_max}, masked: {masked}, prior_score: {prior_score}")
    print(
        f"============================================================================="
    )
    print()

    # Train Score Network
    avg_score_net, train_losses = train(
        score_network,
        dataset=data_train,
        loss_fn=NSELoss(score_network),
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        track_loss=True,
    )
    score_network = avg_score_net.module

    # Save Score Network
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(
        score_network,
        save_path + f"score_network.pkl",
    )
    torch.save(
        train_losses,
        save_path + f"train_losses.pkl",
    )


def run_sample_sgm(
    theta_true,
    context,
    nsamples,
    steps,  # number of ddim steps
    score_network,
    theta_train,
    x_train,
    prior,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    n_obs = context.shape[0]

    # normalize context
    context_norm = (context - x_train.mean(dim=0)) / x_train.std(dim=0)

    # normalize prior
    low_norm = (prior.low - theta_train.mean(dim=0)) / theta_train.std(dim=0) * 2
    high_norm = (prior.high - theta_train.mean(dim=0)) / theta_train.std(dim=0) * 2
    prior_score_fn_norm = get_vpdiff_uniform_score(low_norm.to(device), high_norm.to(device), score_network.to(device))

    # define score function for tall posterior
    score_fn = partial(
        diffused_tall_posterior_score,
        prior_score_fn=prior_score_fn_norm,
        x_obs=context_norm.to(device),
        nse=score_network.to(device),
    )

    print("=======================================================================")
    print(
        f"Sampling from the approximate posterior at {theta_true}: nsamples = {nsamples}."
    )
    print(f"======================================================================")
    print()
    print(f"Using EULER sampler.")
    print()

    # sample from tall posterior
    (
        samples,
        all_samples,
        gradlogL,
        lda,
        posterior_scores,
        means_posterior_backward,
        sigma_posterior_backward,
    ) = euler_sde_sampler(
        score_fn, nsamples, dim_theta=THETA_DIM, beta=score_network.beta, device=device, debug=True
    )

    results_dict = {
        "all_theta_learned": all_samples,
        "gradlogL": gradlogL,
        "lda": lda,
        "posterior_scores": posterior_scores,
        "means_posterior_backward": means_posterior_backward,
        "sigma_posterior_backward": sigma_posterior_backward,
    }

    # unnormalize
    samples = samples.detach().cpu()
    samples = samples * theta_train.std(dim=0) + theta_train.mean(dim=0)

    # save  results
    save_path += f"euler_steps_{steps}_new/"
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(
        samples,
        save_path + f"posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}.pkl",
    )
    torch.save(
        results_dict,
        save_path + f"results_dict_{theta_true.tolist()}_n_obs_{n_obs}.pkl",
    )


if __name__ == "__main__":
    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submitit",
        action="store_true",
        help="whether to use submitit for running the job",
    )
    parser.add_argument(
        "--run", type=str, default="train", choices=["train", "sample"], help="run type"
    )
    parser.add_argument(
        "--n_train", type=int, default=50_000, help="number of training data samples"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for training"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1000,
        help="number of samples from the approximate posterior",
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="number of steps for ddim sampler"
    )
    parser.add_argument(
        "--n_obs", type=int, default=1, help="number of context observations for sampling"
    )

    # Parse Arguments
    args = parser.parse_args()

    # Define Experiment Path
    save_path = PATH_EXPERIMENT + f"n_train_{args.n_train}_n_epochs_{args.n_epochs}_lr_{args.lr}/"
    os.makedirs(save_path, exist_ok=True)

    print()
    print("save_path: ", save_path)
    print("CUDA available: ", torch.cuda.is_available())
    print()

    # SBI Task: prior and simulator
    prior = prior_JRNMM()
    simulator = partial(simulator_JRNMM, input_parameters=["C", "mu", "sigma", "gain"])

    # Summary features
    summary_extractor = summary_JRNMM()
    # let's use the *log* power spectral density
    summary_extractor.embedding.net.logscale = True

    # Simulate Training Data
    filename = PATH_EXPERIMENT + f"dataset_n_train_50000.pkl"
    if os.path.exists(filename):
        dataset_train = torch.load(filename)
        theta_train = dataset_train["theta"][: args.n_train]
        x_train = dataset_train["x"][: args.n_train]
    else:
        theta_train = prior.sample((args.n_train,))
        x_train = simulator(theta_train)
        x_train = summary_extractor(x_train)  # (n_train, x_dim, 1)
        x_train = x_train[:, :, 0]  # (n_train, x_dim)

        dataset_train = {
            "theta": theta_train, "x": x_train
        }
        torch.save(dataset_train, filename)

    if args.run == "train":
        run_fn = run_train_sgm
        kwargs_run = {
            "data_train": dataset_train,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "save_path": save_path,
        }
    elif args.run == "sample":
        # Reference parameter and observations
        theta_true = torch.tensor([135.0, 220.0, 2000.0, 0.0])
        filename = PATH_EXPERIMENT + f"x_obs_100_{theta_true.tolist()}.pkl"
        if os.path.exists(filename):
            x_obs_100 = torch.load(filename)
            context = x_obs_100[:args.n_obs]
        else:
            x_obs_100 = torch.cat(
                [simulator(theta_true).reshape(1, -1) for _ in tqdm(range(100))], dim=0
            )
            torch.save(x_obs_100, filename)

        # Trained Score network
        score_network = torch.load(
            save_path + f"score_network_new.pkl",
            map_location=torch.device("cpu"),
        )

        run_fn = run_sample_sgm
        kwargs_run = {
            "theta_true": theta_true,
            "context": context,
            "nsamples": args.nsamples,
            "score_network": score_network,
            "steps": args.steps,
            "theta_train": theta_train,  # for (un)normalization
            "x_train": x_train,  # for (un)normalization
            "prior": prior, # for score function
            "save_path": save_path,
        }

    if not args.submitit:
        run_fn(**kwargs_run)
    else:
        import submitit

        # function for submitit
        def get_executor_marg(job_name, timeout_hour=60, n_cpus=40):
            executor = submitit.AutoExecutor(job_name)
            executor.update_parameters(
                timeout_min=180,
                slurm_job_name=job_name,
                slurm_time=f"{timeout_hour}:00:00",
                slurm_additional_parameters={
                    "ntasks": 1,
                    "cpus-per-task": n_cpus,
                    "distribution": "block:block",
                    # "partition": "parietal",
                },
            )
            return executor

        # subit job
        executor = get_executor_marg(f"_{args.run}_lensing_sgm")
        # launch batches
        with executor.batch():
            print("Submitting jobs...", end="", flush=True)
            tasks = []
            tasks.append(executor.submit(run_fn, **kwargs_run))
