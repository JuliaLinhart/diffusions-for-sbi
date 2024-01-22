import sys

sys.path.append("tasks/sbibm/")

import argparse
import os
import torch
import sbibm
import pyro

from functools import partial
from nse import NSE, NSELoss
from sm_utils import train

from tqdm import tqdm
from zuko.nn import MLP

from tasks.sbibm.data_generators import get_task
from debug_learned_gaussian import diffused_tall_posterior_score, euler_sde_sampler
from vp_diffused_priors import get_vpdiff_gaussian_score

PATH_EXPERIMENT = "results/sbibm/"
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

N_TRAIN_LIST = [10000, 30000] #1000, 3000, 
N_OBS_LIST = [1, 8, 14, 22, 30]


def run_train_sgm(
    theta_train,
    x_train,
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
    # normalize theta
    theta_train_norm = (theta_train - theta_train.mean(dim=0)) / theta_train.std(dim=0)
    # normalize x
    x_train_norm = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
    # replace nan by 0 (due to std in sir for n_train = 1000)
    x_train_norm = torch.nan_to_num(x_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    # dataset for dataloader
    data_train = torch.utils.data.TensorDataset(theta_train_norm.to(device), x_train_norm.to(device))

    # Score network
    score_network = NSE(
        theta_dim=theta_train.shape[-1],
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
    avg_score_net, train_losses, val_losses = train(
        score_network,
        dataset=data_train,
        loss_fn=NSELoss(score_network),
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        track_loss=True,
        validation_split=0.2,
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
        {"train_losses": train_losses, "val_losses": val_losses},
        save_path + f"train_losses.pkl",
    )


def run_sample_sgm(
    num_obs,
    context,
    nsamples,
    steps,  # number of ddim steps
    score_network,
    theta_train_mean,
    theta_train_std,
    x_train_mean,
    x_train_std,
    prior,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    n_obs = context.shape[0]

    # normalize context
    context_norm = (context - x_train_mean) / x_train_std
    # replace nan by 0 (due to std in sir for n_train = 1000)
    context_norm = torch.nan_to_num(context_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # normalize prior
    loc_norm = (prior.loc - theta_train_mean) / theta_train_std
    cov_norm = torch.diag(1/theta_train_std) @ prior.covariance_matrix @ torch.diag(1/theta_train_std)
    prior_norm = torch.distributions.MultivariateNormal(loc_norm.to(device), cov_norm.to(device))
    prior_score_fn_norm = get_vpdiff_gaussian_score(loc_norm.to(device), cov_norm.to(device), score_network.to(device))

    # define score function for tall posterior
    score_fn = partial(
        diffused_tall_posterior_score,
        prior=prior_norm,
        prior_score_fn=prior_score_fn_norm,
        x_obs=context_norm.to(device),
        nse=score_network.to(device),
    )

    print("=======================================================================")
    print(
        f"Sampling from the approximate posterior for observation {num_obs}: n_obs = {n_obs}, nsamples = {nsamples}."
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
        score_fn, nsamples, dim_theta=theta_train_mean.shape[-1], beta=score_network.beta, device=device, debug=True
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
    samples = samples * theta_train_std + theta_train_mean

    # save  results
    save_path += f"euler_steps_{steps}/"
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(
        samples,
        save_path + f"posterior_samples_{num_obs}_n_obs_{n_obs}.pkl",
    )
    torch.save(
        results_dict,
        save_path + f"results_dict_{num_obs}_n_obs_{n_obs}.pkl",
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
        "--task", type=str, default="gaussian_linear", choices=["gaussian_linear", "gaussian_mixture", "lotka_volterra", "sir"], help="task name"
    )
    parser.add_argument(
        "--run", type=str, default="train", choices=["train", "sample", "sample_all"], help="run type"
    )
    parser.add_argument(
        "--n_train", type=int, default=50_000, help="number of training data samples (1000, 3000, 10000, 30000 in [Geffner et al. 2023])"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for training (1e-3/1e-4 in [Geffner et al. 2023]))"
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
    parser.add_argument(
        "--num_obs", type=int, default=1, help="number of the observation in sbibm"
    )

    # Parse Arguments
    args = parser.parse_args()

    # Define task path
    task_path = PATH_EXPERIMENT + f"{args.task}/"

    def run(n_train=args.n_train, num_obs=args.num_obs, n_obs=args.n_obs, run_type=args.run):

        # Define Experiment Path
        save_path = task_path + f"n_train_{n_train}_n_epochs_{args.n_epochs}_lr_{args.lr}/"
        os.makedirs(save_path, exist_ok=True)

        print()
        print("save_path: ", save_path)
        print("CUDA available: ", torch.cuda.is_available())
        print()

        # SBI Task: prior and simulator
        task = get_task(args.task)
        prior = task.get_prior()
        simulator = task.get_simulator()

        # Simulate Training Data
        filename = task_path + f"dataset_n_train_50000.pkl"
        if os.path.exists(filename):
            print(f"Loading training data from {filename}")
            dataset_train = torch.load(filename)
            theta_train = dataset_train["theta"][: n_train]
            x_train = dataset_train["x"][: n_train]
        else:
            theta_train = prior(n_train)
            x_train = simulator(theta_train)

            dataset_train = {
                "theta": theta_train, "x": x_train
            }
            torch.save(dataset_train, filename)
        # extract training data for given n_train
        theta_train, x_train = theta_train[: n_train], x_train[: n_train]
        # compute mean and std of training data
        theta_train_mean, theta_train_std = theta_train.mean(dim=0), theta_train.std(dim=0)
        x_train_mean, x_train_std = x_train.mean(dim=0), x_train.std(dim=0)
        means_stds_dict = {
            "theta_train_mean": theta_train_mean,
            "theta_train_std": theta_train_std,
            "x_train_mean": x_train_mean,
            "x_train_std": x_train_std,
        }
        torch.save(means_stds_dict, save_path + f"train_means_stds_dict.pkl")

        if run_type == "train":
            run_fn = run_train_sgm
            kwargs_run = {
                "theta_train": theta_train,
                "x_train": x_train,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "save_path": save_path,
            }
        elif run_type == "sample":
            # Reference parameter and observations
            observation_list = [
                task.get_observation(num_observation=n) for n in NUM_OBSERVATION_LIST
            ]
            theta_true_list = [
                task.get_true_parameters(num_observation=n) for n in NUM_OBSERVATION_LIST
            ]
            
            theta_true = theta_true_list[num_obs-1]
            x_obs = observation_list[num_obs-1]

            filename = task_path + f"x_obs_100_num_{num_obs}.pkl"
            if os.path.exists(filename):
                x_obs_100 = torch.load(filename)
            else:
                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in tqdm(range(100))], dim=0
                )
                torch.save(x_obs_100, filename)
            context = x_obs_100[:n_obs]

            # Trained Score network
            score_network = torch.load(
                save_path + f"score_network.pkl",
                map_location=torch.device("cpu"),
            )

            # Mean and std of training data
            means_stds_dict = torch.load(save_path + f"train_means_stds_dict.pkl")
            theta_train_mean = means_stds_dict["theta_train_mean"]
            theta_train_std = means_stds_dict["theta_train_std"]
            x_train_mean = means_stds_dict["x_train_mean"]
            x_train_std = means_stds_dict["x_train_std"]

            run_fn = run_sample_sgm
            kwargs_run = {
                "num_obs": num_obs,
                "context": context,
                "nsamples": args.nsamples,
                "score_network": score_network,
                "steps": args.steps,
                "theta_train_mean": theta_train_mean,  # for (un)normalization
                "theta_train_std": theta_train_std,  # for (un)normalization
                "x_train_mean": x_train_mean,  # for (un)normalization
                "x_train_std": x_train_std,  # for (un)normalization
                "prior": task.prior_dist, # for score function
                "save_path": save_path,
            }
                    
        run_fn(**kwargs_run)

    if args.run == "sample_all":
        for n_train in N_TRAIN_LIST:
            for num_obs in NUM_OBSERVATION_LIST:
                for n_obs in N_OBS_LIST:
                    run(n_train=n_train, num_obs=num_obs, n_obs=n_obs, run_type="sample")
    else:
        run()
    