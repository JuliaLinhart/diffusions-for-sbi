import sys

sys.path.append("tasks/toy_examples/")

import argparse
import os
import torch
import time

from functools import partial
from nse import NSE, NSELoss
from sm_utils import train
from torch.func import vmap
from zuko.nn import MLP

from tqdm import tqdm

from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler
from vp_diffused_priors import get_vpdiff_gaussian_score

PATH_EXPERIMENT = "results/gaussian/"

N_OBS_LIST = [1, 8, 14, 22, 30]
DIM_LIST = [2, 4, 8, 16, 32]

COV_MODES = ["GAUSS", "JAC"]


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
        device = "cuda:1"

    # Prepare training data
    # normalize theta
    theta_train_norm = (theta_train - theta_train.mean(dim=0)) / theta_train.std(dim=0)
    # normalize x
    x_train_norm = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
    # dataset for dataloader
    data_train = torch.utils.data.TensorDataset(
        theta_train_norm.to(device), x_train_norm.to(device)
    )

    # Score network
    # embedding nets
    theta_dim = theta_train.shape[-1]
    x_dim = x_train.shape[-1]
    theta_embedding_net = MLP(theta_dim, 32, [64, 64, 64])
    x_embedding_net = MLP(x_dim, 32, [64, 64, 64])
    score_network = NSE(
        theta_dim=theta_dim,
        x_dim=x_dim,
        embedding_nn_theta=theta_embedding_net,
        embedding_nn_x=x_embedding_net,
        hidden_features=[128, 256, 128],
        freqs=32,
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
    context,
    nsamples,
    steps,  # number of ddim steps
    score_network,
    theta_train_mean,
    theta_train_std,
    x_train_mean,
    x_train_std,
    prior,
    cov_mode,
    langevin=False,
    clip=False,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:1"

    n_obs = context.shape[0]

    # normalize context
    context_norm = (context - x_train_mean) / x_train_std
    # replace nan by 0 (due to std in sir for n_train = 1000)
    context_norm = torch.nan_to_num(context_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # normalize prior
    loc_norm = (prior.loc - theta_train_mean) / theta_train_std
    cov_norm = (
        torch.diag(1 / theta_train_std)
        @ prior.covariance_matrix
        @ torch.diag(1 / theta_train_std)
    )
    prior_norm = torch.distributions.MultivariateNormal(
        loc_norm.to(device), cov_norm.to(device)
    )
    prior_score_fn_norm = get_vpdiff_gaussian_score(
        loc_norm.to(device), cov_norm.to(device), score_network.to(device)
    )

    print("=======================================================================")
    print(
        f"Sampling from the approximate posterior: n_obs = {n_obs}, nsamples = {nsamples}."
    )
    print(f"======================================================================")

    if langevin:
        print()
        print(f"Using LANGEVIN sampler, clip = {clip}.")
        print()

        ext = ""
        if "clip":
            theta_clipping_range = (-3, 3)
            ext = "_clip"
        start_time = time.time()
        samples = score_network.predictor_corrector(
            (nsamples,),
            x=context_norm.to(device),
            steps=400,
            prior_score_fun=prior_score_fn_norm,
            eta=1,
            corrector_lda=0,
            n_steps=5,
            r=0.5,
            predictor_type="id",
            verbose=True,
            theta_clipping_range=theta_clipping_range,
        ).cpu()
        time_elapsed = time.time() - start_time
        results_dict = None

        save_path += f"langevin_steps_400_5/"
        samples_filename = save_path + f"posterior_samples_n_obs_{n_obs}{ext}.pkl"
        time_filename = save_path + f"time_n_obs_{n_obs}{ext}.pkl"
    else:
        print()
        print(f"Using EULER sampler, cov_mode = {cov_mode}, clip = {clip}.")
        print()

        # estimate cov
        cov_est = vmap(
            lambda x: score_network.ddim(shape=(1000,), x=x, steps=100, eta=0.5),
            randomness="different",
        )(context_norm.to(device))
        cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)

        cov_mode_name = cov_mode
        theta_clipping_range = (None, None)
        if "clip":
            theta_clipping_range = (-3, 3)
            cov_mode_name = cov_mode + "_clip"

        score_fn = partial(
            diffused_tall_posterior_score,
            prior=prior_norm,  # normalized prior
            prior_score_fn=prior_score_fn_norm,  # analytical prior score function
            x_obs=context_norm.to(device),  # observations
            nse=score_network,  # trained score network
            dist_cov_est=cov_est,
            cov_mode=cov_mode,
            # warmup_alpha=0.5 if cov_mode == 'JAC' else 0.0,
            # psd_clipping=True if cov_mode == 'JAC' else False,
            # scale_gradlogL=True,
        )

        # sample from tall posterior
        start_time = time.time()
        (
            samples,
            all_samples,
            # gradlogL,
            # lda,
            # posterior_scores,
            # means_posterior_backward,
            # sigma_posterior_backward,
        ) = euler_sde_sampler(
            score_fn,
            nsamples,
            dim_theta=theta_train_mean.shape[-1],
            beta=score_network.beta,
            device=device,
            debug=False,
            theta_clipping_range=theta_clipping_range,
        )
        time_elapsed = time.time() - start_time  # + time_cov_est

        results_dict = {
            "all_theta_learned": all_samples,
            # "gradlogL": gradlogL,
            # "lda": lda,
            # "posterior_scores": posterior_scores,
            # "means_posterior_backward": means_posterior_backward,
            # "sigma_posterior_backward": sigma_posterior_backward,
        }

        save_path += f"euler_steps_{steps}/"
        samples_filename = (
            save_path + f"posterior_samples_n_obs_{n_obs}_{cov_mode_name}.pkl"
        )
        results_dict_filename = (
            save_path + f"results_dict_n_obs_{n_obs}_{cov_mode_name}.pkl"
        )
        time_filename = save_path + f"time_n_obs_{n_obs}_{cov_mode_name}.pkl"

    # unnormalize
    samples = samples.detach().cpu()
    samples = samples * theta_train_std + theta_train_mean

    # save  results
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(samples, samples_filename)
    torch.save(time_elapsed, time_filename)
    if results_dict is not None:
        torch.save(results_dict, results_dict_filename)


if __name__ == "__main__":
    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submitit",
        action="store_true",
        help="whether to use submitit for running the job",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="dimension of the toy example"
    )
    parser.add_argument(
        "--random_prior",
        action="store_true",
        help="whether to use random prior means and stds",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="train",
        choices=["train", "sample", "train_all", "sample_all"],
        help="run type",
    )
    parser.add_argument(
        "--n_train", type=int, default=10000, help="number of training data samples"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10000, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate for training"
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
        "--n_obs",
        type=int,
        default=1,
        help="number of context observations for sampling",
    )
    parser.add_argument(
        "--cov_mode",
        type=str,
        default="GAUSS",
        choices=COV_MODES,
        help="covariance mode",
    )
    parser.add_argument(
        "--langevin",
        action="store_true",
        help="whether to use langevin sampler (Geffner et al. 2023)",
    )
    parser.add_argument(
        "--clip", action="store_true", help="whether to clip the samples"
    )

    # Parse Arguments
    args = parser.parse_args()

    # Seed
    torch.manual_seed(42)

    def run(dim=args.dim, n_obs=args.n_obs, run_type=args.run):
        # Define task path
        task_path = PATH_EXPERIMENT + f"{dim}d"
        if args.random_prior:
            task_path += "_random_prior"
        task_path += "/"

        # Define Experiment Path
        save_path = (
            task_path + f"n_train_{args.n_train}_n_epochs_{args.n_epochs}_lr_{args.lr}/"
        )
        os.makedirs(save_path, exist_ok=True)

        print()
        print("save_path: ", save_path)
        print("CUDA available: ", torch.cuda.is_available())
        print()

        # SBI Task: prior and simulator
        if args.random_prior:
            torch.manual_seed(42)
            means = torch.rand(dim) * 20 - 10  # between -10 and 10
            torch.manual_seed(42)
            stds = torch.rand(dim) * 25 + 0.1  # between 0.1 and 25.1
            task = Gaussian_Gaussian_mD(dim=dim, means=means, stds=stds)
        else:
            task = Gaussian_Gaussian_mD(dim=dim)
        prior = task.prior
        simulator = task.simulator

        # Simulate Training Data
        filename = task_path + f"dataset_n_train_50000.pkl"
        if os.path.exists(filename):
            print(f"Loading training data from {filename}")
            dataset_train = torch.load(filename)
            theta_train = dataset_train["theta"][: args.n_train]
            x_train = dataset_train["x"][: args.n_train]
        else:
            theta_train = prior.sample((args.n_train,))
            x_train = simulator(theta_train)

            dataset_train = {"theta": theta_train, "x": x_train}
            torch.save(dataset_train, filename)
        # extract training data for given n_train
        theta_train, x_train = theta_train[: args.n_train], x_train[: args.n_train]

        # compute mean and std of training data
        theta_train_mean, theta_train_std = theta_train.mean(dim=0), theta_train.std(
            dim=0
        )
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
            filename = task_path + f"x_obs_100.pkl"
            if os.path.exists(filename):
                x_obs_100 = torch.load(filename)
                theta_true = torch.load(task_path + f"theta_true.pkl")
            else:
                torch.manual_seed(1)
                theta_true = prior.sample()
                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in tqdm(range(100))],
                    dim=0,
                )
                torch.save(theta_true, task_path + f"theta_true.pkl")
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
                "context": context,
                "nsamples": args.nsamples,
                "score_network": score_network,
                "steps": args.steps,
                "theta_train_mean": theta_train_mean,  # for (un)normalization
                "theta_train_std": theta_train_std,  # for (un)normalization
                "x_train_mean": x_train_mean,  # for (un)normalization
                "x_train_std": x_train_std,  # for (un)normalization
                "prior": prior,  # for score function
                "cov_mode": args.cov_mode,
                "langevin": args.langevin,
                "clip": args.clip,
                "save_path": save_path,
            }

        run_fn(**kwargs_run)

    if not args.submitit:
        if args.run == "sample_all":
            for dim in DIM_LIST:
                for n_obs in N_OBS_LIST:
                    run(dim=dim, n_obs=n_obs, run_type="sample")
        elif args.run == "train_all":
            for dim in DIM_LIST:
                run(dim=dim, run_type="train")
        else:
            run()

    else:
        import submitit
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
        executor = get_executor_marg(f"_gaussian_{args.run}_epochs_{args.n_epochs}")
        # launch batches
        with executor.batch():
            print("Submitting jobs...", end="", flush=True)
            tasks = []
            if args.run == "sample_all":
                for dim in DIM_LIST:
                    for n_obs in N_OBS_LIST:
                        tasks.append(
                            executor.submit(
                                run, dim=dim, n_obs=n_obs, run_type="sample"
                            )
                        )
            elif args.run == "train_all":
                for dim in DIM_LIST:
                    tasks.append(executor.submit(run, dim=dim, run_type="train"))
            else:
                tasks.append(executor.submit(run))
