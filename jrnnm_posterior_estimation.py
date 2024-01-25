import sys

sys.path.append("tasks/jrnnm/")

import argparse
import os
import torch

from functools import partial
from nse import NSE, NSELoss
from sm_utils import train

# from tasks.jrnnm.summary import summary_JRNMM
from tasks.jrnnm.prior import prior_JRNMM

# from tasks.jrnnm.simulator import simulator_JRNMM
from tqdm import tqdm
from torch.func import vmap
from zuko.nn import MLP

# from debug_learned_uniform import diffused_tall_posterior_score, euler_sde_sampler
from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler
from vp_diffused_priors import get_vpdiff_uniform_score

# time
import time

PATH_EXPERIMENT = "results/jrnnm/"
N_OBS_LIST = [1, 8, 14, 22, 30]
COV_MODES = ["GAUSS", "JAC", "GAUSS_clip"]


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
    data_train = torch.utils.data.TensorDataset(
        theta_train_norm.to(device), x_train.to(device)
    )

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
        validation_split=0.1,
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
    theta_true,
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
    langevin,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    n_obs = context.shape[0]

    # normalize context
    context_norm = (context - x_train_mean) / x_train_std

    # normalize prior
    low_norm = (prior.low - theta_train_mean) / theta_train_std * 2
    high_norm = (prior.high - theta_train_mean) / theta_train_std * 2
    prior_score_fn_norm = get_vpdiff_uniform_score(
        low_norm.to(device), high_norm.to(device), score_network.to(device)
    )

    print("=======================================================================")
    print(
        f"Sampling from the approximate posterior at {theta_true}: n_obs = {n_obs}, nsamples = {nsamples}."
    )
    print(f"======================================================================")
    if langevin:
        print()
        print(f"Using LANGEVIN sampler.")
        print()
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
        ).cpu()
        time_elapsed = time.time() - start_time
        results_dict = None

        # save  path
        save_path += f"langevin_steps_400_5/"
        samples_filename = (
            save_path + f"posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}.pkl"
        )
        time_filename = save_path + f"time_{theta_true.tolist()}_n_obs_{n_obs}.pkl"

    else:
        print()
        print(f"Using EULER sampler.")
        print()
        # estimate cov
        # start_time = time.time()
        cov_est = vmap(
            lambda x: score_network.ddim(shape=(1000,), x=x, steps=100, eta=1),
            randomness="different",
        )(context_norm.to(device))

        cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
        # time_cov_est = time.time() - start_time

        # # define score function for tall posterior
        # score_fn = partial(
        #     diffused_tall_posterior_score,
        #     prior_score_fn=prior_score_fn_norm,
        #     x_obs=context_norm.to(device),
        #     nse=score_network.to(device),
        # )

        if cov_mode == "GAUSS":
            warmup = 0.5
        else:
            warmup = 0.5
        score_fn = partial(
            diffused_tall_posterior_score,
            prior_type="uniform",
            prior=None,  # no need for tweedie prior
            prior_score_fn=prior_score_fn_norm,  # analytical prior score function
            x_obs=context_norm.to(device),  # observations
            nse=score_network,  # trained score network
            cov_mode=cov_mode if cov_mode in ["JAC", "GAUSS"] else "GAUSS",
            warmup_alpha=warmup,
            psd_clipping=True if cov_mode in ["JAC", "GAUSS_clip"] else False,
            scale_gradlogL=True,
            dist_cov_est=cov_est,
        )

        # sample from tall posterior
        start_time = time.time()
        (
            samples,
            all_samples,
            gradlogL,
            lda,
            posterior_scores,
            means_posterior_backward,
            sigma_posterior_backward,
        ) = euler_sde_sampler(
            score_fn,
            nsamples,
            dim_theta=len(theta_true),
            beta=score_network.beta,
            device=device,
            debug=True,
            theta_clipping_range=(-3, 3),
        )
        time_elapsed = time.time() - start_time  # + time_cov_est

        assert torch.isnan(samples).sum() == 0

        results_dict = {
            "all_theta_learned": all_samples,
            "gradlogL": gradlogL,
            "lda": lda,
            "posterior_scores": posterior_scores,
            "means_posterior_backward": means_posterior_backward,
            "sigma_posterior_backward": sigma_posterior_backward,
        }

        # save  path
        save_path += f"euler_steps_{steps}/"
        samples_filename = (
            save_path
            + f"posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}_{cov_mode}.pkl"
        )
        results_dict_filename = (
            save_path
            + f"results_dict_{theta_true.tolist()}_n_obs_{n_obs}_{cov_mode}.pkl"
        )
        time_filename = (
            save_path + f"time_{theta_true.tolist()}_n_obs_{n_obs}_{cov_mode}.pkl"
        )

    # unnormalize
    samples = samples.detach().cpu()
    samples = samples * theta_train_std + theta_train_mean

    # save  results
    save_path += f"euler_steps_{steps}/"
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
        "--run",
        type=str,
        default="train",
        choices=["train", "sample", "sample_all"],
        help="run type",
    )
    parser.add_argument(
        "--n_train", type=int, default=50_000, help="number of training data samples"
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
        "--gain",
        type=float,
        default=0.0,
        help="ground truth gain parameter for simulator",
    )
    parser.add_argument(
        "--n_obs",
        type=int,
        default=1,
        help="number of context observations for sampling",
    )
    parser.add_argument(
        "--theta_dim",
        type=int,
        choices=[3, 4],
        default=4,
        help="if 3, fix the gain parameter to 0",
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
        help="whether to use langevin sampler (Geffner et al., 2023)",
    )

    # Parse Arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(42)

    def run(n_obs=args.n_obs, run_type=args.run):
        # Define Experiment Path
        save_path = (
            PATH_EXPERIMENT
            + f"{args.theta_dim}d/n_train_{args.n_train}_n_epochs_{args.n_epochs}_lr_{args.lr}/"
        )
        os.makedirs(save_path, exist_ok=True)

        print()
        print("save_path: ", save_path)
        print("CUDA available: ", torch.cuda.is_available())
        print()

        # SBI Task: prior and simulator
        parameters = [("C", 10.0, 250.0), ("mu", 50.0, 500.0), ("sigma", 100.0, 5000.0)]
        input_parameters = ["C", "mu", "sigma"]
        if args.theta_dim == 4:
            parameters.append(("gain", -20.0, +20.0))
            input_parameters.append("gain")

        prior = prior_JRNMM(parameters=parameters)
        # simulator = partial(simulator_JRNMM, input_parameters=input_parameters)

        # # Summary features
        # summary_extractor = summary_JRNMM()
        # # let's use the *log* power spectral density
        # summary_extractor.embedding.net.logscale = True

        # Simulate Training Data
        filename = PATH_EXPERIMENT + f"dataset_n_train_50000.pkl"
        if args.theta_dim == 3:
            filename = PATH_EXPERIMENT + f"dataset_n_train_50000_3d.pkl"
        # if os.path.exists(filename):
        dataset_train = torch.load(filename)
        theta_train = dataset_train["theta"][: args.n_train]
        x_train = dataset_train["x"][: args.n_train]
        # else:
        #     theta_train = prior.sample((args.n_train,))
        #     x_train = simulator(theta_train)
        #     x_train = summary_extractor(x_train)  # (n_train, x_dim, 1)
        #     x_train = x_train[:, :, 0]  # (n_train, x_dim)

        #     dataset_train = {
        #         "theta": theta_train, "x": x_train
        #     }
        #     torch.save(dataset_train, filename)
        theta_train_mean = theta_train.mean(dim=0)
        theta_train_std = theta_train.std(dim=0)
        x_train_mean = x_train.mean(dim=0)
        x_train_std = x_train.std(dim=0)
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
                "data_train": dataset_train,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "save_path": save_path,
            }
        elif run_type == "sample":
            # Reference parameter and observations
            theta_true = torch.tensor([135.0, 220.0, 2000.0, args.gain])[
                : args.theta_dim
            ]
            filename = PATH_EXPERIMENT + f"x_obs_100_{theta_true.tolist()}.pkl"
            # if os.path.exists(filename):
            x_obs_100 = torch.load(filename)
            # else:
            #     x_obs_100 = torch.cat(
            #         [summary_extractor(simulator(theta_true)).reshape(1, -1) for _ in tqdm(range(100))], dim=0
            #     )
            #     torch.save(x_obs_100, filename)
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
                "theta_true": theta_true,
                "context": x_obs_100[:n_obs],
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
                "save_path": save_path,
            }
        run_fn(**kwargs_run)

    if args.submitit:
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
                    "partition": "gpu",
                },
            )
            return executor

        # subit job
        executor = get_executor_marg(f"_jrnnm_{args.theta_dim}d_{args.run}")
        # launch batches
        with executor.batch():
            print("Submitting jobs...", end="", flush=True)
            tasks = []
            if args.run == "sample_all":
                for n_obs in N_OBS_LIST:
                    tasks.append(executor.submit(run, n_obs=n_obs))
            else:
                tasks.append(executor.submit(run))
