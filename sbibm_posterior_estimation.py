import sys

sys.path.append("tasks/sbibm/")

import argparse
import os
import torch
import time

from functools import partial
from nse import NSE, NSELoss
from sm_utils import train_with_validation as train
from torch.func import vmap

from tqdm import tqdm
from zuko.nn import MLP

from tasks.sbibm.data_generators import get_task #, get_multiobs_task
from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler
from vp_diffused_priors import get_vpdiff_gaussian_score, get_vpdiff_uniform_score

PATH_EXPERIMENT = "results/sbibm/"
NUM_OBSERVATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

N_TRAIN_LIST = [1000, 3000, 10000, 30000]
N_OBS_LIST = [1, 8, 14, 22, 30]

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
        device = "cuda:0"

    # Prepare training data
    # normalize theta
    theta_train_norm = (theta_train - theta_train.mean(dim=0)) / theta_train.std(dim=0)
    # normalize x
    x_train_norm = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
    # replace nan by 0 (due to std in sir for n_train = 1000)
    x_train_norm = torch.nan_to_num(x_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
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
    print(
        f"============================================================================="
    )
    print()

    if theta_train.shape[0] > 10000:
        min_nb_epochs = n_epochs * 0.8 # 4000
    else:
        min_nb_epochs = 100
    
    # Train Score Network
    avg_score_net, train_losses, val_losses, best_epoch = train(
        score_network,
        dataset=data_train,
        loss_fn=NSELoss(score_network),
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        # track_loss=True,
        validation_split=0.2,
        early_stopping=True,
        min_nb_epochs=min_nb_epochs,
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
        {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch},
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
    prior_type,
    cov_mode,
    langevin=False,
    clip=False,
    log_space=False,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    n_obs = context.shape[0]

    # # normalize context
    # if log_space:
    #     context = torch.log(context)
    context_norm = (context - x_train_mean) / x_train_std
    # replace nan by 0 (due to std in sir for n_train = 1000)
    context_norm = torch.nan_to_num(context_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # normalize prior
    if prior_type == "uniform":
        low_norm = (prior.low - theta_train_mean) / theta_train_std * 2
        high_norm = (prior.high - theta_train_mean) / theta_train_std * 2
        prior_norm = torch.distributions.Uniform(
            low_norm.to(device), high_norm.to(device)
        )
        prior_score_fn_norm = get_vpdiff_uniform_score(
            low_norm.to(device), high_norm.to(device), score_network.to(device)
        )
    elif prior_type == "gaussian":
        if log_space:
            loc = prior.base_dist.loc
            cov = torch.diag_embed(prior.base_dist.scale.square())
        else:
            loc = prior.loc
            cov = prior.covariance_matrix
        loc_norm = (loc - theta_train_mean) / theta_train_std
        cov_norm = torch.diag(1 / theta_train_std) @ cov @ torch.diag(1 / theta_train_std)
        prior_norm = torch.distributions.MultivariateNormal(
            loc_norm.to(device), cov_norm.to(device)
        )
        prior_score_fn_norm = get_vpdiff_gaussian_score(
            loc_norm.to(device), cov_norm.to(device), score_network.to(device)
        )
    else:
        raise NotImplementedError

    print("=======================================================================")
    print(
        f"Sampling from the approximate posterior for observation {num_obs}: n_obs = {n_obs}, nsamples = {nsamples}."
    )
    print(f"======================================================================")

    if langevin:
        print()
        print(f"Using LANGEVIN sampler, clip = {clip}.")
        print()
        ext = ""
        theta_clipping_range = (None, None)
        if clip:
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
        samples_filename = save_path + f"posterior_samples_{num_obs}_n_obs_{n_obs}{ext}.pkl"
        time_filename = save_path + f"time_{num_obs}_n_obs_{n_obs}{ext}.pkl"
    else:
        print()
        print(f"Using EULER sampler, cov_mode = {cov_mode}, clip = {clip}.")
        print()

        # estimate cov
        # start_time = time.time()
        cov_est = vmap(
            lambda x: score_network.ddim(shape=(1000,), x=x, steps=100, eta=0.5),
            randomness="different",
        )(context_norm.to(device))
        cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
        # time_cov_est = time.time() - start_time

        # define score function for tall posterior
        score_fn = partial(
            diffused_tall_posterior_score,
            prior=prior_norm,  # normalized prior
            prior_type=prior_type,
            prior_score_fn=prior_score_fn_norm,  # analytical prior score function
            x_obs=context_norm.to(device),  # observations
            nse=score_network,  # trained score network
            dist_cov_est=cov_est,
            cov_mode=cov_mode,
            # warmup_alpha=0.5 if cov_mode == 'JAC' else 0.0,
            # psd_clipping=True if cov_mode == 'JAC' else False,
            # scale_gradlogL=True,
        )
        cov_mode_name = cov_mode
        theta_clipping_range = (None, None)
        if clip:
            theta_clipping_range = (-3, 3)
            cov_mode_name += "_clip"

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

        assert torch.isnan(samples).sum() == 0

        # results_dict = {
        #     "all_theta_learned": all_samples,
        #     # "gradlogL": gradlogL,
        #     # "lda": lda,
        #     # "posterior_scores": posterior_scores,
        #     # "means_posterior_backward": means_posterior_backward,
        #     # "sigma_posterior_backward": sigma_posterior_backward,
        # }

        save_path += f"euler_steps_{steps}/"
        samples_filename = (
            save_path + f"posterior_samples_{num_obs}_n_obs_{n_obs}_{cov_mode_name}.pkl"
        )
        # results_dict_filename = (
        #     save_path + f"results_dict_{num_obs}_n_obs_{n_obs}_{cov_mode_name}.pkl"
        # )
        time_filename = save_path + f"time_{num_obs}_n_obs_{n_obs}_{cov_mode_name}.pkl"

    # unnormalize
    samples = samples.detach().cpu()
    samples = samples * theta_train_std + theta_train_mean
    if log_space:
        samples = torch.exp(samples)

    # save  results
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(samples, samples_filename)
    torch.save(time_elapsed, time_filename)
    # if results_dict is not None:
    #     torch.save(results_dict, results_dict_filename)

def run_sample_reference(task_name, x_obs, num_obs, theta_true, save_path):
    print("=======================================================================")
    print(
        f"Sampling from the reference posterior for observation {num_obs}: n_obs = {len(x_obs)} ."
    )
    print(f"======================================================================")
    if theta_true.ndim > 1:
        theta_true = theta_true[0]

    task_multiobs = get_multiobs_task(task_name)
    kwargs = {}
    if task_name in ["lotka_volterra", "sir"]:
        kwargs = {"theta_true": theta_true}
    samples_ref = task_multiobs._sample_reference_posterior_multiobs(1000, x_obs, **kwargs)

    # save  results
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(samples_ref, save_path + f"true_posterior_samples_num_{num_obs}_n_obs_{len(x_obs)}.pkl")


if __name__ == "__main__":
    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submitit",
        action="store_true",
        help="whether to use submitit for running the job",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gaussian_linear",
        choices=["gaussian_linear", "gaussian_mixture", "slcp", "lotka_volterra", "sir"],
        help="task name",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="train",
        choices=["train", "sample", "train_all", "sample_all"],
        help="run type",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=50_000,
        help="number of training data samples (1000, 3000, 10000, 30000 in [Geffner et al. 2023])",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5000, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate for training (1e-3/1e-4 in [Geffner et al. 2023]))",
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
        "--num_obs", type=int, default=1, help="number of the observation in sbibm"
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
        "--clip",
        action="store_true",
        help="whether to clip the samples during sampling",
    )
    parser.add_argument(
        "--reference", action="store_true", help="whether to sample from ref. posterior"
    )

    # Parse Arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(42)

    # Define task path
    task_path = PATH_EXPERIMENT + f"{args.task}/"
    if args.task == "lotka_volterra":
        task_path = PATH_EXPERIMENT + f"{args.task}_f2/"

    def run(
        n_train=args.n_train, num_obs=args.num_obs, n_obs=args.n_obs, run_type=args.run
    ):
        batch_size = args.batch_size
        if n_train > 10000:
            batch_size = 256
        # Define Experiment Path
        save_path = (
            task_path
            + f"n_train_{n_train}_bs_{batch_size}_n_epochs_{args.n_epochs}_lr_{args.lr}/"
        )
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
            theta_train = dataset_train["theta"][:n_train].float()
            x_train = dataset_train["x"][:n_train].float()
        else:
            theta_train = prior(50000)
            x_train = simulator(theta_train)

            dataset_train = {"theta": theta_train, "x": x_train}
            torch.save(dataset_train, filename)
        # extract training data for given n_train
        theta_train, x_train = theta_train[:n_train], x_train[:n_train]

        if args.task in ["lotka_volterra", "sir"]:
            # transform theta to log space
            print("Transforming data to log space.")
            theta_train = torch.log(theta_train)
            # x_train = torch.log(x_train)

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
                "batch_size": batch_size,
                "lr": args.lr,
                "save_path": save_path,
            }
        elif run_type == "sample":
            # # Reference parameter and observations
            if os.path.exists(task_path + f"theta_true_list.pkl"):
                theta_true_list = torch.load(task_path + f"theta_true_list.pkl")
            else:
                theta_true_list = [prior(1) for _ in NUM_OBSERVATION_LIST]
                torch.save(theta_true_list, task_path + f"theta_true_list.pkl")

            theta_true = theta_true_list[num_obs - 1]

            filename = task_path + f"x_obs_100_num_{num_obs}_new.pkl"
            if args.task in ["lotka_volterra", "sir"]:
                filename = task_path + f"x_obs_100_num_{num_obs}_plcr.pkl"
            if os.path.exists(filename):
                x_obs_100 = torch.load(filename)
            else:
                x_obs_100 = torch.cat(
                    [simulator(theta_true).reshape(1, -1) for _ in tqdm(range(100))],
                    dim=0,
                )
                torch.save(x_obs_100, filename)
            context = x_obs_100[:n_obs].reshape(n_obs, -1)

            if args.reference:
                run_fn = run_sample_reference
                kwargs_run = {
                    "task_name": args.task,
                    "x_obs": context,
                    "num_obs": num_obs,
                    "theta_true": theta_true,
                    "save_path": task_path + f"reference_posterior_samples/",
                }
            else:
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
                    "prior": task.prior_dist if args.task != "slcp" else task.prior_dist.base_dist, # for score function
                    "prior_type": "uniform" if args.task == "slcp" else "gaussian", 
                    "cov_mode": args.cov_mode,
                    "clip": args.clip,  # for clipping
                    "langevin": args.langevin,
                    "log_space": args.task in ["lotka_volterra", "sir"],
                    "save_path": save_path,
                }

        run_fn(**kwargs_run)

    if not args.submitit:
        if args.run == "sample_all":
            # for n_train in N_TRAIN_LIST:
            for num_obs in NUM_OBSERVATION_LIST:
                for n_obs in N_OBS_LIST:
                    run(
                        # n_train=n_train,
                        num_obs=num_obs,
                        n_obs=n_obs,
                        run_type="sample",
                    )
        elif args.run == "train_all":
            for n_train in N_TRAIN_LIST:
                run(n_train=n_train, run_type="train")
        else:
            run()
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
                    "partition": "parietal",
                },
            )
            return executor

        # subit job
        executor = get_executor_marg(f"_{args.task}_{args.run}_{args.cov_mode}_clip_{args.clip}")
        # launch batches
        with executor.batch():
            print("Submitting jobs...", end="", flush=True)
            tasks = []
            if args.run == "sample_all":
                for n_train in N_TRAIN_LIST:
                    for num_obs in NUM_OBSERVATION_LIST:
                        for n_obs in N_OBS_LIST:
                            tasks.append(
                                executor.submit(
                                    run,
                                    n_train=n_train,
                                    num_obs=num_obs,
                                    n_obs=n_obs,
                                    run_type="sample",
                                )
                            )
            elif args.run == "train_all":
                for n_train in N_TRAIN_LIST:
                    tasks.append(
                        executor.submit(run, n_train=n_train, run_type="train")
                    )
            else:
                tasks.append(executor.submit(run))
            print("done")