import numpy as np
import os
import torch

from nse import NSELoss
from pf_nse import PF_NSE
from sm_utils import train_with_validation as train
from torch.func import vmap

# from zuko.nn import MLP

from tasks.sbibm import get_task
from vp_diffused_priors import get_vpdiff_gaussian_score, get_vpdiff_uniform_score

PATH_EXPERIMENT = "results/sbibm/"
NUM_OBSERVATION_LIST = list(np.arange(1, 26))

N_TRAIN_LIST = [1000, 3000, 10000, 30000]
MAX_N_TRAIN = 50_000
N_OBS_LIST = [30]
N_MAX_LIST = [3, 6, 30]
MAX_N_OBS = 100
NUM_SAMPLES = 1000

COV_MODES = ["GAUSS", "JAC"]


def setup(
    task, all=True, train_data=False, reference_data=False, reference_posterior=False
):
    kwargs = {}
    if task.name != "gaussian_linear":
        from jax import random

        rng_key = random.PRNGKey(1)
        kwargs = {"rng_key": rng_key}
    if all:
        train_data = True
        reference_data = True
        reference_posterior = True
    if train_data:
        data = task.generate_training_data(
            n_simulations=MAX_N_TRAIN, n_obs=max(N_MAX_LIST)
        )
        print("Training data:", data["x"].shape, data["theta"].shape)
    if reference_data:
        data = task.generate_reference_data(
            nb_obs=len(NUM_OBSERVATION_LIST), n_repeat=MAX_N_OBS, **kwargs
        )
        print("Reference data:", data[0].shape, data[1][1].shape)
    if reference_posterior:
        num_obs_list = range(1, len(NUM_OBSERVATION_LIST) + 1)
        n_obs_list = N_OBS_LIST
        for num_obs in num_obs_list:
            for n_obs in n_obs_list:
                data = task.generate_reference_posterior_samples(
                    num_obs=num_obs, n_obs=n_obs, num_samples=NUM_SAMPLES, **kwargs
                )
        print("Posterior samples:", data.shape)
    return


def run_train_sgm(
    theta_train,
    x_train,
    n_max,
    n_epochs,
    batch_size,
    lr,
    clf_free_guidance=False,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Prepare training data
    # normalize theta, x
    theta_train_norm = (theta_train - theta_train.mean(dim=0)) / theta_train.std(dim=0)
    x_train_norm = (x_train - x_train.mean(dim=0)) / x_train.std(dim=0)
    # replace nan by 0 (due to std in sir for n_train = 1000)
    x_train_norm = torch.nan_to_num(x_train_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # create context set sizes
    torch.manual_seed(42)  # for reproducibility
    n_train = theta_train.shape[0]
    ns = torch.randint(1, n_max + 1, (n_train, 1))
    print("Context set sizes:", ns.shape)

    # dataset for data loader
    data_train = torch.utils.data.TensorDataset(
        theta_train_norm.to(device), x_train_norm.to(device), ns.to(device)
    )

    # Score network: PF-NSE
    # embedding?
    theta_dim = theta_train.shape[-1]
    x_dim = x_train.shape[-1]
    score_network = PF_NSE(
        theta_dim=theta_dim,
        x_dim=x_dim,
        n_max=n_max,
        hidden_features=[256, 256, 256],
    ).to(device)

    # Train score network
    print(
        "=============================================================================="
    )
    print("PF-NSE: n_max = ", n_max)
    print(
        f"Training score network: n_train = {theta_train.shape[0]}, n_epochs = {n_epochs}."
    )
    print(
        f"============================================================================="
    )
    print()

    if theta_train.shape[0] > 10000:
        # min_nb_epochs = n_epochs * 0.8 # 4000
        min_nb_epochs = 2000
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
        validation_split=0.2,
        early_stopping=True,
        min_nb_epochs=min_nb_epochs,
        classifier_free_guidance=0.2 if clf_free_guidance else 0.0,
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
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
        },
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
    langevin="geffner",
    clip=False,
    theta_log_space=False,
    x_log_space=False,
    clf_free_guidance=False,
    save_path=PATH_EXPERIMENT,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    n_obs = context.shape[0]

    # reshape for normalization
    context, _ = score_network._create_pf_data(context)

    # normalize context
    if x_log_space:
        context = torch.log(context)
    context_norm = (context - x_train_mean) / x_train_std

    # undo reshape
    context_norm = context_norm.reshape(n_obs, -1)

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
        if theta_log_space:
            loc = prior.base_dist.loc
            cov = torch.diag_embed(prior.base_dist.scale.square())
        else:
            loc = prior.loc
            cov = prior.covariance_matrix
        loc_norm = (loc - theta_train_mean) / theta_train_std
        cov_norm = (
            torch.diag(1 / theta_train_std) @ cov @ torch.diag(1 / theta_train_std)
        )
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
        print(f"Using LANGEVIN sampler ({langevin.upper()}), clip = {clip}.")
        print()

        save_path += f"langevin_steps_400_5/"

        ext = ""
        theta_clipping_range = (None, None)
        if clip:
            theta_clipping_range = (-3, 3)
            ext = "_clip"

        if langevin == "geffner":
            samples = score_network.annealed_langevin_geffner(
                shape=(nsamples,),
                x=context_norm.to(device),
                prior_score_fn=prior_score_fn_norm,
                clf_free_guidance=clf_free_guidance,
                steps=400,
                lsteps=5,
                tau=0.5,
                theta_clipping_range=theta_clipping_range,
                verbose=True,
            ).cpu()
        elif langevin == "tammed":
            samples = score_network.predictor_corrector(
                (nsamples,),
                x=context_norm.to(device),
                steps=400,
                prior_score_fun=prior_score_fn_norm,
                lsteps=5,
                r=0.5,
                predictor_type="id",
                verbose=True,
                theta_clipping_range=theta_clipping_range,
            ).cpu()

            save_path = save_path[:-1] + "_ours/"
        else:
            raise NotImplementedError

        samples_filename = (
            save_path + f"posterior_samples_{num_obs}_n_obs_{n_obs}{ext}_prior.pkl"
        )
    else:
        print()
        print(f"Using DDIM sampler, cov_mode = {cov_mode}, clip = {clip}.")
        print()

        save_path += f"ddim_steps_{steps}/"

        cov_mode_name = cov_mode
        theta_clipping_range = (None, None)
        if clip:
            theta_clipping_range = (-3, 3)
            cov_mode_name += "_clip"

        cov_est, cov_est_prior = None, None
        if cov_mode == "GAUSS":
            # estimate cov for GAUSS
            xs, ns = score_network._create_pf_data(context_norm)
            cov_est = vmap(
                lambda x: score_network.ddim(
                    shape=(nsamples,), x=x, steps=100, eta=0.5
                ),
                randomness="different",
            )(xs.to(device))
            cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)

            if clf_free_guidance:
                x_ = torch.zeros_like(xs[0][None, :])
                n_ = torch.zeros_like(ns[0][None, :])
                cov_est_prior = vmap(
                    lambda x: score_network.ddim(
                        shape=(nsamples,), x=x, steps=100, eta=0.5, n=n_
                    ),
                    randomness="different",
                )(x_.to(device))
                cov_est_prior = vmap(lambda x: torch.cov(x.mT))(cov_est_prior)

        samples = score_network.ddim(
            shape=(nsamples,),
            x=context_norm.to(device),
            eta=1
            if steps == 1000
            else 0.8
            if steps == 400
            else 0.5,  # corresponds to the equivalent time setting from section 4.1
            steps=steps,
            theta_clipping_range=theta_clipping_range,
            prior=prior_norm,
            prior_type=prior_type,
            prior_score_fn=prior_score_fn_norm,
            clf_free_guidance=clf_free_guidance,
            dist_cov_est=cov_est,
            dist_cov_est_prior=cov_est_prior,
            cov_mode=cov_mode,
            verbose=True,
        ).cpu()

        assert (
            torch.isnan(samples).sum() == 0
        ), f"NaN in samples: {torch.isnan(samples).sum()}"

        samples_filename = (
            save_path
            + f"posterior_samples_{num_obs}_n_obs_{n_obs}_{cov_mode_name}_prior.pkl"
        )

    # unnormalize
    samples = samples.detach().cpu()
    samples = samples * theta_train_std + theta_train_mean
    if theta_log_space:
        samples = torch.exp(samples)

    assert samples.shape == (
        nsamples,
        theta_train_mean.shape[0],
    ), f"Shape of samples: {samples.shape}"

    # save  results
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(samples, samples_filename)


if __name__ == "__main__":
    import argparse

    # Define Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup",
        type=str,
        default=None,
        choices=["all", "train_data", "reference_data", "reference_posterior"],
        help="setup task data",
    )
    parser.add_argument(
        "--submitit",
        action="store_true",
        help="whether to use submitit for running the job",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "slcp",
            "lotka_volterra",
            "sir",
            "gaussian_linear",
            "gaussian_mixture",
            "gaussian_mixture_uniform",
            "two_moons",
            "bernoulli_glm",
            "bernoulli_glm_raw",
        ],
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
        default=MAX_N_TRAIN,
        help="number of training data samples (1000, 3000, 10000, 30000 in [Geffner et al. 2023])",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5000, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate for training (1e-3/1e-4 in [Geffner et al. 2023]))",
    )
    parser.add_argument(
        "--n_obs",
        type=int,
        default=30,
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
        type=str,
        default="",
        choices=["geffner", "tammed"],
        help="whether to use langevin sampler (Geffner et al. 2023) or our tammed ULA (Brosse et al. 2017)",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="whether to clip the samples during sampling",
    )
    parser.add_argument(
        "--clf_free_guidance",
        action="store_true",
        help="whether to use classifier free guidance to learn the diffused prior score",
    )
    parser.add_argument(
        "--n_max",
        type=int,
        default=1,
        choices=N_MAX_LIST,
        help="number of maximum observations for the score network",
    )

    # Parse Arguments
    args = parser.parse_args()

    # seed
    torch.manual_seed(42)

    # SBI Task: prior and simulator
    task = get_task(args.task, save_path="tasks/sbibm/data/")

    # Setup task data
    if args.setup is not None:
        print("Setting up task data.")
        setup(
            task,
            all=args.setup == "all",
            train_data=args.setup == "train_data",
            reference_data=args.setup == "reference_data",
            reference_posterior=args.setup == "reference_posterior",
        )
        exit()

    # Define task path
    task_path = PATH_EXPERIMENT + f"{args.task}/"

    def run(
        n_train=args.n_train, num_obs=args.num_obs, n_obs=args.n_obs, run_type=args.run
    ):
        # Define Experiment Path
        save_path = (
            task_path
            + f"n_train_{n_train}_bs_{args.batch_size}_n_epochs_{args.n_epochs}_lr_{args.lr}/pf_n_max_{args.n_max}/"
        )

        if args.clf_free_guidance:
            save_path += "clf_free_guidance/"

        os.makedirs(save_path, exist_ok=True)

        print()
        print("save_path: ", save_path)
        print("CUDA available: ", torch.cuda.is_available())
        print()

        if run_type == "train":
            # get training data
            dataset_train = task.get_training_data(
                n_simulations=MAX_N_TRAIN, n_obs=max(N_MAX_LIST)
            )
            theta_train = dataset_train["theta"].float()
            x_train = dataset_train["x"].float()
            # extract training data for given n_train and n_max
            theta_train, x_train = (
                theta_train[:n_train],
                x_train[:n_train, : args.n_max, :],
            )
            print("Training data:", theta_train.shape, x_train.shape)

            # log space transformation
            if args.task in ["lotka_volterra", "sir"]:
                print("Transforming data to log space.")
                theta_train = torch.log(theta_train)
                if args.task == "lotka_volterra":
                    x_train = torch.log(x_train)

            # compute mean and std of training data
            theta_train_mean, theta_train_std = theta_train.mean(
                dim=0
            ), theta_train.std(dim=0)
            x_train_mean, x_train_std = x_train.mean(dim=0), x_train.std(dim=0)
            means_stds_dict = {
                "theta_train_mean": theta_train_mean,
                "theta_train_std": theta_train_std,
                "x_train_mean": x_train_mean,
                "x_train_std": x_train_std,
            }
            torch.save(means_stds_dict, save_path + f"train_means_stds_dict.pkl")

            run_fn = run_train_sgm
            kwargs_run = {
                "theta_train": theta_train,
                "x_train": x_train,
                "n_max": args.n_max,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "clf_free_guidance": args.clf_free_guidance,
                "save_path": save_path,
            }
        elif run_type == "sample":
            # get reference observations
            x_obs_100 = task.get_reference_observation(num_obs, n_repeat=MAX_N_OBS)
            context = x_obs_100[:n_obs].reshape(n_obs, -1)
            print("Context:", context.shape)
            if args.task in ["bernoulli_glm"]:
                # summary statistics
                context = task.compute_summary_statistics(context)
                print("Summary statistics:", context.shape)

            # Trained Score network
            score_network = torch.load(
                save_path + f"score_network.pkl",
                map_location=torch.device("cpu"),
            )
            score_network.net_type = "default"

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
                "nsamples": NUM_SAMPLES,
                "score_network": score_network,
                "steps": 1000
                if args.cov_mode == "GAUSS"
                else 400,  # corresponds to the equivalent time setting from section 4.1
                "theta_train_mean": theta_train_mean,  # for (un)normalization
                "theta_train_std": theta_train_std,  # for (un)normalization
                "x_train_mean": x_train_mean,  # for (un)normalization
                "x_train_std": x_train_std,  # for (un)normalization
                "prior": task.prior(),  # for score function
                "prior_type": "uniform"
                if args.task in ["slcp", "two_moons", "gaussian_mixture_uniform"]
                else "gaussian",
                "cov_mode": args.cov_mode,
                "clip": args.clip,  # for clipping
                "langevin": args.langevin,
                "theta_log_space": args.task in ["lotka_volterra", "sir"],
                "x_log_space": args.task == "lotka_volterra",
                "clf_free_guidance": args.clf_free_guidance,
                "save_path": save_path,
            }

        run_fn(**kwargs_run)

    if not args.submitit:
        if args.run == "sample_all":
            for n_train in N_TRAIN_LIST:
                for num_obs in NUM_OBSERVATION_LIST:
                    for n_obs in N_OBS_LIST:
                        run(
                            n_train=n_train,
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
        executor = get_executor_marg(
            f"_{args.task}_{args.run}_{args.cov_mode}_clip_{args.clip}"
        )
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
