import os
import torch

from functools import partial
from nse import NSE, NSELoss
from sm_utils import train_with_validation as train

# from tasks.jrnnm.summary import summary_JRNMM
from tasks.jrnnm.prior import prior_JRNMM
# from tasks.jrnnm.simulator import simulator_JRNMM

from tqdm import tqdm
from torch.func import vmap
from zuko.nn import MLP

from tall_posterior_sampler import diffused_tall_posterior_score, euler_sde_sampler, tweedies_approximation
from vp_diffused_priors import get_vpdiff_uniform_score


PATH_EXPERIMENT = "results/jrnnm/"
N_OBS_LIST = [1, 8, 14, 22, 30]
COV_MODES = ["GAUSS", "JAC"]


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
        device = "cuda:0"

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
        hidden_features=[256, 256, 256],
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
        min_nb_epochs=n_epochs * 0.8,  # 4000
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
    sampler_type="ddim",
    langevin=False,
    clip=False,
    save_path=PATH_EXPERIMENT,
    single_obs=None,
    n_cal=None,
):
    # Set Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # normalize context
    context_norm = (context - x_train_mean) / x_train_std

    # reshape context for calibration data if n_obs = 1
    if n_cal is None:
        n_obs = context.shape[0]
    else: 
        if context_norm.ndim < 3:
            context_norm = context_norm.unsqueeze(1)
        n_obs = context_norm.shape[1]

    # normalize prior
    low_norm = (prior.low - theta_train_mean) / theta_train_std * 2
    high_norm = (prior.high - theta_train_mean) / theta_train_std * 2
    prior_norm = torch.distributions.Uniform(low_norm.to(device), high_norm.to(device))
    prior_score_fn_norm = get_vpdiff_uniform_score(
        low_norm.to(device), high_norm.to(device), score_network.to(device)
    )

    print("=======================================================================")
    if len(theta_true) in [3,4]:
        print(
            f"Sampling from the approximate posterior at {theta_true}: n_obs = {n_obs}, nsamples = {nsamples}."
        )
    elif len(theta_true) == n_cal:
        print(
            f"Sampling from the approximate posterior at calibration data: n_obs = {n_obs}. \n x_cal shape = {context_norm.shape}."
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
        # samples = score_network.predictor_corrector(
        #     (nsamples,),
        #     x=context_norm.to(device),
        #     steps=400,
        #     prior_score_fun=prior_score_fn_norm,
        #     lsteps=5,
        #     r=0.5,
        #     predictor_type="id",
        #     verbose=True,
        #     theta_clipping_range=theta_clipping_range,
        # ).cpu()
        sampler = lambda x: score_network.annealed_langevin_geffner(
            shape=(nsamples,),
            x=x.to(device),
            prior_score_fn=prior_score_fn_norm,
            steps=400,
            lsteps=5,
            tau=0.5,
            theta_clipping_range=theta_clipping_range,
            verbose=True,
        )
        if n_cal is None:
            samples = sampler(context_norm.to(device)).cpu()
        else:
            assert n_cal == context_norm.shape[0]
            samples = vmap(sampler, randomness="different")(context_norm.to(device)).squeeze(1).cpu()
        
        # save  path
        save_path += f"langevin_steps_400_5/"
        if n_cal is None:
            if single_obs is not None:
                save_path += f"single_obs/"
                samples_filename = (
                    save_path
                    + f"num_{single_obs}_posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}{ext}.pkl"
                )
            else:
                samples_filename = (
                    save_path
                    + f"posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}{ext}.pkl"
                )
        else:
            samples_filename = (
                save_path
                + f"posterior_samples_n_cal_{n_cal}_n_obs_{n_obs}{ext}.pkl"
            )

    else:
        print()
        print(
            f"Using {sampler_type.upper()} sampler, cov_mode = {cov_mode}, clip = {clip}."
        )
        print()

        cov_mode_name = cov_mode
        theta_clipping_range = (None, None)
        if clip:
            theta_clipping_range = (-3, 3)
            cov_mode_name += "_clip"

        cov_est = None
        if cov_mode == "GAUSS":
            # estimate cov
            cov_est_fn = lambda x: score_network.ddim(shape=(1000,), x=x, steps=100, eta=0.5)
            if n_cal is None:
                cov_est = vmap(cov_est_fn, randomness="different")(context_norm.to(device))
                cov_est = vmap(lambda x: torch.cov(x.mT))(cov_est)
            else:
                # commented part yields cuda memory error (on drago3)
                # print(f"Estimating Covariance on Calibration Data of size {context_norm.shape[0]}...")
                # cov_est = vmap(vmap(
                #     cov_est_fn,
                #     randomness="different",
                # ), randomness='different')(context_norm.to(device))

                # split context into batches
                b_size = 2000 if n_obs == 1 else 200 if n_obs == 8 else 100 if n_obs == 14 else 50 if n_obs in [22, 30] else 1
                context_norm_b = context_norm.reshape(-1, b_size, n_obs, context_norm.shape[-1])
                cov_est = []
                for batch in tqdm(context_norm_b, desc="Estimating Covariance on calibration data over batches"):
                    cov_ = vmap(vmap(
                        cov_est_fn,
                        randomness="different",
                    ), randomness='different')(batch.to(device))
                    assert torch.isnan(cov_).sum() == 0, f"Number of NaNs in cov est: {torch.isnan(cov_).sum()}, {torch.where(torch.isnan(cov_))}"
                    cov_est.append(cov_)
                cov_est = torch.cat(cov_est, dim=0)
                cov_est = vmap(vmap(lambda x: torch.cov(x.mT)))(cov_est)
            assert torch.isnan(cov_est).sum() == 0, f"Number of NaNs in cov est: {torch.isnan(cov_est).sum()}, {torch.where(torch.isnan(cov_est))}"
            print(f"cov est shape: {cov_est.shape}")
            print()

        if sampler_type == "ddim":
            save_path += f"ddim_steps_{steps}/"

            sampler = lambda x, cov_est: score_network.ddim(
                shape=(nsamples,),
                x=x.to(device),
                eta=1
                if steps == 1000
                else 0.8
                if steps == 400
                else 0.5,  # corresponds to the equivalent time setting from section 4.1
                steps=steps,
                theta_clipping_range=theta_clipping_range,
                prior=prior_norm,
                prior_type="uniform",
                prior_score_fn=prior_score_fn_norm,
                dist_cov_est=cov_est,
                cov_mode=cov_mode,
                verbose=True,
            )

            if n_cal is None:
                samples = sampler(context_norm.to(device), cov_est).cpu()
            else:
                assert n_cal == context_norm.shape[0]
                print(f"Sampling from tall posterior on each of the {n_cal} observations from the calibration data...")
                if cov_mode == "GAUSS":
                    samples = vmap(sampler, randomness='different')(context_norm.to(device), cov_est).squeeze(1).cpu()
                else:
                    samples = vmap(partial(sampler, cov_est=None), randomness='different')(context_norm.to(device)).squeeze(1).cpu()
            
        else:
            if n_cal is not None:
                raise NotImplementedError
        
            save_path += f"euler_steps_{steps}/"
            # define score function for tall posterior
            score_fn = partial(
                diffused_tall_posterior_score,
                prior_type="uniform",
                prior=None,
                prior_score_fn=prior_score_fn_norm,  # analytical prior score function
                x_obs=context_norm.to(device),  # observations
                nse=score_network,  # trained score network
                dist_cov_est=cov_est,
                cov_mode=cov_mode,
            )
            # sample from tall posterior
            (
                samples,
                _,
            ) = euler_sde_sampler(
                score_fn,
                nsamples,
                dim_theta=len(theta_true),
                beta=score_network.beta,
                device=device,
                debug=False,
                theta_clipping_range=theta_clipping_range,
            )

        # save  path
        if n_cal is None:
            if single_obs is not None:
                save_path += f"single_obs/"
                samples_filename = (
                    save_path
                    + f"num_{single_obs}_posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}_{cov_mode_name}.pkl"
                )
            else:
                samples_filename = (
                    save_path
                    + f"posterior_samples_{theta_true.tolist()}_n_obs_{n_obs}_{cov_mode_name}.pkl"
                )
        else:
            samples_filename = (
                save_path
                + f"posterior_samples_n_cal_{n_cal}_n_obs_{n_obs}_{cov_mode_name}.pkl"
            )

    if torch.isnan(samples).sum() != 0:
        print(f"Number of NaNs in samples: {torch.isnan(samples).sum()}, {torch.where(torch.isnan(samples))}")
        # replace NaNs with mean if it's less than 10% of the samples
        theshold = 0.1 * samples.shape[0] * samples.shape[1]
        if torch.isnan(samples).sum() < threshold:
            samples_new = torch.where(torch.isnan(samples),0,samples)
            samples = torch.where(samples_new == 0,samples_new.mean(axis=0), samples_new)

    assert torch.isnan(samples).sum() == 0, f"Number of NaNs in samples: {torch.isnan(samples).sum()}, {torch.where(torch.isnan(samples))}"

    # unnormalize
    samples = samples.detach().cpu()
    samples = samples * theta_train_std + theta_train_mean

    # save  results
    os.makedirs(
        save_path,
        exist_ok=True,
    )
    torch.save(samples, samples_filename)

    print(f"samples shape: {samples.shape}")
    print(f"Saved samples to {samples_filename}.")


if __name__ == "__main__":
    import argparse
    
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
        choices=["train", "sample", "sample_all", "sample_cal", "sample_cal_all"],
        help="run type",
    )
    parser.add_argument(
        "--n_train", type=int, default=50_000, help="number of training data samples"
    )
    parser.add_argument(
        "--n_cal", type=int, default=10_000, help="number of clibration data samples for l-c2st"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5000, help="number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size for training"
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
        "--sampler",
        type=str,
        default="ddim",
        choices=["euler", "ddim"],
        help="SDE sampler type",
    )
    parser.add_argument(
        "--langevin",
        action="store_true",
        help="whether to use langevin sampler (Geffner et al., 2023)",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="whether to clip the posterior samples",
    )
    parser.add_argument(
        "--single_obs",
        action="store_true",
        help="whether to sample for every observation seperately with n_obs = 1",
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
        # theta_train = dataset_train["theta"][: args.n_train]
        # x_train = dataset_train["x"][: args.n_train]
        # else:
        #     print("==============================================================================")
        #     print(f"Sampling training data.")
        #     print(f"======================================================================")
        #     theta_train = prior.sample((args.n_train,))
        #     x_train = simulator(theta_train)
        #     x_train = summary_extractor(x_train)  # (n_train, x_dim, 1)
        #     x_train = x_train[:, :, 0]  # (n_train, x_dim)

        #     dataset_train = {
        #         "theta": theta_train, "x": x_train
        #     }
        #     torch.save(dataset_train, filename)
        # theta_train_mean = theta_train.mean(dim=0)
        # theta_train_std = theta_train.std(dim=0)
        # x_train_mean = x_train.mean(dim=0)
        # x_train_std = x_train.std(dim=0)
        # means_stds_dict = {
        #     "theta_train_mean": theta_train_mean,
        #     "theta_train_std": theta_train_std,
        #     "x_train_mean": x_train_mean,
        #     "x_train_std": x_train_std,
        # }
        # torch.save(means_stds_dict, save_path + f"train_means_stds_dict.pkl")

        if run_type == "train":
            run_fn = run_train_sgm
            kwargs_run = {
                "data_train": dataset_train,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "save_path": save_path,
            }
        elif run_type == "sample" or run_type == "sample_cal":
            # Trained Score network
            score_network = torch.load(
                save_path + f"score_network.pkl",
                map_location=torch.device("cpu"),
            )
            score_network.net_type = "default"
            score_network.tweedies_approximator = tweedies_approximation

            # Mean and std of training data
            means_stds_dict = torch.load(save_path + f"train_means_stds_dict.pkl")
            theta_train_mean = means_stds_dict["theta_train_mean"]
            theta_train_std = means_stds_dict["theta_train_std"]
            x_train_mean = means_stds_dict["x_train_mean"]
            x_train_std = means_stds_dict["x_train_std"]

            if run_type == "sample":
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

                if args.single_obs:
                    for i, x_obs in enumerate(x_obs_100[:n_obs]):
                        run_fn = run_sample_sgm
                        kwargs_run = {
                            "theta_true": theta_true,
                            "context": x_obs.unsqueeze(0),
                            "nsamples": args.nsamples,
                            "score_network": score_network,
                            "steps": args.steps,
                            "theta_train_mean": theta_train_mean,  # for (un)normalization
                            "theta_train_std": theta_train_std,  # for (un)normalization
                            "x_train_mean": x_train_mean,  # for (un)normalization
                            "x_train_std": x_train_std,  # for (un)normalization
                            "prior": prior,  # for score function
                            "cov_mode": args.cov_mode,
                            "sampler_type": args.sampler,
                            "langevin": args.langevin,
                            "clip": args.clip,
                            "save_path": save_path,
                            "single_obs": i,
                        }
                        run_fn(**kwargs_run)
                    run_fn = lambda **kwargs: None
                else:
                    run_fn = run_sample_sgm
                    kwargs_run = {
                        "theta_true": theta_true,
                        "context": x_obs_100[:n_obs],
                        "nsamples": args.nsamples,
                        "score_network": score_network,
                        "steps": 1000
                        if args.cov_mode == "GAUSS"
                        else 400,  # corresponds to the equivalent time setting from section 4.1
                        "theta_train_mean": theta_train_mean,  # for (un)normalization
                        "theta_train_std": theta_train_std,  # for (un)normalization
                        "x_train_mean": x_train_mean,  # for (un)normalization
                        "x_train_std": x_train_std,  # for (un)normalization
                        "prior": prior,  # for score function
                        "cov_mode": args.cov_mode,
                        "sampler_type": args.sampler,
                        "langevin": args.langevin,
                        "clip": args.clip,
                        "save_path": save_path,
                    }

            elif run_type == "sample_cal":
                # Simulate Calibration Data
                filename = PATH_EXPERIMENT + f"dataset_n_cal_10000_n_obs_{n_obs}.pkl"
                if args.theta_dim == 3:
                    filename = PATH_EXPERIMENT + f"dataset_n_cal_10000_n_obs_{n_obs}_3d.pkl"
                # if os.path.exists(filename):
                dataset_cal = torch.load(filename)
                # else:
                #     print("==============================================================================")
                #     print(f"Sampling calibration data for n_obs = {n_obs}.")
                #     print(f"======================================================================")
                #     theta_cal = prior.sample((10000,))
                #     theta_cal_big = theta_cal.repeat(n_obs, 1)
                #     x_cal = simulator(theta_cal_big)
                #     x_cal = summary_extractor(x_cal)[:,:,0]  # (n_cal*n_obs, x_dim)
                #     if n_obs > 1:
                #         x_cal = x_cal.reshape(n_obs, args.n_cal, -1).permute(1, 0, 2) # (n_cal, n_obs, x_dim)
                #     print(x_cal.shape)

                #     dataset_cal = {
                #         "theta": theta_cal, "x": x_cal
                #     }
                #     torch.save(dataset_cal, filename)
                theta_cal = dataset_cal["theta"][: args.n_cal]
                x_cal = dataset_cal["x"][: args.n_cal]

                run_fn = run_sample_sgm
                kwargs_run = {
                    "theta_true": theta_cal,
                    "context": x_cal,
                    "nsamples": 1,
                    "score_network": score_network,
                    "steps": 1000
                    if args.cov_mode == "GAUSS"
                    else 400,  # corresponds to the equivalent time setting from section 4.1
                    "theta_train_mean": theta_train_mean,  # for (un)normalization
                    "theta_train_std": theta_train_std,  # for (un)normalization
                    "x_train_mean": x_train_mean,  # for (un)normalization
                    "x_train_std": x_train_std,  # for (un)normalization
                    "prior": prior,  # for score function
                    "cov_mode": args.cov_mode,
                    "sampler_type": args.sampler,
                    "langevin": args.langevin,
                    "clip": args.clip,
                    "save_path": save_path,
                    "n_cal": args.n_cal,
                }

        
        run_fn(**kwargs_run)

    if not args.submitit:
        if args.run == "sample_all":
            for n_obs in N_OBS_LIST:
                run(n_obs=n_obs, run_type="sample")
        elif args.run == "sample_cal_all":
            for n_obs in N_OBS_LIST:
                run(n_obs=n_obs, run_type="sample_cal")
        else:
            run()

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
