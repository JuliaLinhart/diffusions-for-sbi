import os
import torch

from functools import partial

from lc2st import LC2ST

PATH_EXPERIMENT = "results/jrnnm/"
# parameters for the trained score network
N_TRAIN = 50_000
LR = 1e-4
N_EPOCHS = 5000
# parameters for inferred samples
N_OBS_LIST = [1, 8, 14, 22, 30]
COV_MODES = ["GAUSS", "JAC"]

def load_results(
    dim,
    result_name,
    n_obs,
    lr=LR,
    n_epochs=N_EPOCHS,
    gain=0.0,
    cov_mode=None,
    sampler="ddim",
    langevin=False,
    clip=False,
    single_obs=None,
    n_cal=None,
):
    theta_true = [135.0, 220.0, 2000.0, gain]
    if dim == 3:
        theta_true = theta_true[:3]
    path = PATH_EXPERIMENT + f"{dim}d/n_train_50000_n_epochs_{n_epochs}_lr_{lr}/"
    if langevin:
        path = path + "langevin_steps_400_5/"
    else:
        path = (
            path + f"{sampler}_steps_1000/"
            if sampler == "euler" or cov_mode == "GAUSS"
            else path + f"{sampler}_steps_400/"
        )
    if n_cal is not None:
        sample_path = path + f"posterior_samples_n_cal_{n_cal}_n_obs_{n_obs}.pkl"
    else:
        if single_obs is not None:
            sample_path = (
                path + f"single_obs/num_{single_obs}_" + result_name + f"_{theta_true}_n_obs_1.pkl"
            )
        else:
            sample_path = path + result_name + f"_{theta_true}_n_obs_{n_obs}.pkl"
        
    if not langevin:
        sample_path = sample_path[:-4] + f"_{cov_mode}.pkl"
    if clip:
        sample_path = sample_path[:-4] + "_clip.pkl"
    results = torch.load(sample_path)
    return results, path


if __name__ == "__main__":
    import argparse
    
    # Define Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str,
        default="train_data",
        choices=["train_data", "train_null", "eval_data", "eval_null"],
        help="run type",
    )

    parser.add_argument(
        "--n_ensemble",
        type=int,
        default=5,
        help="number of classifiers in ensemble",
    )

    parser.add_argument(
        "--all_nobs", action="store_true", help="run for all n_obs"
    )

    parser.add_argument(
        "--n_cal", type=int, default=10_000, help="number of clibration data samples for l-c2st"
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
            + f"{args.theta_dim}d/n_train_{N_TRAIN}_n_epochs_{N_EPOCHS}_lr_{LR}/"
        )

        print()
        print("CUDA available: ", torch.cuda.is_available())
        print("save_path:", save_path)
        print()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("====================================================================")
        print(f"L-C2ST on {args.theta_dim}D JRNMM: n_cal = {args.n_cal}, n_obs = {n_obs}")
        if args.langevin:
            print("Langevin sampler")
        else:
            print(f"{args.sampler.upper()} sampler, Covariance Mode: {args.cov_mode}")
        print(f"clip = {args.clip}")
        print("====================================================================")
        print()

        # Load calibration data
        filename = PATH_EXPERIMENT + f"dataset_n_cal_10000_n_obs_{n_obs}.pkl"
        if args.theta_dim == 3:
            filename = PATH_EXPERIMENT + f"dataset_n_cal_10000_n_obs_{n_obs}_3d.pkl"
        dataset_cal = torch.load(filename)
        theta_cal = dataset_cal["theta"][: args.n_cal]
        x_cal = dataset_cal["x"][: args.n_cal]

        # Posterior calibration samples
        samples_cal, sampler_path = load_results(
            dim=args.theta_dim,
            result_name="posterior_samples",
            n_obs=n_obs,
            gain=args.gain,
            cov_mode=args.cov_mode,
            sampler=args.sampler,
            langevin=args.langevin,
            clip=args.clip,
            n_cal=args.n_cal,
        )

        print("theta_cal: ", theta_cal.shape)
        print("x_cal: ", x_cal.shape)
        print("samples_cal: ", samples_cal.shape)
        print()

        lc2st_path = sampler_path + f"lc2st_results/"
        os.makedirs(lc2st_path, exist_ok=True)

        filename = f"lc2st_n_cal_{args.n_cal}_n_obs_{n_obs}.pkl"
        if not args.langevin:
            filename = filename[:-4] + f"_{args.cov_mode}.pkl"
        if args.clip:
            filename = filename[:-4] + "_clip.pkl"

        print("lc2st_path: ", lc2st_path + filename)
        print()

        if os.path.exists(lc2st_path + filename):
            print("Loading LC2ST object...")
            print()
            lc2st = torch.load(lc2st_path + filename)
        else:
            lc2st = LC2ST(
                thetas=theta_cal.to(device),
                xs=x_cal.reshape(args.n_cal, -1).to(device),
                posterior_samples=samples_cal.to(device),
                n_ensemble=args.n_ensemble,
                classifier="mlp",
            )
        lc2st.n_ensemble = args.n_ensemble

        if "train" in run_type:
            print("Training on observed data...")
            print()
            _ = lc2st.train_on_observed_data()
            if "null" in run_type:
                _ = lc2st.train_under_null_hypothesis()
            torch.save(lc2st, lc2st_path + filename)

        elif "eval" in run_type:
            # Observations
            theta_true = [135.0, 220.0, 2000.0, args.gain]
            if args.theta_dim == 3:
                theta_true = theta_true[:3]
            x_obs = torch.load(PATH_EXPERIMENT + f'x_obs_100_{theta_true}.pkl')[:n_obs]
            # Posterior samples at observation
            samples_obs, _ = load_results(
                dim=args.theta_dim,
                result_name="posterior_samples",
                n_obs=n_obs,
                gain=args.gain,
                cov_mode=args.cov_mode,
                sampler=args.sampler,
                langevin=args.langevin,
                clip=args.clip,
            )
            probs_data, scores_data = lc2st.get_scores_on_observed_data(theta_o=samples_obs, x_o=x_obs.reshape(-1), return_probs=True,)
            ensemble_probs = probs_data.mean(axis=0)
            ensemble_stat = ((ensemble_probs - [0.5] * len(ensemble_probs)) ** 2).mean()

            results_dict = {"scores_data": scores_data, "ensemble_stat": ensemble_stat}

            if "null" in run_type:
                scores_null = lc2st.get_statistics_under_null_hypothesis(theta_o=samples_obs, x_o=x_obs.reshape(-1), verbosity=1)
                results_dict["scores_null"] = scores_null

                p_values = []
                for stat_data in scores_data:
                    p_values.append((stat_data < scores_null).mean())
                results_dict["p_values"] = p_values

                ensemble_p_value = (ensemble_stat < scores_null).mean()
                results_dict["ensemble_p_value"] = ensemble_p_value

            print()
            print("Results: ")
            for key, value in results_dict.items():
                if key == "scores_null":
                    continue
                print(f"{key}: {value}")
            import ipdb; ipdb.set_trace()

            torch.save(results_dict, lc2st_path + f"results_ensemble_{args.n_ensemble}_{theta_true}_" + filename)
            print("Saved at: ", lc2st_path + f"results_ensemble_{args.n_ensemble}_{theta_true}_" + filename)

    if args.all_nobs:
        for n_obs in N_OBS_LIST:
            run(n_obs=n_obs)
    else:
        run()

