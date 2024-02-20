# Script to compute the results of the Gaussian Mixture toy example experiment (Section 4.1 of the paper).

import numpy as np
import pandas as pd
import torch
import tqdm

from ot.sliced import max_sliced_wasserstein_distance


if __name__ == "__main__":
    import sys

    destination_folder = sys.argv[1]
    data = torch.load(f"{destination_folder}/gaussian_mixture_exp.pt")
    try:
        all_data = pd.read_csv(
            f"{destination_folder}/gaussian_mixture_exp_treated.csv"
        ).set_index(keys=["N_OBS", "dim", "seed", "eps"])
    except FileNotFoundError:
        all_data = pd.DataFrame(
            columns=[
                "N_OBS",
                "dim",
                "eps",
                "seed",
                "alg",
                "dt",
                "sampling_steps",
                "sw",
            ],
            data=[],
        ).set_index(keys=["N_OBS", "dim", "seed", "eps"])
    records = []
    ref_sw_per_dim = {}
    for exp in tqdm.tqdm(data):
        index = (exp["N_OBS"], exp["dim"], exp["seed"], exp["eps"])
        try:
            all_data.loc[index]
        except KeyError:
            ref_samples = exp["ref_samples"]
            ref_sws = []
            for i in range(5):
                ind_1, ind_2 = torch.randint(
                    low=0, high=ref_samples.shape[0], size=(2, 1000)
                )
                sw = max_sliced_wasserstein_distance(
                    X_s=ref_samples[ind_1].cuda(),
                    X_t=ref_samples[ind_2].cuda(),
                    n_projections=10_000,
                ).item()
                ref_sws.append(sw)
            ref_sw_per_dim[(exp["dim"], exp["N_OBS"], exp["seed"])] = {
                "mean": np.mean(ref_sws),
                "std": np.std(ref_sws),
                "len": 5,
            }
            for name, alg_data in exp["exps"].items():
                for e in alg_data:
                    sw = max_sliced_wasserstein_distance(
                        X_s=ref_samples[:1_000].cuda(),
                        X_t=e["samples"].cuda(),
                        n_projections=10_000,
                    ).item()
                    records.append(
                        {
                            "N_OBS": exp["N_OBS"],
                            "dim": exp["dim"],
                            "eps": exp["eps"],
                            "seed": exp["seed"],
                            "alg": name,
                            "dt": e["dt"],
                            "sampling_steps": e["n_steps"],
                            "sw": sw,
                            "sw_norm": sw
                            - ref_sw_per_dim[(exp["dim"], exp["N_OBS"], exp["seed"])][
                                "mean"
                            ],
                        }
                    )
    if len(records) > 0:
        new_all_data = pd.DataFrame.from_records(records)
        all_data = pd.concat(
            (all_data, new_all_data.set_index(keys=["N_OBS", "dim", "seed", "eps"])),
            axis=0,
        )
        all_data.reset_index().to_csv(
            f"{destination_folder}/gaussian_mixture_exp_treated.csv", index=False
        )
    else:
        all_data = all_data.reset_index()
