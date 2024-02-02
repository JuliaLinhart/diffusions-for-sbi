import torch
from ot.sliced import max_sliced_wasserstein_distance
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.lines import Line2D
import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


if __name__ == "__main__":
    destination_folder = "/mnt/data/gabriel/sbi"
    data = torch.load(f"{destination_folder}/gaussian_exp.pt")
    records = {}
    for exp in tqdm.tqdm(data):
        dim = exp["dim"]
        seed = exp["seed"]
        eps = exp["eps"]
        n_obs = exp["N_OBS"]
        if eps not in records:
            records[eps] = {}
        if dim not in records[eps]:
            records[eps][dim] = {}
        if seed not in records[eps][dim]:
            records[eps][dim][seed] = {}
        if n_obs not in records[eps][dim][seed]:
            records[eps][dim][seed][n_obs] = {
                "true_posterior_mean": exp["true_posterior_mean"].cpu(),
                "true_posterior_cov": exp["true_posterior_cov"].cpu(),
                "true_theta": exp["true_theta"].cpu(),
                "ddim": exp["DDIM"]["samples"][:, :, 0].cpu(),
                "GAUSS": exp["exps"]["GAUSS"][-1]["samples"].cpu(),
                "JAC": exp["exps"]["JAC"][-2]["samples"].cpu(),
                "Langevin": exp["exps"]["Langevin"][-2]["samples"].cpu(),
            }

    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("pca", PCA(n_components=2))])
    scatter_data = records[0][2][0]
    fig, axes = plt.subplots(3, 2, sharey=True, sharex=True)
    for n_obs in scatter_data.keys():
        print(scatter_data[n_obs].keys())
        for ax, (alg, alg_data) in zip(axes.flatten(), scatter_data[n_obs].items()):
            print(alg)
            if alg in ['GAUSS', 'JAC', 'Langevin']:
                ax.set_title(alg)
                ax.scatter(alg_data[:, 0], alg_data[:, 1], c=cm.get_cmap("rainbow")(n_obs / 90), label=n_obs, alpha=.2)
                #ax.set_xlim([-5, 5])
                #ax.set_ylim([-5, 5])
    fig.show()
