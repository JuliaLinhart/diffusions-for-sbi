import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


def gaussien_wasserstein(X1, X2):
    mean1 = torch.mean(X1, dim=1)
    mean2 = torch.mean(X2, dim=1)
    cov1 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X1)
    cov2 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X2)
    return torch.linalg.norm(mean1 - mean2, dim=-1)**2 + torch.linalg.matrix_norm(cov1 - cov2, dim=(-2, -1))**2



if __name__ == '__main__':
    all_data = torch.load(f"gaussian_comparison.pt")
    sw_table = []
    c_map = lambda x: cm.get_cmap("rainbow")(x/.75)
    for n_obs, data in all_data.items():
        ref_samples = data["ref_samples"]

        fig, ax = plt.subplots(1, 1)
        fig.suptitle(n_obs)
        gwas_langevin =  gaussien_wasserstein(ref_samples, data["GAFNER2023"]['samples'][1:])
        data["GAFNER2023"]["WDist"] = gwas_langevin
        sw_table.append({"method": "GAFNER2023",
                         "WDist": gwas_langevin[-1].item(),
                        "N_OBS": n_obs})
        ax.plot(gwas_langevin, label='GAFNER 2023')

        for exp in data["experiments"]:
            if True:#exp['cov_mode'] == 'GAUSS':
                gwas = gaussien_wasserstein(ref_samples, exp["samples"])
                exp["WDist"] = gwas
                linestyle = 'dashed' if exp["cov_mode"] == 'GAUSS' else 'solid'#exp["scale"] else 'solid'
                color = c_map(exp["warmup_alpha"])
                #markerstyle = '*' if exp["cov_mode"] == 'GAUSS' else 'o'
                label = f'({exp["warmup_alpha"]}, {exp["cov_mode"]})'
                if exp["scale"]:
                    ax.plot(gwas,
                            label=label,
                            linestyle=linestyle, c=color,
                            #marker=markerstyle
                            )
                    sw_table.append({"method": "OURS",
                                     "cov": exp["cov_mode"],
                                     "warmup_alpha": exp["warmup_alpha"],
                                     "scale": exp["scale"],
                                     "WDist": gwas[-1].item(),
                                     "N_OBS": n_obs})
        ax.set_yscale('log')
        #ax.set_xlim([800, 1000])
        ax.legend()
        fig.show()
        ax.set_xlim([800, 1000])
        fig.show()
        fig, ax = plt.subplots(1, 1)
        ax.scatter(*ref_samples[-1].T,
                   label=f'reference samples')
        for exp in data["experiments"]:
            if exp['cov_mode'] == 'GAUSS':
                ax.scatter(*exp['samples'][-1].T,
                           label=f'scale ({exp["warmup_alpha"]})' if exp['scale'] else f'no scale ({exp["warmup_alpha"]})')
        # ax.set_yscale('log')
        ax.legend()
        fig.show()

        print(pd.DataFrame.from_records(sw_table))