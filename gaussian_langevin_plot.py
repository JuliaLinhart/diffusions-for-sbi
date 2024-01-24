import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    L, V = torch.linalg.eig(matrix)
    L = L.real
    V = V.real
    return V @ torch.diag_embed(L.pow(p)) @ torch.linalg.inv(V)


def gaussien_wasserstein(X1, X2):
    mean1 = torch.mean(X1, dim=1)
    mean2 = torch.mean(X2, dim=1)
    cov1 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X1)
    sqrtcov1 = _matrix_pow(cov1, .5)
    cov2 = torch.func.vmap(lambda x:  torch.cov(x.mT))(X2)
    covterm = torch.func.vmap(torch.trace)(cov1 + cov2 - 2 * _matrix_pow(sqrtcov1 @ cov2 @ sqrtcov1, .5))
    return (1*torch.linalg.norm(mean1 - mean2, dim=-1)**2 + 1*covterm)**.5


if __name__ == '__main__':
    all_data = torch.load(f"gaussian_comparison.pt")
    sw_table = []
    def c_map(x):
        if x == 'both':
            return cm.get_cmap("Reds")(.25)
        elif x == 'ddim':
            return cm.get_cmap("Reds")(.4)
        return cm.get_cmap("Reds")(.5)
    ref_samples_max_covs = []
    for n_obs, data in all_data.items():
        ref_samples = data["ref_samples"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax_was = axes[0, 0]
        ax_was_zoom = axes[0, 1]
        ax_samples = axes[1, 0]
        ax_samples_zoom = axes[1, 1]
        fig.suptitle(n_obs)
        ax_samples.scatter(*ref_samples[-1].T,
                           label=f'reference samples')
        ax_samples_zoom.scatter(*ref_samples[-1].T,
                                label=f'reference samples')
        ref_samples_max_covs.append(torch.linalg.eigvals(torch.cov(ref_samples[-1].mT)).real.max() ** .5)

        gwas_langevin =  gaussien_wasserstein(ref_samples, data["GAFNER2023"]['samples'][1:])
        data["GAFNER2023"]["WDist"] = gwas_langevin
        sw_table.append({"method": "GAFNER2023",
                         "WDist": gwas_langevin[-1].item(),
                         "N_OBS": n_obs,
                         "dt": data["GAFNER2023"]["dt"],
                         "max_std": torch.linalg.eigvals(torch.cov(data["GAFNER2023"]["samples"][-1].mT)).real.max() ** .5
        })
        for ax in [ax_was, ax_was_zoom]:
            ax.plot(gwas_langevin, label='GAFNER 2023')

        for exp in data["experiments"]:
            if True:#exp['cov_mode'] == 'GAUSS':
                gwas = gaussien_wasserstein(ref_samples, exp["samples"])
                exp["WDist"] = gwas
                linestyle = 'dashed' if exp["clip"] else 'solid'#exp["scale"] else 'solid'
                color = c_map(exp["logl_mode"])
                markerstyle = '*' if exp["cov_mode"] == 'GAUSS' else 'o'
                label = f'({exp["warmup_alpha"]}, {exp["logl_mode"]})'
                if True:#exp["scale"]:
                    for ax in [ax_was, ax_was_zoom]:
                        ax.plot(gwas,
                                label=label,
                                linestyle=linestyle, c=color,
                                #marker=markerstyle
                                )
                    max_std = torch.linalg.eigvals(torch.cov(exp["samples"][-1].mT)).real.max()**.5
                    sw_table.append({"method": "OURS",
                                     "cov_mode": exp["cov_mode"],
                                     "warmup_alpha": exp["warmup_alpha"],
                                     "logl_mode": exp["logl_mode"],
                                     #"scale": exp["scale"],
                                     "WDist": gwas[-1].item(),
                                     "N_OBS": n_obs,
                                     "clip": exp["clip"],
                                     "dt": exp["dt"],
                                     "max_std": max_std.item()})
                    for ax in [ax_samples, ax_samples_zoom]:
                        ax.scatter(*exp['samples'][-1].T,
                                   label=label,
                                   c=color,
                                   marker=markerstyle,
                                   alpha=.08)
        ax_was.set_yscale('log')
        ax_was_zoom.set_yscale('log')
        ax_was_zoom.set_xlim([950, 1000])
        ax_was.legend()
        std_ref = ref_samples[-1].std(axis=0)
        ax_samples_zoom.set_xlim(ref_samples[-1, :, 0].min() - std_ref[0]*3,
                                 ref_samples[-1, :, 0].max() + std_ref[0]*3)
        ax_samples_zoom.set_ylim(ref_samples[-1, :, 1].min() - std_ref[1]*3,
                                 ref_samples[-1, :, 1].max() + std_ref[1]*3)
        #ax_was.legend()
        fig.show()

        print(pd.DataFrame.from_records(sw_table))
        dt = pd.DataFrame.from_records(sw_table)

    plt.plot(ref_samples_max_covs, dt.loc[~(dt.method == 'OURS'),'WDist'], label='GAFNER2023')
    plt.plot(ref_samples_max_covs, dt.loc[(dt.method == 'OURS') & (dt.loc[:, "logl_mode"] == 'both')&(dt.cov_mode=='GAUSS')&(dt.warmup_alpha==0),'WDist'], label='both')
    plt.plot(ref_samples_max_covs, dt.loc[(dt.method == 'OURS') & (dt.loc[:, "logl_mode"] == 'ddim')&(dt.cov_mode=='GAUSS')&(dt.warmup_alpha==0),'WDist'], label='ddim')
    plt.plot(ref_samples_max_covs, dt.loc[(dt.method == 'OURS') & (dt.loc[:, "logl_mode"] == 'tweedie')&(dt.cov_mode=='GAUSS')&(dt.warmup_alpha==0),'WDist'], label='tweedie')
    #plt.plot(ref_samples_max_covs, dt.loc[(dt.method == 'OURS') & (~dt.loc[:, "clip"].astype(bool))&(dt.cov_mode=='GAUSS')&(dt.warmup_alpha==0),'WDist'], label='ours not clip')
    plt.ylabel('W2')
    plt.xlabel('Posterior max std')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # plt.plot( dt.loc[~(dt.method == 'OURS'), 'N_OBS'], dt.loc[~(dt.method == 'OURS'),'WDist'], label='GAFNER2023')
    # plt.plot(dt.loc[~(dt.method == 'OURS'), 'N_OBS'], dt.loc[(dt.method == 'OURS') & (dt.loc[:, "clip"].astype(bool))&(dt.cov_mode=='GAUSS')&(dt.warmup_alpha==0),'WDist'], label='ours clip')
    # plt.plot(dt.loc[~(dt.method == 'OURS'), 'N_OBS'], dt.loc[(dt.method == 'OURS') & (~dt.loc[:, "clip"].astype(bool))&(dt.cov_mode=='GAUSS')&(dt.warmup_alpha==0),'WDist'], label='ours not clip')
    # plt.ylabel('W2')
    # plt.xlabel('N Obs')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()