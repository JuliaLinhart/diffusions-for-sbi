import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np


if __name__ == "__main__":
    import sys
    destination_folder = sys.argv[1]

    # Read all the data!
    all_data = pd.read_csv(f"{destination_folder}/gaussian_exp_treated.csv").reset_index()


    # Color Map for algorithms!
    def c_map_alg(alg):
        if alg == 'GAUSS':
            return cm.get_cmap('rainbow')(0.2)
        if alg == 'JAC':
            return cm.get_cmap('rainbow')(0.6)
        if alg == 'Langevin':
            return cm.get_cmap('rainbow')(1)
    # Marker map for n iterations
    marker_map = {50: ".",
                  150: "*",
                  400: "o",
                  1000: 'v'}


    # Make Tite table
    dim = 10
    n_obs = 32
    eps = 1e-2
    time_info = all_data.groupby(['dim', 'N_OBS', "sampling_steps", "alg", "eps"])[['dt', 'sw']].agg(lambda x: f'{x.mean():.2f} +/- {x.std()*1.96 / x.shape[0]**.5:.2f}').reset_index()
    time_data = all_data.pivot(index=['dim', 'N_OBS', 'eps', 'sampling_steps', 'seed'], columns='alg', values='dt')
    time_data = time_data.assign(speed_up_gauss = time_data.GAUSS / time_data.Langevin, speed_up_jac = time_data.JAC / time_data.Langevin)
    agg_time_data = time_data.groupby(['dim'])[['speed_up_gauss', 'speed_up_jac']].agg(lambda x: f'{x.mean():.2f} ± {x.std()*1.96 / x.shape[0]**.5:.2f}').reset_index()
    agg_time_data = agg_time_data.loc[agg_time_data['dim'] < 64]
    agg_time_data.reset_index().to_csv('data/speed_up_comparison.csv', index=False)
    #Load data of "equivalent speed"
    equivalent_speed_data = all_data.loc[(((all_data.alg == 'GAUSS')&(all_data.sampling_steps==1000))|
                                          ((all_data.alg == 'JAC') & (all_data.sampling_steps == 400)) |
                                          ((all_data.alg == 'Langevin') & (all_data.sampling_steps == 400)))
    ]
    # Remove eps high since it does not work!
    equivalent_speed_data = equivalent_speed_data.loc[(equivalent_speed_data.eps <= 1e-1)&(equivalent_speed_data.dim <64)]

    #Should we remove some dims as well? Maybe only plot one?

    # Plot Normalized Wasserstein cols = eps, rows = dims
    n_plots = len(equivalent_speed_data['eps'].unique())
    n_rows = len(equivalent_speed_data['dim'].unique())

    fig, axes_all = plt.subplots(n_rows, n_plots, sharex=True, sharey=True, squeeze=False, figsize=(1 + 4 * n_plots, 1 + 1.5*n_rows))
    fig.subplots_adjust(right=.995, top=.95, bottom=.08, hspace=0, wspace=0, left=.05)

    for ax, eps in zip(axes_all[0], equivalent_speed_data['eps'].unique()):
        ax.set_title(r"ϵ = " + f"{eps}", fontsize=20)
    for ax in axes_all[-1]:
        ax.set_xlabel('N Obs', fontsize=20)

    # Group data per dim (row)
    for axes, (dim, dim_data) in zip(axes_all, equivalent_speed_data.groupby(['dim'])):
        n_plots = len(dim_data['eps'].unique())
        axes[0].set_ylabel(f'd = {dim[0]} \n NsW', fontsize=20)
        for ax in axes.flatten():
            ax.set_ylim([-1e-1, 1e0])
        for ax in axes[1:]:
            ax.set_yticks([])
        #Group data per eps
        for (eps, eps_data), ax in zip(dim_data.groupby(['eps']), axes):
            #Group data per algo
            for (alg, sampling_steps), dt in eps_data.groupby(['alg', "sampling_steps"]):
                #PLOT!
                c = c_map_alg(alg)
                label=f'{alg} {sampling_steps}'
                lst = 'solid'
                ax.errorbar(x=dt.groupby(['N_OBS']).first().index,
                            y=dt.groupby(['N_OBS']).sw_norm.mean(),
                            yerr=1.96*dt.groupby(['N_OBS']).sw.std() / (dt.groupby(['N_OBS'])['seed'].count()**.5),
                            color=c, label=label, alpha=.8, marker='o', linestyle=lst,
                            capsize=10, elinewidth=2)
            #ax.set_yscale('log')
            ax.set_xscale('log')
            #ax.set_ylabel('Sliced Wasserstein')
            #ax.set_xlabel('Number of Observations')
            ax.set_xlim(1.5, 110)
    axes_all[-1, -1].legend()
    #axes[0].set_xlim(1.5, 100.5)
    fig.savefig(f'figures/gaussian/n_obs_vs_sw_per_eps_dim.pdf')
    fig.show()
    plt.close(fig)


    # Same thing but here cols are algs!
    n_cols = len(equivalent_speed_data['alg'].unique())
    n_rows = len(equivalent_speed_data['dim'].unique())
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False, figsize=(1 + 4 * n_cols, 1 + 1.5*n_rows))
    fig.subplots_adjust(right=.995, top=.95, bottom=.08, hspace=0, wspace=0, left=.05)
    for ax, d in zip(axes[:, 0], equivalent_speed_data['dim'].unique()):
        ax.set_ylabel(f'd = {d}  \n sW', fontsize=20)
    for ax in axes[-1, :]:
        ax.set_xlabel('N Obs', fontsize=20)
    for ax in axes[:, 1:].flatten():
        ax.set_yticks([])
    for ax, ((dim, alg), alg_data) in zip(axes.flatten(), equivalent_speed_data.groupby(['dim', 'alg'])):
        #     n_obs = [i[1] for i in ref_sw_per_dim.keys() if i[0] == dim[0]]
        if dim == 2:
            ax.set_title(f'{alg}', fontsize=20)

        print(alg)
        #ax.fill_between(n_obs, -np.array(yerr_ref), yerr_ref, color='red', alpha=.5)
        for eps, dt in alg_data.groupby(['eps']):
            label = eps[0]
            c = cm.get_cmap('rainbow')(-math.log10(eps[0] + 1e-5) / 5)
            ax.errorbar(x=dt.groupby(['N_OBS']).first().index,
                        y=dt.groupby(['N_OBS']).sw_norm.mean(),
                        yerr=1.96*dt.groupby(['N_OBS']).sw.std() / (dt.groupby(['N_OBS'])['seed'].count()**.5),
                        color=c, label=label, alpha=.8, marker='o', linestyle=lst,
                        capsize=10, elinewidth=2)
        ax.set_ylim([-1e-1, 1e0])
        ax.set_xscale('log')
        ax.set_xlim(1.5, 110)
    axes[-1, -1].legend()
    fig.savefig(f'figures/gaussian/n_obs_vs_sw_per_alg_dim.pdf')
    fig.show()
    plt.close(fig)