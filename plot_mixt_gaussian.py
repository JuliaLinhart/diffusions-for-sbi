import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np


if __name__ == '__main__':
    destination_folder = "/mnt/data/gabriel/sbi"

    # Read all the data!
    all_data = pd.read_csv(f"{destination_folder}/gaussian_mixture_exp_treated.csv").reset_index()

    #Stuff for plotting
    def c_map_alg(alg):
        if alg == 'GAUSS':
            return cm.get_cmap('rainbow')(0.2)
        if alg == 'JAC':
            return cm.get_cmap('rainbow')(0.6)
        if alg == 'Langevin':
            return cm.get_cmap('rainbow')(1)

    marker_map = {50: ".",
                  150: "*",
                  400: "o",
                  1000: 'v'}
    #Same, choosing data with equivalent speed
    equivalent_speed_data = all_data.loc[(((all_data.alg == 'GAUSS')&(all_data.sampling_steps==1000))|
                                         ((all_data.alg == 'JAC') & (all_data.sampling_steps == 400)) |
                                         ((all_data.alg == 'Langevin') & (all_data.sampling_steps == 400)))
                                         ]

    #Selecting only dim 10 and eps < 1
    equivalent_speed_data = equivalent_speed_data.loc[(equivalent_speed_data.dim == 10) & (all_data.eps < 1)]

    #plotting per epsilon
    n_plots = len(equivalent_speed_data['eps'].unique())
    fig, axes_all = plt.subplots(len(equivalent_speed_data['dim'].unique()), n_plots, sharex=True, sharey=True, squeeze=False, figsize=(1 + 4 * n_plots, 4))
    fig.subplots_adjust(right=.995, top=.92, bottom=.15, hspace=0, wspace=0, left=.08)
    for ax, eps in zip(axes_all[0], equivalent_speed_data['eps'].unique()):
        ax.set_title(r"ϵ = " + f"{eps}", fontsize=20)
    for ax in axes_all[-1]:
        ax.set_xlabel('N Obs', fontsize=20)
    axes_all[0, 0].set_ylabel(f'd = {10} \n sW', fontsize=20)
    #Group data by dim (row)
    for (eps, eps_data), ax in zip(equivalent_speed_data.groupby(['eps']), axes_all.flatten()):
        ax.set_ylim([-1e-1, 1e0])

        ax.set_yticks([])

    #Group by eps and iter over columns of axis
        #Group by alg to plot
        for (alg, sampling_steps), dt in eps_data.groupby(['alg', "sampling_steps"]):
            #Plotting
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
    fig.savefig(f'figures/gaussian_mixture/n_obs_vs_sw_per_eps_dim.pdf')
    fig.show()
    plt.close(fig)