# diffusions-for-sbi
Diffusion Generative Modeling and Posterior Sampling in Simulation-Based Inference

## Install requirements

- Create a conda environment with python version 3.10.
- Install packages via `pip install -e .` inside the repository folder.
- Additional requirements for the simulator-models from the `sbibm` and `jrnmm` tasks (see `requirements.md` in corresponding `tasks/<name>` folder)

## Code

### Score models and loss functions, diffused priors and samplers:
- `nse.py`: implementation of the conditional neural score estimator (`NSE`) class (with integrated `LANGEVIN` and `DDIM` sampler for single observation posterior inference) and corresponding (DSM) loss function.
- `vp_diffused_priors.py`: analytic diffused (uniform and gaussian) prior for a VP-SDE diffusion process.
- `tall_posterior_sampler.py`: implementation of our `GAUSS` and `JAC` methods.

### Utils:
- `sm_utils.py`: train function for `NSE` 
- `experiment_utils.py`: implementation of the metrics and other utilities for the result computation of all experiments.
- `plot_utils.py`: plotting styles for to reproduce figures from paper and functions to plot multivariate posterior (1D and 2D) histograms from samples

## Experiments

### Toy Models (cf. Section 4.1):
- To generate the raw data (samples and metadata), run the scripts `gen_gaussian_gaussian.py path_to_save` and `gen_mixt_gauss_gaussian.py path_to_save` where
`path_to_save` is the path one wants to save the raw files.
- To generate a CSV with sliced wasserstein, run `treat_gaussian_data.py path_to_save` and `treat_mixt_gaussian_data.py path_to_save` where `path_to_save` is as above.
This will create CSV files in `path_to_save`
- To generate the plots, run `plot_gaussian.py path_to_save` and run `plot_mixt_gaussian.py path_to_save` to reproduce the plots.
The plots will be saved in the figures folder in the root of the repository and the time table datas in the data repository.
- To reproduce the exact figures from the paper (with correct stlye) run the scripts `toy_example_gaussian_results.py` and `toy_example_gaussian_mixture_results.py`.

### SBIBM examples (cf. Section 4.2):

The script to reproduce experiments and generate figures are `sbibm_posterior_estomation.py` and `sbibm_results.py`:
- To train the score models run:
  ```
  python sbibm_posterior_estimation.py --run train --n_train <1000/3000/10000/30000> --lr <1e-3/1e-4> --task <lotka_volterra/sir/slcp>
  ```
  
- To sample from the approximate posterior for all observations (`num_obs = 1, ... 10`) and number of observations (`n_obs = 1,8,14,22,30`), run:
  ```
    python sbibm_posterior_estimation.py --run sample_all --n_train <1000/3000/10000/30000> --lr <1e-3/1e-4> --task <lotka_volterra/sir/slcp>
  ```
  and add the arguments `--cov_mode <GAUSS/JAC>` and `--langevin` with optional `--clip` to indicate which algorithm should be used.
  
- To generate reference posterior samples run, add the argument `--reference` or use the precomputed ones (see `results/sbibm/<task_name>/reference_posterior_samples/`)
  
- To produce the figures for the `sW` (resp. `MMD` or `MMD to Dirac`) metric, run:
  ```
  python sbibm_results.py --w_dist
  ```
  (resp `--mmd_dist` or `--dirac_dist`). Loss functions can be plotted using the `--losses` argument.

### JR-NMM example (cf. Section 4.3)

The script to reproduce experiments and generate figures are `jrnmm_posterior_estomation.py` and `jrnmm_results.py`:
- To train the score models run:
  ```
  python jrnmm_posterior_estimation.py --run train --lr <1e-3/1e-4> --theta_dim <3/4>
  ```
  
- To sample from the approximate posterior, run:
  ```
  python sbibm_posterior_estimation.py --run sample --lr <1e-3/1e-4> --theta_dim <3/4>
  ```
  and add the arguments `--cov_mode <GAUSS/JAC>` and `--langevin` with optional `--clip` to indicate which algorithm should be used.
  
- To reproduce the figures run `python jrnmm_results.py` with the argument `--dirac_dist` for the `MMD to Dirac` plots, `--pairplot` for the full pairplots with 1D and 2D histograms of the posterior, and `signle_vs_multi_obs` for the 1D histograms of the 3D case.
