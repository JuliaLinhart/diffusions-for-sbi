# diffusions-for-sbi

Offical Code for the paper "Diffusion posterior sampling for simulation-based inference in tall data settings".

## Requirements

- Create a conda environment with python version 3.10. 
- Install the repository as a python package with all its dependencies via `pip install -e .` inside the repository folder. (This executes the `setup.py` file.)
- Make sure you have a CUDA compatible PyTorch version. You can check CUDA availability via `torch.cuda.is_available()`. If needed, [install the right Python version](https://pytorch.org/get-started/previous-versions/).
- For the sbi benchmark experiment (section 4.2), `jax` and `numpyro` are required (to generate data and sample the reference posterior via MCMC). Make sure your `jax` version is compatible with your CUDA environment (see [installation instructions](https://jax.readthedocs.io/en/latest/installation.html)). 
- To run the Jansen and Rit simulator (section 4.3) please follow the instructions in `tasks/jrnnm/requirements.md`.

The results were computed with PyTorch version `2.1.0+cu118` and jax verion `0.4.23+cuda11.cudnn86`.


## Code content

### Diffusion generative modeling and posterior sampling:
- `nse.py`: implementation of the conditional neural score estimator class (`NSE`) and corresponding loss function (`NSELoss`). The `NSE` class has integrated *LANGEVIN*, *DDIM* and *Predictor-Corrector* samplers for single observation and tall posterior inference tasks. The `ddim` method combined with the `factorized_score` corresponds to our algorithm (`GAUSS` and `JAC`), the `annealed_langevin_geffner` method corresponds to the `F-NPSE` method from [Geffner et al., 2023](https://arxiv.org/abs/2209.14249).
- `sm_utils.py`: train function for `NSE`.
- `tall_posterior_sampler.py`: implementation of our tall posterior sampler (`GAUSS` and `JAC` algorithms) and the `euler` sampler.
- `vp_diffused_priors.py`: analytic diffused (uniform and gaussian) prior for a VP-SDE diffusion process.

### Experiment utils:
- `embedding_nets.py`: implementation of some networks for the score model (used in the toy models experiments from section 4.1)
- `experiment_utils.py`: implementation of the metrics and other utilities for the result computation of all experiments.
- `plot_utils.py`: plotting styles for to reproduce figures from paper and functions to plot multivariate posterior pairplots from samples
- `tasks`: folder containing implementations of the simulator, prior and true posterior (if known) for each task (toy example from section 4.1, SBI benchmark examples from section 4.2 and the Jansen and Rit Neural Mass Model from section 4.3). Pre-computed training and reference data for the `sbibm` tasks can be found in the `tasks/sbibm/data` folder.

Other files include the scripts to run experiments and reproduce the figures from the paper as described below.

## Reproduce experiments and figures from the paper

### Toy Models (cf. Section 4.1):
- To generate the raw data (samples and metadata), run the scripts `gen_gaussian_gaussian.py path_to_save` and `gen_mixt_gauss_gaussian.py path_to_save` where
`path_to_save` is the path one wants to save the raw files.
- To generate a CSV with sliced wasserstein, run `treat_gaussian_data.py path_to_save` and `treat_mixt_gaussian_data.py path_to_save` where `path_to_save` is as above.
This will create CSV files in `path_to_save`.
- To reproduce the results from the paper (with correct stlye) run the scripts `toy_example_gaussian_results.py path_to_save` and `toy_example_gaussian_mixture_results.py path_to_save`. The figures will be saved in the `figures/` folder and the time table datas in the `data/` folder in the `path_to_save` directory (here `results/toy_models/<gaussian/gaussian_mixture>/`).

### SBIBM examples (cf. Section 4.2):

The script to reproduce experiments and generate figures are `sbibm_posterior_estomation.py` and `sbibm_results_rebuttal.py`:
- To generate the training and reference data/posterior samples run:
  ```
  python sbibm_posterior_estimation.py --setup <all/train_data/reference_data/reference_posterior> --task <task_name>
  ```
  The data will be saved in the `tasks/sbibm/data/<task_name>` folder. 
- To train the score models run:
  ```
  python sbibm_posterior_estimation.py --run train --n_train <1000/3000/10000/30000> --lr <1e-3/1e-4> --task <lotka_volterra/sir/slcp>
  ```
  The trained score models will be saved in `results/sbibm`.
  
- To sample from the approximate posterior for all observations (`num_obs = 1, ... 25`) and number of observations (`n_obs = 1,8,14,22,30`), run:
  ```
    python sbibm_posterior_estimation.py --run sample_all --n_train <1000/3000/10000/30000> --lr <1e-3/1e-4> --task <lotka_volterra/sir/slcp>
  ```
  and add the arguments `--cov_mode <GAUSS/JAC>` and `--langevin` with optional `--clip` to indicate which algorithm should be used.

- To reproduce the figures for the `sW` (resp. `MMD` or `MMD to Dirac`) metric, run:
  ```
  python sbibm_results_rebuttal.py --compute_dist
  ```
  to compute the distances (this might take some time) and then 
  ```
  python sbibm_results_rebuttal.py --plot_dist --swd
  ```
  (resp `--mmd` or `--dirac`) to quickly generate the figures. Loss functions can be plotted using the `--losses` argument. To visualize the reference and estimated posterior samples, you can specify the `--plot_samples` argument.

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
  
- To reproduce the figures run `python jrnmm_results.py` with the argument `--dirac_dist` for the `MMD to Dirac` plots, `--pairplot` for the full pairplots with 1D and 2D histograms of the posterior, and `--single_multi_obs` for the 1D histograms in the 3D case.
