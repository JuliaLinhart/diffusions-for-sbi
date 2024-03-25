# diffusions-for-sbi

Offical Code for the paper "Diffusion posterior sampling for simulation-based inference in tall data settings".

## Requirements

- Create a conda environment with python version 3.10. 
- Install the repository as a python package with all its dependencies via `pip install -e .` inside the repository folder (this executes the `setup.py` file).
- Make sure you have a CUDA compatible PyTorch version (see [`torch` installation instructions](https://pytorch.org/get-started/previous-versions/)).
- For the SBI benchmark experiment (section 4.2), a CUDA compatible Jax version is required (see [`jax` installation instructions](https://jax.readthedocs.io/en/latest/installation.html)). 
- To run the Jansen and Rit simulator (section 4.3) please follow the instructions in `tasks/jrnnm/requirements.md`.

All experiments were run with `torch` version `2.1.0+cu118` and `jax` version `0.4.25+cuda12.cudnn89` installed via:
```
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
The `environment.yml` file details the environment (packages and versions) used to produce the results of all experiments.

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

The script to reproduce experiments and generate figures are `sbibm_posterior_estomation.py` and `sbibm_results_rebuttal.py`. The tasks for which the results are shown in the main paper are `task_name = lotka_volterra`, `sir` and `slcp`. We also computes results for the following tasks: `gaussian_linear`, `gaussian_mixture/_uniform`, `bernoulli_glm/_raw`, `two_moons`. They will be included in the Appendix of the camera ready version of the paper.

- To generate the training and reference data/posterior samples run:
  ```
  python sbibm_posterior_estimation.py --setup <all/train_data/reference_data/reference_posterior> --task <task_name>
  ```
  The data will be saved in the `tasks/sbibm/data/<task_name>` folder. Pre-computed data that was used for our experiments is available in this folder.

- To train the score models run:
  ```
  python sbibm_posterior_estimation.py --run train --n_train <1000/3000/10000/30000> --lr <1e-3/1e-4> --task <lotka_volterra/sir/slcp>
  ```
  The trained score models will be saved in `results/sbibm/<task_name>/`. Pre-trained models that were used for our experiments are available in this folder.

- To sample from the approximate posterior for all observations (`num_obs = 1, ... 25`) and number of observations (`n_obs = 1,8,14,22,30`), run:
  ```
    python sbibm_posterior_estimation.py --run sample_all --n_train <1000/3000/10000/30000> --lr <1e-3/1e-4> --task <lotka_volterra/sir/slcp>
  ```
  and add the arguments `--cov_mode <GAUSS/JAC>` and `--langevin` with optional `--clip` to indicate which algorithm should be used. The samples will be saved in the same folder as the used trained score model.

- To reproduce the figures for the `sW` (resp. `MMD` or `MMD to Dirac`) metric, first run
  ```
  python sbibm_results_rebuttal.py --compute_dist
  ```
  to compute the distances (this might take some time). Precomputed results can be found in `results/sbibm/<task_name>/metrics/`. 

  To quickly generate the figures, run 
  ```
  python sbibm_results_rebuttal.py --plot_dist --swd
  ```
  (resp `--mmd` or `--dirac`). 
  
- Use the `--losses` argument to plot the loss functions and the `--plot_samples` argument to visualize the reference and estimated posterior samples.

### JR-NMM example (cf. Section 4.3)

The script to reproduce experiments and generate figures are `jrnnm_posterior_estomation.py` and `jrnnm_results.py`. Results are saved in `results/jrnnm/<theta_dim>d/`.
- To train the score models run:
  ```
  python jrnnm_posterior_estimation.py --run train --lr <1e-3/1e-4> --theta_dim <3/4>
  ```
  
- To sample from the approximate posterior for `n_obs = 1,8,14,22,30`, run:
  ```
  python jrnnm_posterior_estimation.py --run sample_all --lr <1e-3/1e-4> --theta_dim <3/4>
  ```
  and add the arguments `--cov_mode <GAUSS/JAC>` and `--langevin` with optional `--clip` to indicate which algorithm should be used. Specifying `--run sample --n_obs 30 --single_obs` will generate results for each of the `n_obs=30` observation seperately.
  
- To reproduce the figures run `python jrnnm_results.py` with the argument `--dirac_dist` for the `MMD to Dirac` plots, `--pairplot` for the full pairplots with 1D and 2D histograms of the posterior, and `--single_multi_obs` for the 1D histograms in the 3D case.


### New features:

- **Classifier-free guidance:** it is possible to implicitly **learn the prior score** by randomly dropping the context variables during the training of the posterior score. This is useful in cases where the diffused prior score cannot be computed analytically. To do so, specify the "drop rate" in the `classifier_free_guidance` variable of the training function implemented in `sm_utils.py`.

These features were integrated into the scripts of the SBI benchmark experiment and results will be added to the Appendix of the paper. 

- To learn the prior score while training the posterior score model and then use it for sampling the tall posterior, run the `sbibm_posterior_estimation.py` script as explained above, and add the `--clf_free_guidance` argument in the command line. To generate results (compute metrics and plot figures) run the `sbibm_results_rebuttal.py`, again with added `--clf_free_guidance` argument.
