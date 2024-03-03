Following installation steps are required to reproduce experiments for the SBI benchmark examples:

Install additional packages via `pip`:
- `sbibm` and [follow instructions of `diffeqtorch`](https://github.com/sbi-benchmark/diffeqtorch#installation) for ODE-based models (SIR and Lotka-Volterra)
- `jax` and `numpyro` (for MCMC sampling of the reference posteriors)

Install the general dependencies of this repository via `pip install -e .`