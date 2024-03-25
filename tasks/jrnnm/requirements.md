Following installation steps are required to reproduce the experiemts for the JR-NMM SBI example:

Create conda environment:
`conda create --channel conda-forge -n jrnnm`

Prepare work environment:
- `nano ~/.bashrc` and write `export LC_ALL=fr_FR.UTF-8`
- `source ~/.bashrc`

Install packages with `conda`:
- r-devtools
- rpy2
- pytorch torchvision -c pytorch
- numpy
- -c r r-bh

Install `sdbmABC` package (for the simualtor model):
- run `Rscript -e "devtools::install_github('massimilianotamborrino/sdbmpABC')"`

Install the general dependencies of this repository via `pip install -e .`