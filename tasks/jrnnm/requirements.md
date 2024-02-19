Following installations are required for the `jrnnm` sbi-task:

create conda environment:
`conda create --channel conda-forge -n jrnnm`

prepare work environment:
- `nano ~/.bashrc` and write `export LC_ALL=fr_FR.UTF-8`
- `source ~/.bashrc`

install packages with `conda`:
- r-devtools
- rpy2
- pytorch torchvision -c pytorch
- numpy
- -c r r-bh

install `sdbmABC` package (simualtor):
run `Rscript -e "devtools::install_github('massimilianotamborrino/sdbmpABC')"`


install additional packages with `pip`:
- sbi
