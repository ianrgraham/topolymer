# Setup

First, create a `.env` file that specifies the `DATA_DIR`. You can look at `.env.in` as an example.  This is the path where input files will be searched for and output will be dumped. Also look at `workflows/Snakefile` for examples of how this will be used later.

If you have `direnv` installed, setting up the environment is as easy as running `direnv allow` in the project root. This will build a local conda environment and automatically enable it upon entering the project directory. If you don't have `direnv`, create a conda environment as you normally would using `conda env create -f environment.yml`. Additionally, execute `./build.sh`. This will build the utility to dump grid densities to text files.

# Usage

Once the environment is properly configured, a basic `snakemake` workflow is present in `workflows`. Within the directory, simply run

```
snakemake --cores <number-of-cores>
```

on your local machine. On the cluster a bit more setup is requied... I'll document that later. But for now the computations are really basic so doing things locally is fine.