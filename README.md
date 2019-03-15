# vampire

## Install

First [install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### vampire itself
Install dependencies:
```
conda create -n py36 python=3.6
source activate py36
conda install -y biopython click flake8 keras matplotlib pandas pydot pytest scikit-learn scons seaborn yapf
conda install -y -c conda-forge parallel
pip install exrex delegator.py nestly versioneer
```
Then:
```
git clone https://github.com/matsengrp/vampire.git
cd vampire
pip install .
```

### OLGA
The full SCons pipeline includes running [OLGA](https://github.com/zsethna/OLGA).
If you would like to run this component, we need a separate conda environment named `olga` because OLGA is a Python 2.7 program.
Create it as follows:

```
conda create -y -n olga python=2.7
source activate olga
conda install -y numpy
pip install olga
```

Test it by running

```
olga-compute_pgen --humanTRB CASSLGRDGGHEQYF
```

inside the `olga` conda environment.


### sumrep
As if it wasn't enough to have Python 2.7 and 3.6, to reproduce the full comparative analysis you also need R, the [sumrep](https://github.com/matsengrp/sumrep/) package, and its many dependencies.
You also need the `argparse` package.

sumrep is a submodule in the `vampire/R` directory.
To use it, do
```
git submodule update --init --recursive
```

There is a script `vampire/R/install_packages.sh` which should serve as a starting point for your installation adventure.



## Running

Get a list of example commands by running `scons -n` inside the `vampire` directory.
Execute the commands on example data by running `scons`.
You can run these in parallel using the `-j` flag for scons.

In order to run on your own data, use `python util.py split-repertoires` to split your repertoires into train and test.
This will make a JSON file pointing to various paths.
You can run the pipeline on those data by running `scons --data=/path/to/your/file.json`.


## Documentation

The documentation consists of

1. the example pipeline, which will give you commands to try
2. command line help, which is accessed for example via `tcr-vae --help` and `tcr-vae train --help`
3. lots of docstrings in the source code


## Limitations

* Our preprocessing scripts exclude TCRBJ2-5, which Adaptive annotates badly, and TCRBJ2-7, which appears to be problematic for OLGA.
* We use Adaptive gene names and sequences, but will extend to more flexible TCR gene sets in the future.


## Contributors

* Original version (immortalized in the [`original` branch](https://github.com/matsengrp/vampire/tree/original)) by Kristian Davidsen.
* Pedantic rewrite, sconsery, extension, and additional models by Erick Matsen.
* Contributions from Phil Bradley, Will DeWitt, Jean Feng, Eli Harkins, and Branden Olson.


## Code styling

This project uses [YAPF](https://github.com/google/yapf) for code formatting with a format defined in `setup.cfg`.
You can easily run yapf on all the files with a call to `yapf -ir .` in the root directory.

Code also checked using [flake8](http://flake8.pycqa.org/en/latest/).
`# noqa` comments cause lines to be ignored by flake8.

