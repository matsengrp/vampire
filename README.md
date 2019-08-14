# vampire

This is a package to fit and test variational autoencoder (VAE) models for T cell receptor sequences.


## Install

### Setting up your environment

[Conda](https://conda.io) is required to run the pipeline, although not strictly a dependency for fitting and using VAEs using this code.
To follow these instructions, [install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

The install process for the entire pipeline is documented in the Dockerfile.

However, if you simply want to train and use vampire models, you can only execute

    conda env create -f install/environment.yml

This will create a `vampire` Conda environment which you can use.
If you also want to be able to compare repertoires using [sumrep](https://github.com/matsengrp/sumrep/) you will need to run the R installation steps in the Dockerfile.
We also provide an `install/environment-olga.yml` to make a Conda environment in which one can run [OLGA](https://github.com/zsethna/OLGA/).

### Installation

After setting up your environment (if you followed the steps above you'll need to `conda activate vampire`),  and run

    pip install .

in the repository.


## Running

Get a list of example commands by running `scons -n` inside the `vampire` directory.
Execute the commands on example data by running `scons`.
You can run these in parallel using the `-j` flag for scons.
Note that this pipeline runs on a very small data set just for example purposes-- it does not give an appropriately trained model.

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
* Pedantic rewrite, sconsery, extension, additional models, and comparative evaluation by Erick Matsen.
* Contributions from Phil Bradley, Will DeWitt, Jean Feng, Eli Harkins, and Branden Olson.


## Code styling

This project uses [YAPF](https://github.com/google/yapf) for code formatting with a format defined in `setup.cfg`.
You can easily run yapf on all the files with a call to `yapf -ir .` in the root directory.

Code also checked using [flake8](http://flake8.pycqa.org/en/latest/).
`# noqa` comments cause lines to be ignored by flake8.

