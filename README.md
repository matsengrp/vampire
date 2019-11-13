# vampire

[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/matsengrp/vampire.svg)](https://cloud.docker.com/u/matsengrp/repository/docker/matsengrp/vampire/general) &nbsp;
[![Travis CI status](https://travis-ci.org/matsengrp/vampire.svg?branch=master)](https://travis-ci.org/matsengrp/vampire)

This is a package to fit and test variational autoencoder (VAE) models for T cell receptor sequences.

It is described in the paper [_Deep generative models for T cell receptor protein sequences_](https://elifesciences.org/articles/46935) by Kristian Davidsen, Branden J Olson, William S DeWitt III, Jean Feng, Elias Harkins, Philip Bradley and Frederick A Matsen IV.


## Install

### Setting up your environment

[Conda](https://conda.io) is the canonical way to prepare your environment and is required run the pipeline, although not strictly a dependency for fitting and using VAEs using this code.
These instructions will assume that you have [installed Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

If you want to see the entire environment preparation process process, see the Dockerfile.
However, if you simply want to train and use vampire models, you can only execute

    conda env create -f install/environment.yml

This will create a `vampire` Conda environment which you can enter and use for running vampire.
If you also want to be able to compare repertoires using [sumrep](https://github.com/matsengrp/sumrep/) you will need to run the R installation steps in the Dockerfile.
We also provide an `install/environment-olga.yml` to make a Conda environment in which one can run [OLGA](https://github.com/zsethna/OLGA/).

### vampire installation

After setting up your environment (if you followed the steps above you'll need to `conda activate vampire`), and run

    pip install .

in the repository to install vampire.

If you want to use sumrep, see `install/test.sh` for additional install instructions.


## Running

To get started, check out the demonstration script in `vampire/demo/demo.sh`, which will show you how models and training parameters are specified.

To run the main pipeline on sample data, try running `scons -n` inside the `vampire` directory.
Execute the commands on example data by running `scons`.
You can run these in parallel using the `-j` flag for scons.
Note that this pipeline runs on a very small data set (mixing training and testing) just for example purposes-- it does not give an appropriately trained model.

In order to run on your own data, use `python util.py split-repertoires` to split your repertoires into train and test.
This will make a JSON file pointing to various paths.
You can run the pipeline on those data by running `scons --data=/path/to/your/file.json`.

Note that the frequency estimation pipeline is run using `scons --pipe=pipe_freq`.

### Cluster execution

The pipeline includes a `--clusters` flag that, if used, will attempt to submit jobs to a [SLURM](https://slurm.schedmd.com/overview.html) cluster with the specified name.
If you have access to a cluster with a different cluster scheduler, hopefully you can modify the `execute.py` script accordingly.


## Documentation

The documentation consists of

0. the demonstration script
1. the two pipelines, which will give you commands to try
2. command line help, which is accessed for example via `tcr-vae --help` and `tcr-vae train --help`
3. lots of docstrings in the source code

Please get in touch if anything isn't clear.


## Limitations

* Our preprocessing scripts exclude TCRBJ2-5, which Adaptive annotates badly, and TCRBJ2-7, which appears to be problematic for OLGA.
* We use Adaptive gene names and sequences, but will extend to more flexible TCR gene sets in the future.


## Contributors

* Original version (immortalized in the [`original` branch](https://github.com/matsengrp/vampire/tree/original)) by Kristian Davidsen.
* Pedantic rewrite, sconsery, extension, additional models, and comparative evaluation by Frederick "Erick" Matsen.
* Contributions from Phil Bradley, Will DeWitt, Jean Feng, Eli Harkins, and Branden Olson.


## Code styling

This project uses [YAPF](https://github.com/google/yapf) for code formatting with a format defined in `setup.cfg`.
You can easily run yapf on all the files with a call to `yapf -ir .` in the root directory.

Code also checked using [flake8](http://flake8.pycqa.org/en/latest/).
`# noqa` comments cause lines to be ignored by flake8.

