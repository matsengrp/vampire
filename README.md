# vampire

## Install

Install dependencies:
```
conda create -n py36 python=3.6
source activate py36
conda install -y biopython click flake8 keras matplotlib pandas pydot pytest scikit-learn scons seaborn yapf
pip install nestly
```
Then:
```
git clone https://github.com/matsengrp/vampire.git
cd vampire
pip install .
```

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



## Running

Get a list of example commands by running `scons -n` inside the `vampire` directory.
Execute the commands by running `scons`.


## Contributors

* Original version (immortalized in the [`original` branch](https://github.com/matsengrp/vampire/tree/original)) by Kristian Davidsen.
* Pedantic rewrite, sconsery, extension, and additional models by Erick Matsen.
* Contributions from Phil Bradley, Will DeWitt, Jean Feng, Eli Harkins, and Branden Olson.


## Code styling

This project uses [YAPF](https://github.com/google/yapf) for code formatting with a format defined in `setup.cfg`.
You can easily run yapf on all the files with a call to `yapf -ir .` in the root directory.

Code also checked using [flake8](http://flake8.pycqa.org/en/latest/).
`# noqa` comments cause lines to be ignored by flake8.

