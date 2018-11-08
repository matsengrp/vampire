# vampire

## Install

```
conda create -n py36 python=3.6
source activate py36
conda install -y biopython click flake8 keras matplotlib pandas pydot pytest scikit-learn scons seaborn yapf
pip install nestly
```


## Running

Get a demo by running `scons` inside the `vampire` directory.


## Contributors

* Original version (immortalized in the [`original` branch](https://github.com/matsengrp/vampire/tree/original)) by Kristian Davidsen.
* Pedantic rewrite, sconsery, and extension by Erick Matsen.
* Contributions from Phil Bradley, Will DeWitt, Jean Feng, Eli Harkins, and Branden Olson.


## Code styling

This project uses [YAPF](https://github.com/google/yapf) for code formatting with a format defined in `setup.cfg`.
You can easily run yapf on all the files with a call to `yapf -ir .` in the root directory.

Code also checked using [flake8](http://flake8.pycqa.org/en/latest/).
`# noqa` comments cause lines to be ignored by flake8.

