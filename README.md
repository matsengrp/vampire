# vampire

## Install

```
conda create -n py36 python=3.6
source activate py36
conda install -y biopython click flake8 keras matplotlib pandas pydot pytest scikit-learn scons yapf
pip install nestly
```

## Code styling

This project uses [YAPF](https://github.com/google/yapf) for code formatting with the default (i.e. pep8) style.
You can easily run yapf on all the files with a call to `yapf -ir .` in the root directory.

Code also checked using [flake8](http://flake8.pycqa.org/en/latest/).
`# noqa` comments cause lines to be ignored by flake8.

