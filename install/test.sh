set -eux

# The following two lines are only necessary if you are using sumrep. The second line checks if the sumrep dependencies are available.
git submodule update --init --recursive
SUMREP_PATH=$(git submodule | grep sumrep | cut -d ' ' -f3); /opt/conda/bin/conda run -n vampire R --vanilla --slave -e "library(devtools); devtools::load_all('$SUMREP_PATH')"

# Install vampire.
/opt/conda/bin/conda run -n vampire pip install .

# Run tests.
/opt/conda/bin/conda run -n vampire pytest

# Run the demo.
cd vampire/demo
/opt/conda/bin/conda run -n vampire sh demo.sh
