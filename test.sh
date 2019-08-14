git clone https://github.com/matsengrp/vampire.git
cd vampire
git submodule update --init --recursive
SUMREP_PATH=$(git submodule | grep sumrep | cut -d ' ' -f3); /opt/conda/bin/conda run -n vampire R --vanilla --slave -e "library(devtools); devtools::load_all('$SUMREP_PATH')"
/opt/conda/bin/conda run -n vampire pip install .
cd vampire/
/opt/conda/bin/conda run -n vampire scons
