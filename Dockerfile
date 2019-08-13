FROM matsengrp/vampire

RUN git submodule update --init --recursive
RUN /opt/conda/bin/conda run -n vampire ./install/install_R_packages.sh
RUN /opt/conda/bin/conda run -n vampire R --vanilla --slave -e 'install.packages("BiocManager",repos = "http://cran.us.r-project.org"); BiocManager::install("Biostrings")'

# The following uses a silly hack to get the sumrep path. If it doesn't work just plug the path in.
RUN SUMREP_PATH=$(git submodule | grep sumrep | cut -d ' ' -f3); /opt/conda/bin/conda run -n vampire R --vanilla --slave -e "library(devtools); devtools::load_all('$SUMREP_PATH')"
