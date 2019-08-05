#! /bin/bash
set -eu

conda install -y -c anaconda glpk libgfortran libgfortran-ng libiconv r-base r-git2r
# I had to add the following line to get svglite to install. You may not need it.
conda install -y -c conda-forge r-gdtools
for PACKAGE in alakazam ape argparse cowplot CollessLike data.table devtools dplyr entropy HDMD hexbin jsonlite latex2exp magrittr Peptides pROC RecordLinkage shazam seqinr stringdist stringr svglite testthat textmineR yaml
do
    R --vanilla --slave -e "if(!require($PACKAGE)){install.packages('$PACKAGE', '$CONDA_PREFIX/lib/R/library', repos='https://cloud.r-project.org')}"
done
R --vanilla --slave -e "if(!require(Biostrings)){source('https://bioconductor.org/biocLite.R'); biocLite('Biostrings')}"

git submodule update --init --recursive
# The following is a silly hack to get the sumrep path. If it doesn't work just plug the path in.
SUMREP_PATH=$(git submodule | grep sumrep | cut -d ' ' -f3)
R --vanilla --slave -e "library(devtools); devtools::load_all('$SUMREP_PATH')"
