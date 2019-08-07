#! /bin/bash
set -eu

for PACKAGE in alakazam ape argparse cowplot CollessLike data.table devtools dplyr entropy HDMD hexbin jsonlite latex2exp magrittr Peptides pROC RecordLinkage shazam seqinr stringdist stringr svglite testthat textmineR yaml
do
    R --vanilla --slave -e "if(!require($PACKAGE)){install.packages('$PACKAGE', '$CONDA_PREFIX/lib/R/library', repos='https://cloud.r-project.org')}"
done
