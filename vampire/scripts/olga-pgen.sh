#!/bin/bash
# The following is commented out because of a conda issue https://github.com/ContinuumIO/anaconda-issues/issues/8838
# set -eu

# INFILE should be a TSV with CDR3 amino acid sequence, V, and J genes.

INFILE=$1
OUTFILE=$2

source activate olga
# To avoid olga confirming over-write.
rm -f $OUTFILE
olga-compute_pgen --display_off --humanTRB -i $INFILE -o $OUTFILE
