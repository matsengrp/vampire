#!/bin/bash
set -eu

# INFILE should be a TSV with CDR3 amino acid sequence, V, and J genes.

INFILE=$1
OUTFILE=$2

# To avoid olga confirming over-write.
rm -f $OUTFILE
conda run -n olga olga-compute_pgen --display_off --humanTRB -i $INFILE -o $OUTFILE
