#!/bin/bash
# The following is commented out because of a conda issue https://github.com/ContinuumIO/anaconda-issues/issues/8838
# set -eu

NSEQS=$1
OUTFILE=$2

source activate olga
# To avoid olga confirming over-write.
rm -f $OUTFILE
olga-generate_sequences --humanTRB -o $OUTFILE -n $NSEQS
