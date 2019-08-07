#!/bin/bash
set -eu

NSEQS=$1
OUTFILE=$2

# To avoid olga confirming over-write.
rm -f $OUTFILE
conda run -n olga olga-generate_sequences --humanTRB -o $OUTFILE -n $NSEQS
