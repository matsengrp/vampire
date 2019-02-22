#!/bin/bash

# This wrapper converts an Adaptive sequence file to an OLGA TSV, runs OLGA's
# Pgen computation, and puts the result in a file with the original data.

# The following is commented out because of a conda issue https://github.com/ContinuumIO/anaconda-issues/issues/8838
# set -eux

INFILE=$1
OUTFILE=$2

TMPDIR=$(mktemp -d)
# Clean up after ourselves in all cases except `kill -9`.
trap "rm -rf $TMPDIR" EXIT
CONVERTED=$TMPDIR/converted.tsv
PGEN=$TMPDIR/pgen.tsv
SPLITDIR=$TMPDIR/split

# Use all the cores.
THREADS=$(nproc)

mkdir -p $SPLITDIR
gene-name-conversion adaptive2olga $INFILE $CONVERTED
# Using split with l/N preserves lines.
split -n l/$THREADS $CONVERTED $SPLITDIR/
parallel -j $THREADS olga-pgen.sh {} {}.pgen ::: $SPLITDIR/*
cat $SPLITDIR/*.pgen > $PGEN
test $(cat $PGEN | wc -l) -eq $(cat $CONVERTED | wc -l) || {
    echo "Files that should be the same size are not! Refusing to paste."
    exit 1
}
cut -f 2 $PGEN | paste $CONVERTED - > $OUTFILE
