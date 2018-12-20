#!/bin/bash
set -eux

NSEQS=1e8
OLGA_DEST=_ignore/olga-generated.tsv

mkdir -p _ignore
../../scripts/olga-generate.sh $NSEQS $OLGA_DEST
python ../../thymic_Q.py lvj-frequency --col-name model_P_lvj _ignore/olga-generated.tsv model-lvj-frequency.csv
bzip2 model-lvj-frequency.csv
rm $OLGA_DEST
