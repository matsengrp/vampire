# vampire

## Setup

```
conda install biopython jupyter keras matplotlib scikit-learn
scp /fh/fast/matsen_e/data/dnnir/spurf_heavy_chain_AHo.fasta ..
```


## TCRA

### Plan 1

* Encode each sequence as a V, a J, and a CDR3 amino acid sequence.
* For the V and J genes, use an embedding layer to project a 1-hot encoding down to a lower-dimensional space.
* Align using midpoint gapping if using standard VAE. Encode gaps as their own symbol, or do 0-hot encoding.

### Plan 2

* Work with some representation of nucleotides or codons
* Encode trimming somehow: the obvious thing would be to use an integer input, but will this be sufficient to find non-monotonic relationships between the amount of trimming and the insertion sequence? If we are thinking of insertions as AAs, do we round the amount of trimming down to the nearest codon boundary and pretend that the insertion was exclusively responsible for the AA?
* How do we encode the insertion sequence: AA? NT? Codons? Multiple encodings? AA seems obvious from the functional level, but not for the rearrangement process.


## Architecture

1. Kristian's existing two-layer VAE
2. [VRNN](http://arxiv.org/abs/1506.02216)


## Evaluation

Evaluate using joint distribution of summary statistics:

* germline gene use
* AA frequencies (per-site and joint)
* biochemical properties
