import pandas as pd
import numpy as np


def aa_encoding_tensors(germline_cdr3_csv, aa_order, v_gene_list, j_gene_list, max_cdr3_len):
    """
    Build tensors that one-hot-encode the germline sequences that extend into the CDR3.
    V genes are left-aligned, while J genes are right-aligned.
    """
    aa_list = list(aa_order)
    aa_dict = {c: i for i, c in enumerate(aa_list)}
    v_gene_dict = {c: i for i, c in enumerate(v_gene_list)}
    j_gene_dict = {c: i for i, c in enumerate(j_gene_list)}

    # keep_default_na=False means empty cells are read as empty strings.
    cdr3_aas = pd.read_csv(germline_cdr3_csv, keep_default_na=False)

    d = {
        locus: {gene: seq['sequence'].iloc[0]
                for gene, seq in df.groupby('gene')}
        for locus, df in cdr3_aas.groupby(['locus'])
    }

    # Make sure that we have all of the desired genes in our dictionary.
    assert set(v_gene_list).issubset(d['TRBV'].keys())
    assert set(j_gene_list).issubset(d['TRBJ'].keys())

    v_gene_encoding = np.zeros((len(v_gene_list), max_cdr3_len, len(aa_list)))
    for gene in v_gene_list:
        seq = d['TRBV'][gene]
        gene_index = v_gene_dict[gene]
        for i, c in enumerate(seq):
            v_gene_encoding[gene_index, i, aa_dict[c]] = 1

    j_gene_encoding = np.zeros((len(j_gene_list), max_cdr3_len, len(aa_list)))
    for gene in j_gene_list:
        seq = d['TRBJ'][gene]
        gene_index = j_gene_dict[gene]
        # Here's how the right-aligned indexing works:
        start = max_cdr3_len - len(seq)
        for i, c in enumerate(seq):
            j_gene_encoding[gene_index, i + start, aa_dict[c]] = 1

    return (v_gene_encoding, j_gene_encoding)


def max_germline_aas(encoding):
    """
    Given an encoding vector like that made by aa_encoding_tensors, get the
    maximum number of germline-encoded amino acids.
    """
    return int(np.max(np.sum(encoding, axis=(1, 2))))
