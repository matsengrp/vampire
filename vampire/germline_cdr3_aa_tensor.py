import pandas as pd
import numpy as np

AA_ORDER = 'ABC'
TCRB_V_GENE_LIST = ['TCRBV01-01', 'TCRBV02-01']
TCRB_J_GENE_LIST = ['TCRBJ01-01', 'TCRBJ01-02', 'TCRBJ01-03']

x = 'data/germline-cdr3-aas.test.csv'


def make_aa_encoding_tensors(germline_cdr3_csv, aa_order, v_gene_list, j_gene_list, max_cdr3_len):
    """
    Build tensors that one-hot-encode the germline sequences that extend into the CDR3.
    V genes are left-aligned, while J genes are right-aligned.
    """
    aa_list = list(aa_order)
    aa_dict = {c: i for i, c in enumerate(aa_list)}
    v_gene_dict = {c: i for i, c in enumerate(v_gene_list)}
    j_gene_dict = {c: i for i, c in enumerate(j_gene_list)}

    cdr3_aas = pd.read_csv(germline_cdr3_csv)

    d = {
        locus: {gene: seq['sequence'].iloc[0]
                for gene, seq in df.groupby('gene')}
        for locus, df in cdr3_aas.groupby(['locus'])
    }
    assert set(v_gene_list) == set(d['TRBV'].keys())
    assert set(j_gene_list) == set(d['TRBJ'].keys())

    v_gene_encoding = np.zeros((len(v_gene_list), max_cdr3_len, len(aa_list)))
    for gene, seq in d['TRBV'].items():
        gene_index = v_gene_dict[gene]
        for i, c in enumerate(seq):
            v_gene_encoding[gene_index, i, aa_dict[c]] = 1

    j_gene_encoding = np.zeros((len(j_gene_list), max_cdr3_len, len(aa_list)))
    for gene, seq in d['TRBJ'].items():
        gene_index = j_gene_dict[gene]
        # Here's how the right-aligned indexing works:
        start = max_cdr3_len - len(seq)
        for i, c in enumerate(seq):
            j_gene_encoding[gene_index, i + start, aa_dict[c]] = 1

    return (v_gene_encoding, j_gene_encoding)
