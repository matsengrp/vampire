"""
Encoding TCRs as one-hot vectors and back again.
Someday this may do the same for BCRs.

The current gene names are for Adaptive data.
"""

import numpy as np
import pandas as pd

# ### Amino Acids ###

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY-'
AA_LIST = list(AA_ORDER)
AA_DICT = {c: i for i, c in enumerate(AA_LIST)}
AA_DICT_REV = {i: c for i, c in enumerate(AA_LIST)}
AA_SET = set(AA_LIST)


def seq_to_onehot(seq):
    v = np.zeros((len(seq), len(AA_SET)))
    for i, a in enumerate(seq):
        v[i][AA_DICT[a]] = 1
    return v


def onehot_to_seq(onehot):
    return ''.join([AA_DICT_REV[v.argmax()] for v in onehot])


# ### TCRB ###
# V genes:
TCRB_V_GENE_LIST = [
    'TCRBV01-01', 'TCRBV02-01', 'TCRBV03-01', 'TCRBV03-02', 'TCRBV04-01',
    'TCRBV04-02', 'TCRBV04-03', 'TCRBV05-01', 'TCRBV05-02', 'TCRBV05-03',
    'TCRBV05-04', 'TCRBV05-05', 'TCRBV05-06', 'TCRBV05-07', 'TCRBV05-08',
    'TCRBV06-01', 'TCRBV06-04', 'TCRBV06-05', 'TCRBV06-06', 'TCRBV06-07',
    'TCRBV06-08', 'TCRBV06-09', 'TCRBV07-01', 'TCRBV07-02', 'TCRBV07-03',
    'TCRBV07-04', 'TCRBV07-05', 'TCRBV07-06', 'TCRBV07-07', 'TCRBV07-08',
    'TCRBV07-09', 'TCRBV08-02', 'TCRBV09-01', 'TCRBV10-01', 'TCRBV10-02',
    'TCRBV10-03', 'TCRBV11-01', 'TCRBV11-02', 'TCRBV11-03', 'TCRBV12-01',
    'TCRBV12-02', 'TCRBV12-05', 'TCRBV13-01', 'TCRBV14-01', 'TCRBV15-01',
    'TCRBV16-01', 'TCRBV18-01', 'TCRBV19-01', 'TCRBV20-01', 'TCRBV21-01',
    'TCRBV22-01', 'TCRBV23-01', 'TCRBV23-or09_02', 'TCRBV25-01', 'TCRBV27-01',
    'TCRBV28-01', 'TCRBV29-01', 'TCRBV30-01', 'TCRBVA-or09_02'
]
TCRB_V_GENE_DICT = {c: i for i, c in enumerate(TCRB_V_GENE_LIST)}
TCRB_V_GENE_DICT_REV = {i: c for i, c in enumerate(TCRB_V_GENE_LIST)}
TCRB_V_GENE_SET = set(TCRB_V_GENE_LIST)


def vgene_to_onehot(v_gene):
    v = np.zeros(len(TCRB_V_GENE_SET))
    v[TCRB_V_GENE_DICT[v_gene]] = 1
    return v


def onehot_to_vgene(onehot):
    return TCRB_V_GENE_DICT_REV[onehot.argmax()]


# J genes:
TCRB_J_GENE_LIST = [
    'TCRBJ01-01', 'TCRBJ01-02', 'TCRBJ01-03', 'TCRBJ01-04', 'TCRBJ01-05',
    'TCRBJ01-06', 'TCRBJ02-01', 'TCRBJ02-02', 'TCRBJ02-03', 'TCRBJ02-04',
    'TCRBJ02-05', 'TCRBJ02-06', 'TCRBJ02-07'
]
TCRB_J_GENE_DICT = {c: i for i, c in enumerate(TCRB_J_GENE_LIST)}
TCRB_J_GENE_DICT_REV = {i: c for i, c in enumerate(TCRB_J_GENE_LIST)}
TCRB_J_GENE_SET = set(TCRB_J_GENE_LIST)


def jgene_to_onehot(j_gene):
    v = np.zeros(len(TCRB_J_GENE_SET))
    v[TCRB_J_GENE_DICT[j_gene]] = 1
    return v


def onehot_to_jgene(onehot):
    return TCRB_J_GENE_DICT_REV[onehot.argmax()]


def pad_middle(seq, desired_length):
    """
    Pad the middle of a sequence with gaps so that it is a desired length.
    Fail assertion if it's already longer than `desired_length`.
    """
    seq_len = len(seq)
    assert seq_len <= desired_length
    pad_start = seq_len // 2
    pad_len = desired_length - seq_len
    return seq[:pad_start] + '-' * pad_len + seq[pad_start:]


def unpadded_tcrbs_to_onehot(df, desired_length):
    """
    Translate a data frame of TCR betas written as (CDR3 sequence, V gene name,
    J gene name) into onehot-encoded format with CDR3 padding out to
    `desired_length`.
    If a CDR3 sequence exceeds `desired_length` this will fail an assertion.
    """

    return pd.DataFrame({
        'amino_acid':
        df['amino_acid'].apply(
            lambda s: seq_to_onehot(pad_middle(s, desired_length))),
        'v_gene':
        df['v_gene'].apply(vgene_to_onehot),
        'j_gene':
        df['j_gene'].apply(jgene_to_onehot)
    })


def onehot_to_padded_tcrbs(df):
    """
    Convert back from onehot encodings to TCR betas.
    """

    return pd.DataFrame({
        'amino_acid': df['amino_acid'].apply(onehot_to_seq),
        'v_gene': df['v_gene'].apply(onehot_to_vgene),
        'j_gene': df['j_gene'].apply(onehot_to_jgene)
    })