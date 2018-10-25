"""
Encoding TCRs as one-hot vectors and back again.
Someday this may do the same for BCRs.

The current gene names are for Adaptive data.
"""

import numpy as np

# ### Amino Acids ###

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY-'
AA_LIST = list(AA_ORDER)
AA_DICT = {c: i for i, c in enumerate(AA_LIST)}
AA_DICT_REV = {i: c for i, c in enumerate(AA_LIST)}
AA_SET = set(AA_LIST)

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

# J genes:
TCRB_J_GENE_LIST = [
    'TCRBJ01-01', 'TCRBJ01-02', 'TCRBJ01-03', 'TCRBJ01-04', 'TCRBJ01-05',
    'TCRBJ01-06', 'TCRBJ02-01', 'TCRBJ02-02', 'TCRBJ02-03', 'TCRBJ02-04',
    'TCRBJ02-05', 'TCRBJ02-06', 'TCRBJ02-07'
]
TCRB_J_GENE_DICT = {c: i for i, c in enumerate(TCRB_J_GENE_LIST)}
TCRB_J_GENE_DICT_REV = {i: c for i, c in enumerate(TCRB_J_GENE_LIST)}
TCRB_J_GENE_SET = set(TCRB_J_GENE_LIST)


def tcrb2onehot(TCRB_list):
    """
    Translate a list of TCR betas into onehot encodings.
    NB. all CDR3 sequences must be of equal length.
    """
    seqlen = len(TCRB_list[0][0])
    assert (not [True for s in TCRB_list if len(s[0]) != seqlen])
    onehot_seq = np.zeros((len(TCRB_list), seqlen, len(AA_SET)))
    onehot_vgene = np.zeros((len(TCRB_list), len(TCRB_V_GENE_SET)))
    onehot_jgene = np.zeros((len(TCRB_list), len(TCRB_J_GENE_SET)))
    for i, el in enumerate(TCRB_list):
        seq, vgene, jgene = el
        onehot_vgene[i][TCRB_V_GENE_DICT[vgene]] = 1
        onehot_jgene[i][TCRB_J_GENE_DICT[jgene]] = 1
        for j, a in enumerate(seq):
            onehot_seq[i][j][AA_DICT[a]] = 1
    return (onehot_seq, onehot_vgene, onehot_jgene)


def onehot2tcrb(onehot_seq, onehot_vgene, onehot_jgene):
    """
    Convert back from onehot encodings to TCR betas.
    """
    TCRB_list = list()
    for i in range(onehot_seq.shape[0]):
        seq = list()
        for j in range(onehot_seq.shape[1]):
            seq.append(AA_DICT_REV[onehot_seq[i][j].argmax()])
        seq = ''.join(seq)
        vgene = TCRB_V_GENE_DICT_REV[onehot_vgene[i].argmax()]
        jgene = TCRB_J_GENE_DICT_REV[onehot_jgene[i].argmax()]
        TCRB_list.append((seq, vgene, jgene))
    return (TCRB_list)
