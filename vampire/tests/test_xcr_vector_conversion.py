import pytest

import vampire.xcr_vector_conversion as conversion


def test_pad_middle():
    with pytest.raises(AssertionError):
        conversion.pad_middle('AAAAA', 2)
    assert 'A---B' == conversion.pad_middle('AB', 5)
    assert 'A--BC' == conversion.pad_middle('ABC', 5)
    assert 'AB' == conversion.pad_middle('AB', 2)
    assert '----' == conversion.pad_middle('---', 4)


def test_gene_conversion():
    for gene in conversion.TCRB_V_GENE_LIST:
        assert conversion.onehot_to_vgene(conversion.vgene_to_onehot(gene)) == gene
    for gene in conversion.TCRB_J_GENE_LIST:
        assert conversion.onehot_to_jgene(conversion.jgene_to_onehot(gene)) == gene


def test_aa_conversion():
    target = 'CASY'
    assert conversion.onehot_to_seq(conversion.seq_to_onehot(target)) == target
    target = 'C-SY'
    assert conversion.onehot_to_seq(conversion.seq_to_onehot(target)) == target
