import numpy as np
import pytest

import vampire.common as common
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


def test_cdr3_length_of_onehots():
    data = common.read_data_csv('adaptive-filter-test.correct.csv')
    lengths = data['amino_acid'].apply(len).apply(float)
    onehots = conversion.unpadded_tcrbs_to_onehot(data, 30)
    assert lengths.equals(conversion.cdr3_length_of_onehots(onehots['amino_acid']))


def test_contiguous_match_counts():
    v, j = conversion.adaptive_aa_encoding_tensors(30)
    v01_01 = v[0]
    j01_02 = j[1]
    v01_v02_mixture = (v[0] + v[1]) / 2

    v01_j02_matching = conversion.seq_to_onehot('CTSSQ-------------------NYGYTF')
    two_v01_match = conversion.seq_to_onehot('CTWSQ-------------------NYGYTF')
    three_j02_match = conversion.seq_to_onehot('CTSSQ-------------------NYAYTF')

    # Here we have a complete match for both genes.
    assert np.array_equal(conversion.contiguous_match_counts(v01_j02_matching, v01_01, j01_02), np.array([5., 6.]))
    # Here the V match is interrupted by a W instead of an S, and we can see the "contiguous" requirement working.
    assert np.array_equal(conversion.contiguous_match_counts(two_v01_match, v01_01, j01_02), np.array([2., 6.]))
    # Equivalent test for J.
    assert np.array_equal(conversion.contiguous_match_counts(three_j02_match, v01_01, j01_02), np.array([5., 3.]))
    # For the mixture, we have one residue that matches both, then one that
    # only matches V01, then another two that match both. You can see that in
    # the indicator decay from left to right.
    # Here are the indicator vectors before summing:
    # 1., 0.5, 0.5, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.
    assert np.array_equal(
        conversion.contiguous_match_counts(v01_j02_matching, v01_v02_mixture, j01_02), np.array([2.75, 6.]))


def test_contiguous_match_counts_df():
    test = conversion.unpadded_tcrbs_to_onehot(common.read_data_csv('adaptive-filter-test.correct.csv'), 30)
    v_germline_tensor, j_germline_tensor = conversion.adaptive_aa_encoding_tensors(30)
    result = conversion.contiguous_match_counts_df(test, v_germline_tensor, j_germline_tensor)

    assert np.array_equal(result[0], np.array([4., 6.]))
    assert np.array_equal(result[1], np.array([5., 5.]))
    assert np.array_equal(result[5], np.array([5., 0.]))
