import numpy as np
import pkg_resources

from vampire.germline_cdr3_aa_tensor import aa_encoding_tensors


def test_aa_encoding_tensors():
    aa_order = 'ABC'
    v_gene_list = ['TCRBV01-01', 'TCRBV02-01']
    j_gene_list = ['TCRBJ01-01', 'TCRBJ01-02', 'TCRBJ01-03']
    germline_cdr3_csv = pkg_resources.resource_filename('vampire', 'data/germline-cdr3-aas.test.csv')
    max_cdr3_len = 4
    #                                    B         A
    #                                        C
    v_encoding_correct = np.array([[[0., 1., 0.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                   [[0., 0., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])

    #                                                            A                     C
    #                                                                          A
    #                                                                    C         B
    j_encoding_correct = np.array([[[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 1., 0.]]])

    (v_enc, j_enc) = aa_encoding_tensors(germline_cdr3_csv, aa_order, v_gene_list, j_gene_list, max_cdr3_len)
    assert np.array_equal(v_enc, v_encoding_correct)
    assert np.array_equal(j_enc, j_encoding_correct)
