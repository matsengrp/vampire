import numpy as np
import tensorflow as tf

import vampire.layers as layers


def test_cumprod_sum():
    a = np.array([[[1, 1, 1, 0], [1, 0, 0, 0]], [[1, 1, 0, 1], [1, 0, 1, 1]], [[0, 1, 1, 1], [0, 0, 1, 1]]],
                 dtype=np.float32)

    def np_cumprod_sum(a, length, reverse=False):
        if reverse:
            a = np.flip(a, axis=1)
        return np.sum(np.cumprod(a[:, :length], axis=1), axis=1)

    for k in range(a.shape[0]):
        for length in range(1, 4):
            for reverse in [True, False]:
                with tf.Session() as sess:
                    tf_input = tf.constant(a[k, :, :], dtype=tf.float32)
                    tf_result = layers.cumprod_sum(tf_input, length, reverse=reverse)
                    tf_result = sess.run(tf_result)
                    assert np.allclose(np_cumprod_sum(a[k, :, :], length, reverse), tf_result)
