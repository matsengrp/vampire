from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf


class EmbedViaMatrix(Layer):
    """
    This layer defines a (learned) matrix M such that given matrix input X the
    output is XM. The number of columns of M is embedding_dim, and the number
    of rows is set so that X and M can be multiplied.

    If the rows of the input give the coordinates of a series of objects, we
    can think of this layer as giving an embedding of each of the encoded
    objects in a embedding_dim-dimensional space.
    """

    def __init__(self, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        super(EmbedViaMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # The first component of input_shape is the batch size (see https://keras.io/layers/core/#dense).
        self.kernel = self.add_weight(
            name='kernel', shape=(input_shape[2], self.embedding_dim), initializer='uniform', trainable=True)
        super(EmbedViaMatrix, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding_dim)


class RightTensordot(Layer):
    """
    Given a numpy tensor Y, this layer tensordots input on the right with Y
    over `axes` numbers of coordinates.

    That is, if the input is X and axes=1, this performs
    sum_i X_{...,i} Y_{i,...}.
    """

    def __init__(self, right_np_tensor, axes, **kwargs):
        self.right_tf_tensor = tf.convert_to_tensor(right_np_tensor, dtype=tf.float32)
        assert type(axes) == int
        self.axes = axes
        super(RightTensordot, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RightTensordot, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # Tensordotting with sums over the last `axes` indices of the first argument
        # and the first `axes` indices of the second argument.
        return tf.tensordot(x, self.right_tf_tensor, axes=self.axes)

    def compute_output_shape(self, input_shape):
        # Make sure that the last `axes` dimensions of the first argument match
        # the first `axes` dimensions of the second argument.
        axes = self.axes
        assert input_shape[-axes:] == self.right_tf_tensor.shape[0:axes]
        right_remaining_shape = tuple(self.right_tf_tensor.shape[axes:])
        # If the whole right tensor gets consumed by the calculation, then the
        # resulting dimension for that component is 1, a scalar. In principle
        # we might have to do something equivalent on the left, but we always
        # have the batch_size that won't get contracted.
        if right_remaining_shape == ():
            right_remaining_shape = tuple([1])
        return tuple(input_shape[:-axes] + right_remaining_shape)


class CDR3Length(Layer):
    """
    Given a onehot-encoded CDR3 sequence, return the number of non-gap sites.

    Note that this layer requires that we have 20 amino acids, and that gap is
    the last state.
    """

    def __init__(self, **kwargs):
        super(CDR3Length, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CDR3Length, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # We want to count the sites for which gap is not the most likely state.
        return K.sum(  # Sum across sites
            # 20 - argmax will be >= 1 for any site that doesn't have gap in it.
            # Then we clip so it's = 1 for any site that doesn't have gap in it.
            # Note argmax with axis-=1 means over the last (amino acid) axis.
            tf.clip_by_value(20. - tf.to_float(tf.argmax(x, axis=-1)), 0., 1.),
            axis=1)

    def compute_output_shape(self, input_shape):
        # Make sure we have 21 states for our amino acid space.
        assert input_shape[-1] == 21
        # We are contracting two dimensions (positions and amino acids) and replacing with a scalar.
        return tuple(input_shape[:-2] + tuple([1]))


def cumprod(tensor, axis=0):
    """
    Cumulative product.

    There is a known problem with TF's cumprod, giving nans, so this is from
    https://github.com/tensorflow/tensorflow/issues/3862#issuecomment-327462205
    """
    transpose_permutation = None
    n_dim = len(tensor.get_shape())
    if n_dim > 1 and axis != 0:

        if axis < 0:
            axis = n_dim + axis

        transpose_permutation = np.arange(n_dim)
        transpose_permutation[-1], transpose_permutation[0] = 0, axis

    tensor = tf.transpose(tensor, transpose_permutation)

    def prod(acc, x):
        return acc * x

    prob = tf.scan(prod, tensor)
    tensor = tf.transpose(prob, transpose_permutation)
    return tensor


def cumprod_sum(a, length, reverse=False):
    """
    Given a matrix a, take sum_j prod_{i=0}^j a_{,,i} in TensorFlow.

    If reverse is True, flip the input across the last axis before the operations.

    See test for numpy equivalent.
    """
    last_dim = a.shape[-1]
    assert length <= last_dim
    if reverse:
        start = last_dim - 1
        it = range(start - 1, start - length, -1)
    else:
        start = 0
        it = range(1, length)
    cum = tf.identity(a[:, :, start])
    result = tf.identity(a[:, :, start])
    for i in it:
        cum *= a[:, :, i]
        result += cum
    return result


class ContiguousMatch(Layer):
    """
    """

    def __init__(self, **kwargs):
        super(ContiguousMatch, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ContiguousMatch, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        (x, v_germline_aa_onehot, j_germline_aa_onehot) = inputs

        def single_contiguous_match(single_x):
            return tf.convert_to_tensor([
                # The inner sum is across alternative germline genes.
                # K.prod(K.sum(tf.multiply(single_x, v_germline_aa_onehot), axis=1)),
                K.sum(cumprod(K.sum(tf.multiply(single_x, v_germline_aa_onehot), axis=1))),
                K.sum(tf.multiply(single_x, j_germline_aa_onehot))
                # The inner sum is across alternative germline genes.
                # K.sum(tf.cumprod(K.sum(tf.multiply(single_x, v_germline_aa_onehot), axis=1))),
                # K.sum(tf.cumprod(K.sum(tf.multiply(single_x, j_germline_aa_onehot), axis=1), reverse=True))
            ])

        return tf.map_fn(single_contiguous_match, x)

    def compute_output_shape(self, input_shape):
        # All input should be of shape (batch_size, max_cdr3_len, len(AA_LIST)).
        assert input_shape[0] == input_shape[1]
        assert input_shape[1] == input_shape[2]
        # We just return two numbers for every input.
        return tuple(input_shape[0][:-2] + tuple([2]))
