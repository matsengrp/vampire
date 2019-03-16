"""
Analogous to CDR3 length, we can deterministically find the contiguous string
of amino acids that match the corresponding V and J gene. Here we reconstruct
the one-hot vector that encodes this information and add a loss concerning how
well we are doing in matching it.

Here's something we can use to check to make sure that the VAE's assessment of
the contiguous match length is the same as manually computing it.

v = tcr_vae.TCRVAE.of_directory('.')
(cdr3_output, cdr3_length_output, v_gene_output, j_gene_output, contiguous_match_output) = v.decoder.predict(
    np.random.normal(0, 1, size=(10, 35)))
v_germline, j_germline = conversion.adaptive_aa_encoding_tensors(30)
onehot_df = conversion.avj_raw_triple_to_tcr_df(cdr3_output, v_gene_output, j_gene_output)
np.allclose(conversion.contiguous_match_counts_df(onehot_df, v_germline, j_germline), contiguous_match_output)

Model diagram with 35 latent dimensions and 100 dense nodes:
https://user-images.githubusercontent.com/112708/48671639-5ede4900-eae0-11e8-8361-95afc8f9f2f7.png
"""

import numpy as np

import keras
from keras.models import Model
from keras.layers import Activation, Add, Dense, Lambda, Input, Reshape
from keras import backend as K
from keras import objectives

import vampire.common as common
import vampire.xcr_vector_conversion as conversion
from vampire.custom_keras import BetaWarmup, CDR3Length, ContiguousMatch, EmbedViaMatrix, RightTensordot

from vampire.germline_cdr3_aa_tensor import max_germline_aas


def build(params):

    beta = K.variable(params['beta'])

    def sampling(args):
        """
        This function draws a sample from the multivariate normal defined by
        the latent variables.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(params['batch_size'], params['latent_dim']), mean=0.0, stddev=1.0)
        # Reparameterization trick!
        return (z_mean + K.exp(z_log_var / 2) * epsilon)

    def vae_cdr3_loss(io_encoder, io_decoder):
        """
        The loss function is the sum of the cross-entropy and KL divergence. KL
        gets a weight of beta.
        """
        # Here we multiply by the number of sites, so that we have a
        # total loss across the sites rather than a mean loss.
        xent_loss = params['max_cdr3_len'] * K.mean(
            objectives.categorical_crossentropy(io_encoder, io_decoder))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss *= beta
        return (xent_loss + kl_loss)

    def mean_squared_error_2d(io_encoder, io_decoder):
        def mse(i):
            return keras.losses.mean_squared_error(io_encoder[:, i], io_decoder[:, i])

        return mse(0) + mse(1)

    # Input:
    cdr3_input_shape = (params['max_cdr3_len'], params['n_aas'])
    cdr3_input = Input(shape=cdr3_input_shape, name='cdr3_input')
    cdr3_length_input = Input(shape=(1, ), name='cdr3_length_input')
    v_gene_input = Input(shape=(params['n_v_genes'], ), name='v_gene_input')
    j_gene_input = Input(shape=(params['n_j_genes'], ), name='j_gene_input')
    contiguous_match_input = Input(shape=(2, ), name='contiguous_match_input')

    # Encoding layers:
    cdr3_embedding = EmbedViaMatrix(params['aa_embedding_dim'], name='cdr3_embedding')(cdr3_input)
    cdr3_embedding_flat = Reshape([params['aa_embedding_dim'] * params['max_cdr3_len']],
                                  name='cdr3_embedding_flat')(cdr3_embedding)
    v_gene_embedding = Dense(params['v_gene_embedding_dim'], name='v_gene_embedding')(v_gene_input)
    j_gene_embedding = Dense(params['j_gene_embedding_dim'], name='j_gene_embedding')(j_gene_input)
    merged_embedding = keras.layers.concatenate(
        [cdr3_embedding_flat, cdr3_length_input, v_gene_embedding, j_gene_embedding, contiguous_match_input],
        name='merged_embedding')
    encoder_dense_1 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_1')(merged_embedding)
    encoder_dense_2 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_2')(encoder_dense_1)

    # Latent layers:
    z_mean = Dense(params['latent_dim'], name='z_mean')(encoder_dense_2)
    z_log_var = Dense(params['latent_dim'], name='z_log_var')(encoder_dense_2)

    # Decoding layers:
    z_l = Lambda(sampling, output_shape=(params['latent_dim'], ), name='z')
    decoder_dense_1_l = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_1')
    decoder_dense_2_l = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_2')
    cdr3_post_dense_flat_l = Dense(np.array(cdr3_input_shape).prod(), activation='linear', name='cdr3_post_dense_flat')
    cdr3_post_dense_reshape_l = Reshape(cdr3_input_shape, name='cdr3_post_dense')
    cdr3_output_l = Activation(activation='softmax', name='cdr3_output')
    v_gene_output_l = Dense(params['n_v_genes'], activation='softmax', name='v_gene_output')
    j_gene_output_l = Dense(params['n_j_genes'], activation='softmax', name='j_gene_output')

    post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var])))
    v_gene_output = v_gene_output_l(post_decoder)
    j_gene_output = j_gene_output_l(post_decoder)

    # Here's where we incorporate germline amino acid sequences into the output.
    germline_cdr3_tensors = conversion.adaptive_aa_encoding_tensors(params['max_cdr3_len'])
    (v_germline_cdr3_tensor, j_germline_cdr3_tensor) = germline_cdr3_tensors
    (v_max_germline_aas, j_max_germline_aas) = [max_germline_aas(g) for g in germline_cdr3_tensors]
    v_germline_cdr3_l = RightTensordot(v_germline_cdr3_tensor, axes=1, name='v_germline_cdr3')
    j_germline_cdr3_l = RightTensordot(j_germline_cdr3_tensor, axes=1, name='j_germline_cdr3')
    cdr3_length_output_l = CDR3Length(name='cdr3_length_output')
    contiguous_match_output_l = ContiguousMatch(v_max_germline_aas, j_max_germline_aas, name='contiguous_match_output')
    v_germline_cdr3 = v_germline_cdr3_l(v_gene_output)
    j_germline_cdr3 = j_germline_cdr3_l(j_gene_output)
    cdr3_output = cdr3_output_l(
        Add(name='cdr3_pre_activation')([
            cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(post_decoder)),
            Add(name='germline_cdr3')([v_germline_cdr3, j_germline_cdr3])
        ]))
    cdr3_length_output = cdr3_length_output_l(cdr3_output)
    contiguous_match_output = contiguous_match_output_l([cdr3_output, v_germline_cdr3, j_germline_cdr3])

    # Define the decoder components separately so we can have it as its own model.
    z_mean_input = Input(shape=(params['latent_dim'], ))
    decoder_post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_mean_input))
    decoder_v_gene_output = v_gene_output_l(decoder_post_decoder)
    decoder_j_gene_output = j_gene_output_l(decoder_post_decoder)
    decoder_v_germline_cdr3 = v_germline_cdr3_l(decoder_v_gene_output)
    decoder_j_germline_cdr3 = j_germline_cdr3_l(decoder_j_gene_output)
    decoder_cdr3_output = cdr3_output_l(
        Add(name='cdr3_pre_activation')([
            cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(decoder_post_decoder)),
            Add(name='germline_cdr3')([decoder_v_germline_cdr3, decoder_j_germline_cdr3])
        ]))
    decoder_cdr3_length_output = cdr3_length_output_l(decoder_cdr3_output)
    decoder_contiguous_match_output = contiguous_match_output_l(
        [decoder_cdr3_output, decoder_v_germline_cdr3, decoder_j_germline_cdr3])

    encoder = Model([cdr3_input, cdr3_length_input, v_gene_input, j_gene_input, contiguous_match_input],
                    [z_mean, z_log_var])
    decoder = Model(z_mean_input, [
        decoder_cdr3_output, decoder_cdr3_length_output, decoder_v_gene_output, decoder_j_gene_output,
        decoder_contiguous_match_output
    ])
    vae = Model([cdr3_input, cdr3_length_input, v_gene_input, j_gene_input, contiguous_match_input],
                [cdr3_output, cdr3_length_output, v_gene_output, j_gene_output, contiguous_match_output])
    vae.compile(
        optimizer="adam",
        loss={
            'cdr3_output': vae_cdr3_loss,
            'cdr3_length_output': keras.losses.mean_squared_error,
            'v_gene_output': keras.losses.categorical_crossentropy,
            'j_gene_output': keras.losses.categorical_crossentropy,
            'contiguous_match_output': mean_squared_error_2d
        },
        loss_weights={
            # Keep the cdr3_output weight to be 1. The weights are relative
            # anyhow, and buried inside the vae_cdr3_loss is a beta weight that
            # determines how much weight the KL loss has. If we keep this
            # weight as 1 then we can interpret beta in a straightforward way.
            "cdr3_length_output": 0.5524,
            "cdr3_output": 1,
            "j_gene_output": 0.1596,
            "v_gene_output": 0.9282,
            "contiguous_match_output": 0.05
        })

    callbacks = [BetaWarmup(beta, params['beta'], params['warmup_period'])]

    return {'encoder': encoder, 'decoder': decoder, 'vae': vae, 'callbacks': callbacks}


def prepare_data(x_df):
    cdr3_data, v_gene_data, j_gene_data = common.cols_of_df(x_df)
    cdr3_length_data = conversion.cdr3_length_of_onehots(x_df['amino_acid'])
    max_cdr3_len = cdr3_data.shape[1]
    v_germline_tensor, j_germline_tensor = conversion.adaptive_aa_encoding_tensors(max_cdr3_len)
    contiguous_match_data = conversion.contiguous_match_counts_df(x_df, v_germline_tensor, j_germline_tensor)
    return [cdr3_data, cdr3_length_data, v_gene_data, j_gene_data, contiguous_match_data]


def interpret_output(output):
    (cdr3_output, cdr3_length_output, v_gene_output, j_gene_output, contiguous_match_output) = output
    return (cdr3_output, v_gene_output, j_gene_output)
