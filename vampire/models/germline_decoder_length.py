"""
This model provides an estimate of the germline-encoded CDR3 amino acid
sequence to the final step of the CDR3 decoder. This estimate is marginalized
over the probablistic weight assigned to the various V and J genes.

Model diagram with 35 latent dimensions and 100 dense nodes:
https://user-images.githubusercontent.com/112708/48358943-b7c95f80-e650-11e8-97e4-ed483ec7846a.png
"""

import numpy as np

import keras
from keras.models import Model
from keras.layers import Activation, Add, Dense, Lambda, Input, Reshape
from keras import backend as K
from keras import objectives

import vampire.common as common
import vampire.xcr_vector_conversion as conversion
from vampire.layers import EmbedViaMatrix, RightTensordot


def build(params):
    def sampling(args):
        """
        This function draws a sample from the multivariate normal defined by
        the latent variables.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(params['batch_size'], params['latent_dim']), mean=0.0, stddev=1.0)
        # Reparameterization trick!
        return (z_mean + K.exp(z_log_var / 2) * epsilon)

    def vae_loss(io_encoder, io_decoder):
        """
        The loss function is the sum of the cross-entropy and KL divergence.
        """
        # Notice that "objectives.categorical_crossentropy(io_encoder,
        # io_decoder)" is a vector so it is averaged using "K.mean":
        xent_loss = io_decoder.shape.num_elements() * K.mean(
            objectives.categorical_crossentropy(io_encoder, io_decoder))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss *= 1 / 4 * params['batch_size']  # Because we have four input/output pairs.
        return (xent_loss + kl_loss)

    # Input:
    cdr3_input_shape = (params['max_cdr3_len'], params['n_aas'])
    cdr3_input = Input(shape=cdr3_input_shape, name='cdr3_input')
    cdr3_length_input = Input(shape=(1, ), name='cdr3_length_input')
    v_gene_input = Input(shape=(params['n_v_genes'], ), name='v_gene_input')
    j_gene_input = Input(shape=(params['n_j_genes'], ), name='j_gene_input')

    # Encoding layers:
    cdr3_embedding = EmbedViaMatrix(params['aa_embedding_dim'], name='cdr3_embedding')(cdr3_input)
    cdr3_embedding_flat = Reshape([params['aa_embedding_dim'] * params['max_cdr3_len']],
                                  name='cdr3_embedding_flat')(cdr3_embedding)
    v_gene_embedding = Dense(params['v_gene_embedding_dim'], name='v_gene_embedding')(v_gene_input)
    j_gene_embedding = Dense(params['j_gene_embedding_dim'], name='j_gene_embedding')(j_gene_input)
    merged_embedding = keras.layers.concatenate(
        [cdr3_embedding_flat, cdr3_length_input, v_gene_embedding, j_gene_embedding], name='merged_embedding')
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
    cdr3_post_dense_l = Reshape(cdr3_input_shape, name='cdr3_post_dense')
    cdr3_output_l = Activation(activation='softmax', name='cdr3_output')
    v_gene_output_l = Dense(params['n_v_genes'], activation='softmax', name='v_gene_output')
    j_gene_output_l = Dense(params['n_j_genes'], activation='softmax', name='j_gene_output')

    v_gene_output = v_gene_output_l(decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var]))))
    j_gene_output = j_gene_output_l(decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var]))))

    # Here's where we incorporate germline amino acid sequences into the output.
    (v_germline_cdr3_tensor, j_germline_cdr3_tensor) = conversion.adaptive_aa_encoding_tensors(params['max_cdr3_len'])
    v_germline_cdr3_l = RightTensordot(v_germline_cdr3_tensor, axes=1, name='v_germline_cdr3')
    j_germline_cdr3_l = RightTensordot(j_germline_cdr3_tensor, axes=1, name='j_germline_cdr3')
    # TODO flag 20
    cdr3_length_output_l = RightTensordot(
        np.array([[1.] * 20 + [0.]] * params['max_cdr3_len']), axes=2, name='cdr3_length_output')
    # This untrimmed_cdr3 gives a probability-marginalized one-hot encoding of
    # what the cdr3 would look like if there was zero trimming and zero
    # insertion. The gaps in the middle don't get any hotness.
    cdr3_output = cdr3_output_l(
        Add(name='cdr3_pre_activation')([
            cdr3_post_dense_l(cdr3_post_dense_flat_l(decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var]))))),
            Add(name='germline_cdr3')([v_germline_cdr3_l(v_gene_output),
                                       j_germline_cdr3_l(j_gene_output)])
        ]))
    cdr3_length_output = cdr3_length_output_l(cdr3_output)

    # Define the decoder components separately so we can have it as its own model.
    z_mean_input = Input(shape=(params['latent_dim'], ))
    decoder_v_gene_output = v_gene_output_l(decoder_dense_2_l(decoder_dense_1_l(z_mean_input)))
    decoder_j_gene_output = j_gene_output_l(decoder_dense_2_l(decoder_dense_1_l(z_mean_input)))
    decoder_cdr3_output = cdr3_output_l(
        Add(name='cdr3_pre_activation')([
            cdr3_post_dense_l(cdr3_post_dense_flat_l(decoder_dense_2_l(decoder_dense_1_l(z_mean_input)))),
            Add(name='germline_cdr3')(
                [v_germline_cdr3_l(decoder_v_gene_output),
                 j_germline_cdr3_l(decoder_j_gene_output)])
        ]))
    decoder_cdr3_length_output = cdr3_length_output_l(decoder_cdr3_output)

    encoder = Model([cdr3_input, cdr3_length_input, v_gene_input, j_gene_input], [z_mean, z_log_var])
    decoder = Model(z_mean_input,
                    [decoder_cdr3_output, decoder_cdr3_length_output, decoder_v_gene_output, decoder_j_gene_output])
    vae = Model([cdr3_input, cdr3_length_input, v_gene_input, j_gene_input],
                [cdr3_output, cdr3_length_output, v_gene_output, j_gene_output])
    vae.compile(
        optimizer="adam",
        loss={
            'cdr3_output': vae_loss,
            'cdr3_length_output': keras.losses.poisson,
            'v_gene_output': vae_loss,
            'j_gene_output': vae_loss
        })

    return {'encoder': encoder, 'decoder': decoder, 'vae': vae}


def prepare_data(x_df):
    cdr3_data, v_gene_data, j_gene_data = common.cols_of_df(x_df)
    cdr3_length_data = conversion.cdr3_length_of_onehots(x_df['amino_acid'])
    return [cdr3_data, cdr3_length_data, v_gene_data, j_gene_data]


def interpret_output(output):
    (cdr3_output, cdr3_length_output, v_gene_output, j_gene_output) = output
    return (cdr3_output, v_gene_output, j_gene_output)
