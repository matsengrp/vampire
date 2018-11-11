"""
Kristian's original 2-layer VAE.
"""

import numpy as np

import keras
from keras.models import Model
from keras.layers import Activation, Dense, Lambda, Input, Reshape
from keras import backend as K
from keras import objectives

import common
from layers import EmbedViaMatrix


def build(params):
    def sampling(args):
        """
        This function draws a sample from the multivariate normal defined by
        the latent variables.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(params['batch_size'], params['latent_dim']), mean=0.0, stddev=1.0)
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
        kl_loss *= 1 / 3 * params['batch_size']  # Because we have three input/output
        return (xent_loss + kl_loss)

    cdr3_input_shape = (params['max_cdr3_len'], params['n_aas'])

    # Encoding layers:
    encoder_input_CDR3 = Input(shape=cdr3_input_shape, name='onehot_CDR3')
    encoder_input_Vgene = Input(shape=(params['n_v_genes'], ), name='onehot_Vgene')
    encoder_input_Jgene = Input(shape=(params['n_j_genes'], ), name='onehot_Jgene')

    embedding_CDR3 = EmbedViaMatrix(params['aa_embedding_dim'], name='CDR3_embedding')(encoder_input_CDR3)
    # AA_embedding = Model(encoder_input_CDR3, embedding_CDR3)
    embedding_CDR3_flat = Reshape([params['aa_embedding_dim'] * params['max_cdr3_len']],
                                  name='CDR3_embedding_flat')(embedding_CDR3)
    embedding_Vgene = Dense(params['v_gene_embedding_dim'], name='Vgene_embedding')(encoder_input_Vgene)
    # Vgene_embedding = Model(encoder_input_Vgene, embedding_Vgene)
    embedding_Jgene = Dense(params['j_gene_embedding_dim'], name='Jgene_embedding')(encoder_input_Jgene)
    # Jgene_embedding = Model(encoder_input_Jgene, embedding_Jgene)

    merged_input = keras.layers.concatenate([embedding_CDR3_flat, embedding_Vgene, embedding_Jgene],
                                            name='flat_CDR3_Vgene_Jgene')
    dense_encoder1 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_1')(merged_input)
    dense_encoder2 = Dense(params['dense_nodes'], activation='elu', name='encoder_dense_2')(dense_encoder1)

    # Latent layers:
    z_mean = Dense(params['latent_dim'], name='z_mean')(dense_encoder2)
    z_log_var = Dense(params['latent_dim'], name='z_log_var')(dense_encoder2)

    encoder = Model([encoder_input_CDR3, encoder_input_Vgene, encoder_input_Jgene], [z_mean, z_log_var])

    # Decoding layers:
    z = Lambda(sampling, output_shape=(params['latent_dim'], ), name='reparameterization_trick')
    dense_decoder1 = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_1')
    dense_decoder2 = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_2')

    decoder_out_CDR3 = Dense(np.array(cdr3_input_shape).prod(), activation='linear', name='flat_CDR_out')
    reshape_CDR3 = Reshape(cdr3_input_shape, name='CDR_out')
    position_wise_softmax_CDR3 = Activation(activation='softmax', name='CDR_prob_out')
    decoder_out_Vgene = Dense(params['n_v_genes'], activation='softmax', name='Vgene_prob_out')
    decoder_out_Jgene = Dense(params['n_j_genes'], activation='softmax', name='Jgene_prob_out')

    decoder_output_CDR3 = position_wise_softmax_CDR3(
        reshape_CDR3(decoder_out_CDR3(dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))))
    decoder_output_Vgene = decoder_out_Vgene(dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))
    decoder_output_Jgene = decoder_out_Jgene(dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))

    # Define the decoding part separately:
    z_mean_generator = Input(shape=(params['latent_dim'], ))
    decoder_generator_CDR3 = position_wise_softmax_CDR3(
        reshape_CDR3(decoder_out_CDR3(dense_decoder2(dense_decoder1(z_mean_generator)))))
    decoder_generator_Vgene = decoder_out_Vgene(dense_decoder2(dense_decoder1(z_mean_generator)))
    decoder_generator_Jgene = decoder_out_Jgene(dense_decoder2(dense_decoder1(z_mean_generator)))

    decoder = Model(z_mean_generator, [decoder_generator_CDR3, decoder_generator_Vgene, decoder_generator_Jgene])

    vae = Model([encoder_input_CDR3, encoder_input_Vgene, encoder_input_Jgene],
                [decoder_output_CDR3, decoder_output_Vgene, decoder_output_Jgene])
    vae.compile(optimizer="adam", loss=vae_loss)

    return {'encoder': encoder, 'decoder': decoder, 'vae': vae, 'train_model': vae}


def prepare_data(x_df):
    return common.cols_of_df(x_df)
