import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation, Reshape
from keras import backend as K
from keras.engine.topology import Layer
from keras import objectives


class OnehotEmbedding2D(Layer):
    """
    This is an alternative to the normal keras embedding layer which works on
    categorical data. This provides the same functionality but on a onehot
    encoding of the categorical data. 2D refers to the input data being two
    dimensional.
    """

    def __init__(self, Nembeddings, **kwargs):
        self.Nembeddings = Nembeddings
        super(OnehotEmbedding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[2], self.Nembeddings),
            initializer='uniform',
            trainable=True)
        super(OnehotEmbedding2D,
              self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.Nembeddings)


def encoder_decoder_vae(input_shape,
                        batch_size=100,
                        embedding_output_dim=[21, 30, 13],
                        latent_dim=40,
                        dense_nodes=125):
    """
    Build us a encoder, a decoder, and a VAE!
    """

    max_len = input_shape[0][0]

    def sampling(args):
        """
        This function draws a sample from the multinomial defined by the latent
        variables.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
        return (z_mean + K.exp(z_log_var / 2) * epsilon)

    def vae_loss(io_encoder, io_decoder):
        """
        The loss function is the sum of the cross-entropy and KL divergence.
        """
        # Notice that "objectives.categorical_crossentropy(io_encoder,
        # io_decoder)" is a vector so it is averaged using "K.mean":
        xent_loss = io_decoder.shape.num_elements() * K.mean(
            objectives.categorical_crossentropy(io_encoder, io_decoder))
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss *= 1 / 3 * batch_size  # Because we have three input/output
        return (xent_loss + kl_loss)

    # Encoding layers:
    encoder_input_CDR3 = Input(shape=input_shape[0], name='onehot_CDR3')
    encoder_input_Vgene = Input(shape=input_shape[1], name='onehot_Vgene')
    encoder_input_Jgene = Input(shape=input_shape[2], name='onehot_Jgene')

    embedding_CDR3 = OnehotEmbedding2D(
        embedding_output_dim[0], name='CDR3_embedding')(encoder_input_CDR3)
    # AA_embedding = Model(encoder_input_CDR3, embedding_CDR3)
    embedding_CDR3_flat = Reshape([embedding_output_dim[0] * max_len],
                                  name='CDR3_embedding_flat')(embedding_CDR3)
    embedding_Vgene = Dense(
        embedding_output_dim[1], name='Vgene_embedding')(encoder_input_Vgene)
    # Vgene_embedding = Model(encoder_input_Vgene, embedding_Vgene)
    embedding_Jgene = Dense(
        embedding_output_dim[2], name='Jgene_embedding')(encoder_input_Jgene)
    # Jgene_embedding = Model(encoder_input_Jgene, embedding_Jgene)

    merged_input = keras.layers.concatenate(
        [embedding_CDR3_flat, embedding_Vgene, embedding_Jgene],
        name='flat_CDR3_Vgene_Jgene')
    dense_encoder1 = Dense(
        dense_nodes, activation='elu', name='encoder_dense_1')(merged_input)
    dense_encoder2 = Dense(
        dense_nodes, activation='elu', name='encoder_dense_2')(dense_encoder1)

    # Latent layers:
    z_mean = Dense(latent_dim, name='z_mean')(dense_encoder2)
    z_log_var = Dense(latent_dim, name='z_log_var')(dense_encoder2)

    encoder = Model(
        [encoder_input_CDR3, encoder_input_Vgene, encoder_input_Jgene],
        [z_mean, z_log_var])

    # Decoding layers:
    z = Lambda(
        sampling, output_shape=(latent_dim, ), name='reparameterization_trick'
    )  # This is the reparameterization trick
    dense_decoder1 = Dense(
        dense_nodes, activation='elu', name='decoder_dense_1')
    dense_decoder2 = Dense(
        dense_nodes, activation='elu', name='decoder_dense_2')

    decoder_out_CDR3 = Dense(
        np.array(input_shape[0]).prod(),
        activation='linear',
        name='flat_CDR_out')
    reshape_CDR3 = Reshape(input_shape[0], name='CDR_out')
    position_wise_softmax_CDR3 = Activation(
        activation='softmax', name='CDR_prob_out')
    decoder_out_Vgene = Dense(
        input_shape[1][0], activation='softmax', name='Vgene_prob_out')
    decoder_out_Jgene = Dense(
        input_shape[2][0], activation='softmax', name='Jgene_prob_out')

    decoder_output_CDR3 = position_wise_softmax_CDR3(
        reshape_CDR3(
            decoder_out_CDR3(
                dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))))
    decoder_output_Vgene = decoder_out_Vgene(
        dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))
    decoder_output_Jgene = decoder_out_Jgene(
        dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))

    # Define the decoding part separately:
    z_mean_generator = Input(shape=(latent_dim, ))
    decoder_generator_CDR3 = position_wise_softmax_CDR3(
        reshape_CDR3(
            decoder_out_CDR3(dense_decoder2(
                dense_decoder1(z_mean_generator)))))
    decoder_generator_Vgene = decoder_out_Vgene(
        dense_decoder2(dense_decoder1(z_mean_generator)))
    decoder_generator_Jgene = decoder_out_Jgene(
        dense_decoder2(dense_decoder1(z_mean_generator)))

    decoder = Model(z_mean_generator, [
        decoder_generator_CDR3, decoder_generator_Vgene,
        decoder_generator_Jgene
    ])

    vae = Model(
        [encoder_input_CDR3, encoder_input_Vgene, encoder_input_Jgene],
        [decoder_output_CDR3, decoder_output_Vgene, decoder_output_Jgene])
    vae.compile(optimizer="adam", loss=vae_loss)

    return (encoder, decoder, vae)
