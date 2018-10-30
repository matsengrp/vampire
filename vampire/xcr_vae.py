import click
import json
import numpy as np
import os
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation, Reshape
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras import objectives

import vampire.xcr_vector_conversion as conversion


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


def encoder_decoder_vae(input_shape, batch_size, embedding_output_dim,
                        latent_dim, dense_nodes):
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


def cols_of_df(df):
    """
    Extract the data columns of a dataframe into a list of appropriately-sized
    numpy arrays.
    """
    return [np.stack(col.values) for _, col in df.items()]


class XCRVAE:
    def __init__(
            self,
            *,  # Forces everything after this to be a keyword argument.
            input_shape,
            batch_size=100,
            embedding_output_dim=[21, 30, 13],
            latent_dim=40,
            dense_nodes=125):
        kwargs = dict(locals())
        kwargs.pop('self')
        (self.encoder, self.decoder, self.vae) = encoder_decoder_vae(**kwargs)
        self.params = kwargs

    @classmethod
    def of_json_file(cls, fname):
        """
        Build a XCRVAE from a parameter dictionary dumped to JSON.
        """
        with open(fname, 'r') as fp:
            return cls(**json.load(fp))

    @classmethod
    def of_directory(cls, path):
        """
        Build an XCRVAE from the information contained in a directory.

        By convention we are dumping information to a parameter file called
        `model_params.json` and a weights file called `best_weights.h5`. Here
        we load that information in.
        """
        v = cls.of_json_file(os.path.join(path, 'model_params.json'))
        v.vae.load_weights(os.path.join(path, 'best_weights.h5'))
        return v

    def serialize_params(self, fp):
        """
        Dump model parameters to a file.
        """
        return json.dump(self.params, fp)

    def fit(self,
            df: pd.DataFrame,
            epochs: int,
            validation_split: float,
            best_weights_fname: str,
            patience=10):
        """
        Fit the model with early stopping.
        """
        data = cols_of_df(df)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        save_best_weights = ModelCheckpoint(
            best_weights_fname,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min')
        self.vae.fit(
            x=data,  # y=X for a VAE.
            y=data,
            epochs=epochs,
            batch_size=self.params['batch_size'],
            validation_split=validation_split,
            callbacks=[early_stopping, save_best_weights])

    def evaluate(self, df):
        """
        Wrapping Model.evaluate for this setting.
        """
        data = cols_of_df(df)
        return self.vae.evaluate(
            x=data, y=data, batch_size=self.params['batch_size'])

    def assess_losses(self, train, test):
        """
        Print out the losses on the train vs. the hold out test set.
        """
        trainset_loss = self.evaluate(train)
        testset_loss = self.evaluate(test)

        print("Component-wise loss Train vs. Test:")
        for i in [1, 2, 3]:
            print('{}: {:.2f} vs. {:.2f}'.format(self.vae.metrics_names[i],
                                                 float(trainset_loss[i]),
                                                 float(testset_loss[i])))
        print('# Sum of losses #\nTrain set: {:.2f}\nTest set: {:.2f}'.format(
            float(trainset_loss[0]), float(testset_loss[0])))
        print('# Difference of summed of losses #\ntest-train : {:.2f}'.format(
            float(testset_loss[0]) - float(trainset_loss[0])))


# ### CLI ###


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_csv', type=click.File('r'))
@click.argument('test_csv', type=click.File('r'))
@click.argument('best_weights_fname', type=click.Path(writable=True))
@click.argument('model_params_fname', type=click.File('w'))
def train_tcr(train_csv, test_csv, best_weights_fname, model_params_fname):
    """
    Train the model, print out a model assessment, saving the best weights
    to BEST_WEIGHTS_FNAME and the input model params to MODEL_PARAMS_FNAME.
    """

    # TODO: less stupid
    MAX_LEN = 30
    epochs = 300
    validation_split = 0.1
    validation_split_multiplier = 10
    sub_chunk_size = validation_split * validation_split_multiplier
    # If this fails then we may have problems with chunks of the data being the
    # wrong length.
    assert sub_chunk_size == float(int(sub_chunk_size))
    batch_size = 100
    min_data_size = validation_split_multiplier * batch_size
    input_shape = [(MAX_LEN, len(conversion.AA_LIST)),
                   (len(conversion.TCRB_V_GENE_LIST), ),
                   (len(conversion.TCRB_J_GENE_LIST), )]

    def get_data(fname):
        df = pd.read_csv(fname, usecols=['amino_acid', 'v_gene', 'j_gene'])
        assert len(df) >= min_data_size
        # If we deliver chunks of data to Keras of min_data_size then it will
        # be able to split them into its internal train and test sets for
        # val_loss. Here we trim off the extra that won't fit into such a
        # setup.
        n_to_take = len(df) - len(df) % min_data_size
        return conversion.unpadded_tcrbs_to_onehot(df[:n_to_take], MAX_LEN)

    train = get_data(train_csv)
    test = get_data(test_csv)

    tcr_vae = XCRVAE(input_shape=input_shape, batch_size=batch_size)
    tcr_vae.fit(train, epochs, validation_split, best_weights_fname)
    tcr_vae.assess_losses(train, test)
    tcr_vae.serialize_params(model_params_fname)


if __name__ == '__main__':
    cli()
