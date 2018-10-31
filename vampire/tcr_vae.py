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

import scipy.stats as stats

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


def logprob_of_obs_vect(probs, obs):
    """
    Calculate the log of probability of the observations.

    :param probs: a matrix with each row giving the probability of
        observations.
    :param obs: a matrix with each row one-hot-encoding an observation.

    Kristian implemented this as
        np.sum(np.log(np.matmul(probs, obs.T).diagonal()))
    but that's equivalent but harder to follow.
    """
    # Here axis=1 means sum across columns (the sum will be empty except for
    # the single nonzero entry).
    return np.sum(np.log(np.sum(probs * obs, axis=1)))


class TCRVAE:
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
        self.max_len = input_shape[0][0]

    @classmethod
    def of_json_file(cls, fname):
        """
        Build a TCRVAE from a parameter dictionary dumped to JSON.
        """
        with open(fname, 'r') as fp:
            return cls(**json.load(fp))

    @classmethod
    def of_directory(cls, path):
        """
        Build an TCRVAE from the information contained in a directory.

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

    def evaluate(self, x_df):
        """
        Wrapping Model.evaluate for this setting.

        :param x_df: A onehot encoded dataframe representing input sequences.

        :return: loss
        """
        data = cols_of_df(x_df)
        return self.vae.evaluate(
            x=data, y=data, batch_size=self.params['batch_size'])

    def encode(self, x_df):
        """
        Get the VAE encoding of a given collection of sequences x.

        :param x_df: A onehot encoded dataframe representing input sequences.

        :return: z_mean and z_sd, the embedding mean and standard deviation.
        """
        z_mean, z_log_var = self.encoder.predict(cols_of_df(x_df))
        z_sd = np.sqrt(np.exp(z_log_var))
        return z_mean, z_sd

    def decode(self, z):
        """
        Get the decoding of z in the latent space.

        """
        return self.decoder.predict(z)

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

    def log_p_of_x_importance_sample(self, x_df, out_ps):
        """
        One importance sample to calculate the probability of generating some
        observed x's by decoding from the prior on z.

        Say we just have one x. We want p(x), which we can calculate as the
        expectation of p(x|z) where z is drawn from p(z). Instead, we use
        importance sampling, calculating the expectation of

        p(x|z) (p(z) / q(z|x))

        where the ratio in parentheses is the importance weight.

        We emphasize that this is _one_ importance sample. Run this lots and
        take the average to get a good estimate.

        Stupid notes:
        * We could save time by only computing the encoding and the _obs
        variables once.
        * Perhaps there is some way to avoid looping like this?

        :param x_df: A onehot encoded dataframe representing input sequences.
        :param out_ps: An np array in which to store the importance sampled ps.
        """

        # We're going to be getting a one-sample estimate, so we want one slot
        # in our output array for each input sequence.
        assert (len(x_df) == len(out_ps))

        # Get encoding of x in the latent space.
        z_mean, z_sd = self.encode(x_df)
        # Get samples from q(z|x) in the latent space, one for each input x.
        z_sample = stats.norm.rvs(z_mean, z_sd)
        # These are decoded samples from z. They are, thus, probability vectors
        # that get sampled if we want to realize actual sequences.
        aa_probs, v_gene_probs, j_gene_probs = self.decode(z_sample)

        # Onehot-encoded observations.
        aa_obs, v_gene_obs, j_gene_obs = cols_of_df(x_df)

        # Loop over observations.
        for i in range(len(x_df)):
            log_p_x_given_z = \
                logprob_of_obs_vect(aa_probs[i], aa_obs[i]) + \
                np.log(np.sum(v_gene_probs[i] * v_gene_obs[i])) + \
                np.log(np.sum(j_gene_probs[i] * j_gene_obs[i]))
            # p(z)
            # Here we use that the PDF of a multivariate normal with
            # diagonal covariance is the product of the PDF of the
            # individual normal distributions.
            log_p_z = np.sum(stats.norm.logpdf(z_sample[i], 0, 1))
            # q(z|x)
            log_q_z_given_x = np.sum(
                stats.norm.logpdf(z_sample[i], z_mean[i], z_sd[i]))
            # Importance weight: p(z)/q(z|x)
            log_imp_weight = log_p_z - log_q_z_given_x
            # p(x|z) p(z) / q(z|x)
            out_ps[i] = log_p_x_given_z + log_imp_weight


# ### CLI ###


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_csv', type=click.File('r'))
@click.argument('test_csv', type=click.File('r'))
@click.argument('model_params_fname', type=click.File('w'))
@click.argument('best_weights_fname', type=click.Path(writable=True))
def train_tcr(train_csv, test_csv, model_params_fname, best_weights_fname):
    """
    Train the model, print out a model assessment, saving the best weights
    to best_weights_fname and the input model params to model_params_fname.
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

    tcr_vae = TCRVAE(input_shape=input_shape, batch_size=batch_size)
    tcr_vae.fit(train, epochs, validation_split, best_weights_fname)
    tcr_vae.assess_losses(train, test)
    tcr_vae.serialize_params(model_params_fname)


@cli.command()
@click.option(
    '--nsamples', default=500, help='Number of importance samples to use.')
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('test_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def importance(nsamples, params_json, model_weights, test_csv, out_csv):
    """
    Estimate the log generation probability of the sequences in test_csv on the
    VAE determined by params_json and model_weights.

    Spit the results into out_csv, one estimate per line.
    """

    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    df_x = conversion.unpadded_tcrbs_to_onehot(
        pd.read_csv(test_csv, usecols=['amino_acid', 'v_gene', 'j_gene']),
        v.max_len)

    log_p_x = np.zeros((nsamples, len(df_x)))
    click.echo(
        f"Calculating p(x) for {test_csv.name} via importance sampling...")

    with click.progressbar(range(nsamples)) as bar:
        for i in bar:
            v.log_p_of_x_importance_sample(df_x, log_p_x[i])

    avg = np.sum(log_p_x, axis=0)
    avg /= nsamples

    pd.DataFrame({'log_p_x': avg}).to_csv(out_csv, index=False)


if __name__ == '__main__':
    cli()
