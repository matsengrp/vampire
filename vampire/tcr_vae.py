import click
import json
import numpy as np
import os
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation, Reshape
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer
from keras import objectives

import scipy.special as special
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
            name='kernel', shape=(input_shape[2], self.Nembeddings), initializer='uniform', trainable=True)
        super(OnehotEmbedding2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.Nembeddings)


def encoder_decoder_vae(input_shape, batch_size, embedding_output_dim, latent_dim, dense_nodes):
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
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
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
        kl_loss *= 1 / 3 * batch_size  # Because we have three input/output
        return (xent_loss + kl_loss)

    # Encoding layers:
    encoder_input_CDR3 = Input(shape=input_shape[0], name='onehot_CDR3')
    encoder_input_Vgene = Input(shape=input_shape[1], name='onehot_Vgene')
    encoder_input_Jgene = Input(shape=input_shape[2], name='onehot_Jgene')

    embedding_CDR3 = OnehotEmbedding2D(embedding_output_dim[0], name='CDR3_embedding')(encoder_input_CDR3)
    # AA_embedding = Model(encoder_input_CDR3, embedding_CDR3)
    embedding_CDR3_flat = Reshape([embedding_output_dim[0] * max_len], name='CDR3_embedding_flat')(embedding_CDR3)
    embedding_Vgene = Dense(embedding_output_dim[1], name='Vgene_embedding')(encoder_input_Vgene)
    # Vgene_embedding = Model(encoder_input_Vgene, embedding_Vgene)
    embedding_Jgene = Dense(embedding_output_dim[2], name='Jgene_embedding')(encoder_input_Jgene)
    # Jgene_embedding = Model(encoder_input_Jgene, embedding_Jgene)

    merged_input = keras.layers.concatenate([embedding_CDR3_flat, embedding_Vgene, embedding_Jgene],
                                            name='flat_CDR3_Vgene_Jgene')
    dense_encoder1 = Dense(dense_nodes, activation='elu', name='encoder_dense_1')(merged_input)
    dense_encoder2 = Dense(dense_nodes, activation='elu', name='encoder_dense_2')(dense_encoder1)

    # Latent layers:
    z_mean = Dense(latent_dim, name='z_mean')(dense_encoder2)
    z_log_var = Dense(latent_dim, name='z_log_var')(dense_encoder2)

    encoder = Model([encoder_input_CDR3, encoder_input_Vgene, encoder_input_Jgene], [z_mean, z_log_var])

    # Decoding layers:
    z = Lambda(
        sampling, output_shape=(latent_dim, ), name='reparameterization_trick')  # This is the reparameterization trick
    dense_decoder1 = Dense(dense_nodes, activation='elu', name='decoder_dense_1')
    dense_decoder2 = Dense(dense_nodes, activation='elu', name='decoder_dense_2')

    decoder_out_CDR3 = Dense(np.array(input_shape[0]).prod(), activation='linear', name='flat_CDR_out')
    reshape_CDR3 = Reshape(input_shape[0], name='CDR_out')
    position_wise_softmax_CDR3 = Activation(activation='softmax', name='CDR_prob_out')
    decoder_out_Vgene = Dense(input_shape[1][0], activation='softmax', name='Vgene_prob_out')
    decoder_out_Jgene = Dense(input_shape[2][0], activation='softmax', name='Jgene_prob_out')

    decoder_output_CDR3 = position_wise_softmax_CDR3(
        reshape_CDR3(decoder_out_CDR3(dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))))
    decoder_output_Vgene = decoder_out_Vgene(dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))
    decoder_output_Jgene = decoder_out_Jgene(dense_decoder2(dense_decoder1(z([z_mean, z_log_var]))))

    # Define the decoding part separately:
    z_mean_generator = Input(shape=(latent_dim, ))
    decoder_generator_CDR3 = position_wise_softmax_CDR3(
        reshape_CDR3(decoder_out_CDR3(dense_decoder2(dense_decoder1(z_mean_generator)))))
    decoder_generator_Vgene = decoder_out_Vgene(dense_decoder2(dense_decoder1(z_mean_generator)))
    decoder_generator_Jgene = decoder_out_Jgene(dense_decoder2(dense_decoder1(z_mean_generator)))

    decoder = Model(z_mean_generator, [decoder_generator_CDR3, decoder_generator_Vgene, decoder_generator_Jgene])

    vae = Model([encoder_input_CDR3, encoder_input_Vgene, encoder_input_Jgene],
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

    def get_data(self, fname, data_chunk_size):
        """
        Get data in the correct format from fname, then trim so the data length
        is a multiple of data_chunk_size.
        """
        df = pd.read_csv(fname, usecols=['amino_acid', 'v_gene', 'j_gene'])
        assert len(df) >= data_chunk_size
        n_to_take = len(df) - len(df) % data_chunk_size
        return conversion.unpadded_tcrbs_to_onehot(df[:n_to_take], self.max_len)

    def serialize_params(self, fname):
        """
        Dump model parameters to a file.
        """
        with open(fname, 'w') as fp:
            json.dump(self.params, fp)

    def fit(self, df: pd.DataFrame, epochs: int, validation_split: float, patience: int, tensorboard_log_dir: str):
        """
        Fit the model with early stopping.
        """
        data = cols_of_df(df)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        early_stopping = EarlyStopping(monitor='loss', patience=patience)
        tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
        self.vae.fit(
            x=data,  # y=X for a VAE.
            y=data,
            epochs=epochs,
            batch_size=self.params['batch_size'],
            validation_split=validation_split,
            callbacks=[early_stopping, tensorboard])

    def evaluate(self, x_df):
        """
        Wrapping Model.evaluate for this setting.

        :param x_df: A onehot encoded dataframe representing input sequences.

        :return: loss
        """
        data = cols_of_df(x_df)
        return self.vae.evaluate(x=data, y=data, batch_size=self.params['batch_size'])

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
            log_q_z_given_x = np.sum(stats.norm.logpdf(z_sample[i], z_mean[i], z_sd[i]))
            # Importance weight: p(z)/q(z|x)
            log_imp_weight = log_p_z - log_q_z_given_x
            # p(x|z) p(z) / q(z|x)
            out_ps[i] = log_p_x_given_z + log_imp_weight


# ### CLI ###


@click.group()
def cli():
    pass


def _train_tcr(latent_dim, dense_nodes, train_csv, model_params_fname, best_weights_fname, diagnostics_fname):
    # TODO: less stupid
    MAX_LEN = 30
    epochs = 300
    patience = 10
    validation_split = 0.1
    validation_split_multiplier = 10
    sub_chunk_size = validation_split * validation_split_multiplier
    # If this fails then we may have problems with chunks of the data being the
    # wrong length.
    assert sub_chunk_size == float(int(sub_chunk_size))
    batch_size = 100
    min_data_size = validation_split_multiplier * batch_size
    input_shape = [(MAX_LEN, len(conversion.AA_LIST)), (len(conversion.TCRB_V_GENE_LIST), ),
                   (len(conversion.TCRB_J_GENE_LIST), )]

    v = TCRVAE(input_shape=input_shape, batch_size=batch_size, latent_dim=latent_dim, dense_nodes=dense_nodes)
    train_data = v.get_data(train_csv, min_data_size)
    tensorboard_log_dir = os.path.join(os.path.dirname(best_weights_fname), 'logs')
    v.fit(train_data, epochs, validation_split, patience, tensorboard_log_dir)
    v.serialize_params(model_params_fname)
    v.vae.save_weights(best_weights_fname, overwrite=True)

    # Test weights reloading.
    vp = TCRVAE.of_json_file(model_params_fname)
    vp.vae.load_weights(best_weights_fname)

    df = pd.DataFrame({'train': v.evaluate(train_data), 'vp_train': vp.evaluate(train_data)}, index=v.vae.metrics_names)
    df.to_csv(diagnostics_fname)
    return v


@cli.command()
@click.option('--latent-dim', default=40, show_default=True, help='Number of latent space dimensions.')
@click.option('--dense-nodes', default=125, show_default=True, help='Number of dense nodes.')
@click.argument('train_csv', type=click.File('r'))
@click.argument('model_params_fname', type=click.Path(writable=True))
@click.argument('best_weights_fname', type=click.Path(writable=True))
@click.argument('diagnostics_fname', type=click.Path(writable=True))
def train_tcr(latent_dim, dense_nodes, train_csv, model_params_fname, best_weights_fname, diagnostics_fname):
    """
    Train the model, print out a model assessment, saving the best weights
    to best_weights_fname and the input model params to model_params_fname.
    """
    _train_tcr(latent_dim, dense_nodes, train_csv, model_params_fname, best_weights_fname, diagnostics_fname)


@cli.command()
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('train_csv', type=click.File('r'))
@click.argument('test_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def loss(params_json, model_weights, train_csv, test_csv, out_csv):
    """
    Record the losses on the train vs. the hold out test set.
    """

    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    df = pd.DataFrame({
        'train': v.evaluate(v.get_data(train_csv, v.params['batch_size'])),
        'test': v.evaluate(v.get_data(test_csv, v.params['batch_size']))
    },
                      index=v.vae.metrics_names)
    df.to_csv(out_csv)


@cli.command()
@click.option('--limit-input-to', default=None, help='Only use the first <argument> input sequences.')
@click.option('--nsamples', default=500, show_default=True, help='Number of importance samples to use.')
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('test_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def importance(limit_input_to, nsamples, params_json, model_weights, test_csv, out_csv):
    """
    Estimate the log generation probability of the sequences in test_csv on the
    VAE determined by params_json and model_weights.

    Spit the results into out_csv, one estimate per line.
    """

    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    df_x = conversion.unpadded_tcrbs_to_onehot(
        pd.read_csv(test_csv, usecols=['amino_acid', 'v_gene', 'j_gene']), v.max_len)

    if limit_input_to is not None:
        df_x = df_x.iloc[:int(limit_input_to)]

    log_p_x = np.zeros((nsamples, len(df_x)))
    click.echo(f"Calculating p(x) for {test_csv.name} via importance sampling...")

    with click.progressbar(range(nsamples)) as bar:
        for i in bar:
            v.log_p_of_x_importance_sample(df_x, log_p_x[i])

    # Calculate log of mean of numbers given in log space.
    avg = special.logsumexp(log_p_x, axis=0) - np.log(nsamples)
    pd.DataFrame({'log_p_x': avg}).to_csv(out_csv, index=False)


if __name__ == '__main__':
    cli()
