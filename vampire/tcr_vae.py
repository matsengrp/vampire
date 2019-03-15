"""
This file contains the object and CLI for training our VAEs.

The models themselves are in the `models/` directory. Each of these Python
files should have a `build` function that returns a dictionary with
entries for: encoder, decoder, and vae.

We also require each model to define a corresponding `prepare_data` function
that prepares data for input into the vae, and a `interpret_output` function
that can convert whatever the VAE spits out back to our familiar triple of
amino_acid, v_gene, and j_gene.

This file is written in Python 3.5 so that we can run in the Tensorflow
Docker container.
"""

from collections import OrderedDict
import importlib
import json
import math
import os

import click
import numpy as np
import pandas as pd

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import scipy.special as special
import scipy.stats as stats

import vampire.common as common
import vampire.tcregex as tcregex
import vampire.xcr_vector_conversion as conversion


def logprob_of_obs_vect(probs, obs):
    """
    Calculate the log of probability of the observations.

    :param probs: a matrix with each row giving the probability of
        observations.
    :param obs: a matrix with each row one-hot-encoding an observation.

    Kristian implemented this as
        np.sum(np.log(np.matmul(probs, obs.T).diagonal()))
    which is equivalent but harder to follow.
    """
    # Here axis=1 means sum across columns (the sum will be empty except for
    # the single nonzero entry).
    return np.sum(np.log(np.sum(probs * obs, axis=1)))


class TCRVAE:
    def __init__(self, params):
        self.params = params
        model = importlib.import_module('vampire.models.' + params['model'])
        # Digest the dictionary returned by model.build into self attributes.
        for submodel_name, submodel in model.build(params).items():
            setattr(self, submodel_name, submodel)
        self.prepare_data = model.prepare_data
        self.interpret_output = model.interpret_output

    @classmethod
    def default_params(cls):
        """
        Return a dictionary with default parameters.

        The parameters below should be self explanatory except for:

        * beta is the weight put on the KL term of the VAE. See the models for
          how it gets incorporated.
        """
        return dict(
            # Models:
            model='basic',
            # Model parameters.
            latent_dim=20,
            dense_nodes=75,
            aa_embedding_dim=21,
            v_gene_embedding_dim=30,
            j_gene_embedding_dim=13,
            beta=0.75,
            # Input data parameters.
            max_cdr3_len=30,
            n_aas=len(conversion.AA_LIST),
            n_v_genes=len(conversion.TCRB_V_GENE_LIST),
            n_j_genes=len(conversion.TCRB_J_GENE_LIST),
            # Training parameters.
            stopping_monitor='val_loss',
            batch_size=100,
            pretrains=10,
            warmup_period=20,
            epochs=500,
            patience=20)

    @classmethod
    def default(cls):
        """
        Return a VAE with default parameters.
        """
        return cls(cls.default_params())

    @classmethod
    def of_json_file(cls, fname):
        """
        Build a TCRVAE from a parameter dictionary dumped to JSON.
        """
        with open(fname, 'r') as fp:
            return cls(json.load(fp))

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

    def serialize_params(self, fname):
        """
        Dump model parameters to a file.
        """
        with open(fname, 'w') as fp:
            json.dump(self.params, fp)

    def reinitialize_weights(self):
        session = K.get_session()
        for layer in self.vae.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def get_data(self, fname, data_chunk_size=0):
        """
        Get data in the correct format from fname. If data_chunk_size is
        nonzero, trim so the data length is a multiple of data_chunk_size.
        """
        df = pd.read_csv(fname, usecols=['amino_acid', 'v_gene', 'j_gene'])
        if data_chunk_size == 0:
            sub_df = df
        else:
            assert len(df) >= data_chunk_size
            n_to_take = len(df) - len(df) % data_chunk_size
            sub_df = df[:n_to_take]
        return conversion.unpadded_tcrbs_to_onehot(sub_df, self.params['max_cdr3_len'])

    def fit(self, x_df: pd.DataFrame, validation_split: float, best_weights_fname: str, tensorboard_log_dir: str):
        """
        Fit the vae with warmup and early stopping.
        """
        data = self.prepare_data(x_df)

        best_val_loss = np.inf

        # We pretrain a given number of times and take the best run for the full train.
        for pretrain_idx in range(self.params['pretrains']):
            self.reinitialize_weights()
            # In our first fitting phase we don't apply EarlyStopping so that
            # we get the number of specifed warmup epochs.
            # Below we apply the fact that right now the only thing in self.callbacks is the BetaSchedule callback.
            # If other callbacks appear we'll need to change this.
            if tensorboard_log_dir:
                callbacks = [keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir + '_warmup_' + str(pretrain_idx))]
            else:
                callbacks = []
            callbacks += self.callbacks  # <- here re callbacks
            history = self.vae.fit(
                x=data,  # y=X for a VAE.
                y=data,
                epochs=1 + self.params['warmup_period'],
                batch_size=self.params['batch_size'],
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=2)
            new_val_loss = history.history['val_loss'][-1]
            if new_val_loss < best_val_loss:
                best_val_loss = new_val_loss
                self.vae.save_weights(best_weights_fname, overwrite=True)

        self.vae.load_weights(best_weights_fname)

        checkpoint = ModelCheckpoint(best_weights_fname, save_best_only=True, mode='min')
        early_stopping = EarlyStopping(
            monitor=self.params['stopping_monitor'], patience=self.params['patience'], mode='min')
        callbacks = [checkpoint, early_stopping]
        if tensorboard_log_dir:
            callbacks += [keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)]
        self.vae.fit(
            x=data,  # y=X for a VAE.
            y=data,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=2)

    def evaluate(self, x_df, per_sequence=False):
        """
        By default, wrapping Model.evaluate for this setting.
        Also providing an option for per-sequence loss evaluation.

        :param x_df: A onehot encoded dataframe representing input sequences.
        :param per_sequence: A flag determining if we should return a
            collection of evaluation outputs, one per input. (This is not
            especially efficiently done.)

        :return: loss as a list, or a list of lists if per_sequence=True.
        """

        def our_evaluate(data):
            return self.vae.evaluate(x=data, y=data, batch_size=self.params['batch_size'], verbose=0)

        data = self.prepare_data(x_df)

        if not per_sequence:
            return our_evaluate(data)

        # data is a list of data elements. Here we repeat_row each of these
        # data elements the minimum amount for evaluate to work (the
        # batch_size) and evaluate.
        return [
            our_evaluate([common.repeat_row(data_elt, i, self.params['batch_size']) for data_elt in data])
            for i in range(len(x_df))
        ]

    def encode(self, x_df):
        """
        Get the VAE encoding of a given collection of sequences x.

        :param x_df: A onehot encoded dataframe representing input sequences.

        :return: z_mean and z_sd, the embedding mean and standard deviation.
        """
        z_mean, z_log_var = self.encoder.predict(self.prepare_data(x_df))
        z_sd = np.sqrt(np.exp(z_log_var))
        return z_mean, z_sd

    def decode(self, z):
        """
        Get the decoding of z.
        """
        return self.interpret_output(self.decoder.predict(z))

    def generate(self, n_seqs):
        """
        Generate a data frame of n_seqs sequences.
        """
        batch_size = self.params['batch_size']
        # Increase the number of desired sequences as needed so it's divisible by batch_size.
        n_actual = batch_size * math.ceil(n_seqs / batch_size)
        # Sample from the latent space to generate sequences:
        z_sample = np.random.normal(0, 1, size=(n_actual, self.params['latent_dim']))
        amino_acid_arr, v_gene_arr, j_gene_arr = self.decode(z_sample)
        # Convert back, restricting to the desired number of sequences.
        return conversion.onehot_to_tcrbs(amino_acid_arr[:n_seqs], v_gene_arr[:n_seqs], j_gene_arr[:n_seqs])

    def log_pvae_importance_sample(self, x_df, out_ps):
        """
        One importance sample to calculate the probability of generating some
        observed x's by decoding from the prior on z.

        Say we just have one x. We want p(x), which in principle we could
        calculate as the expectation of p(x|z) where z is drawn from p(z). That
        would be very inefficient given the size of the latent space. Instead,
        we use importance sampling, calculating the expectation of

        p(x|z) (p(z) / q(z|x))

        over z drawn from q(z|x). The ratio in parentheses is the importance
        weight.

        We emphasize that this is _one_ importance sample. Run this lots and
        take the average to get a good estimate.

        The VAE is allowed to have features besides the input of CDR3 amino
        acids and V/J genes. We have to have those be deterministically
        computed from the input, otherwise the methods below won't work.

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

        # Get encoding of x's in the latent space.
        z_mean, z_sd = self.encode(x_df)
        # Get samples from q(z|x) in the latent space, one for each input x.
        z_sample = stats.norm.rvs(z_mean, z_sd)
        # These are decoded samples from z. They are, thus, probability vectors
        # that get sampled if we want to realize actual sequences.
        aa_probs, v_gene_probs, j_gene_probs = self.decode(z_sample)

        # Onehot-encoded observations.
        # We use interpret_output to cut down to what we care about.
        aa_obs, v_gene_obs, j_gene_obs = self.interpret_output(self.prepare_data(x_df))

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


@cli.command()
@click.option('--tensorboard', is_flag=True, help="Record logs for TensorBoard.")
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('train_csv', type=click.File('r'))
@click.argument('best_weights_fname', type=click.Path(writable=True))
@click.argument('diagnostics_fname', type=click.Path(writable=True))
def train(tensorboard, params_json, train_csv, best_weights_fname, diagnostics_fname):
    """
    Train the model described in params_json using data in train_csv, saving
    the best weights to best_weights_fname and some diagnostics to
    diagnostics_fname.
    """
    v = TCRVAE.of_json_file(params_json)
    # Leaving this hardcoded for now.
    validation_split = 0.1
    validation_split_multiplier = 10
    sub_chunk_size = validation_split * validation_split_multiplier
    # If this fails then we may have problems with chunks of the data being the
    # wrong length.
    assert sub_chunk_size == float(int(sub_chunk_size))
    min_data_size = validation_split_multiplier * v.params['batch_size']

    train_data = v.get_data(train_csv, min_data_size)
    if tensorboard:
        tensorboard_log_dir = os.path.join(os.path.dirname(best_weights_fname), 'logs')
    else:
        tensorboard_log_dir = None
    v.fit(train_data, validation_split, best_weights_fname, tensorboard_log_dir)
    v.vae.save_weights(best_weights_fname, overwrite=True)

    # Test weights reloading.
    vp = TCRVAE.of_json_file(params_json)
    vp.vae.load_weights(best_weights_fname)

    df = pd.DataFrame(
        OrderedDict([('train', v.evaluate(train_data)), ('vp_train', vp.evaluate(train_data))]),
        index=v.vae.metrics_names)
    df.to_csv(diagnostics_fname)
    return v


@cli.command()
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('train_csv', type=click.File('r'))
@click.argument('validation_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def loss(params_json, model_weights, train_csv, validation_csv, out_csv):
    """
    Record aggregate losses.
    """

    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    df = pd.DataFrame(
        OrderedDict([('train', v.evaluate(v.get_data(train_csv, v.params['batch_size']))),
                     ('validation', v.evaluate(v.get_data(validation_csv, v.params['batch_size'])))]),
        index=v.vae.metrics_names)
    df.to_csv(out_csv)


@cli.command()
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('in_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def per_seq_loss(params_json, model_weights, in_csv, out_csv):
    """
    Record per-sequence losses.
    """

    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    df = pd.DataFrame(
        np.array(v.evaluate(v.get_data(in_csv, v.params['batch_size']), per_sequence=True)),
        columns=v.vae.metrics_names)
    df.to_csv(out_csv, index=False)


@cli.command()
@click.option('--limit-input-to', default=None, type=int, help='Only use the first <argument> input sequences.')
@click.option('--nsamples', default=500, show_default=True, help='Number of importance samples to use.')
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('test_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def pvae(limit_input_to, nsamples, params_json, model_weights, test_csv, out_csv):
    """
    Estimate Pvae of the sequences in test_csv for the VAE determined by
    params_json and model_weights.

    Output the results into out_csv, one estimate per line.
    """

    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    df_x = v.get_data(test_csv)

    if limit_input_to is not None:
        df_x = df_x.iloc[:int(limit_input_to)]

    log_p_x = np.zeros((nsamples, len(df_x)))
    click.echo("Calculating pvae for {} via importance sampling...".format(test_csv.name))

    with click.progressbar(range(nsamples)) as bar:
        for i in bar:
            v.log_pvae_importance_sample(df_x, log_p_x[i])

    # Calculate log of mean of numbers given in log space.
    avg = special.logsumexp(log_p_x, axis=0) - np.log(nsamples)
    pd.DataFrame({'log_p_x': avg}).to_csv(out_csv, index=False)


@cli.command()
@click.option('--nsamples', default=100, show_default=True, help="Number of importance samples to use.")
@click.option('--batch-size', default=100, show_default=True, help="Batch size for tcregex calculation.")
@click.option('--max-iters', default=100, show_default=True, help="The maximum number of batch iterations to use.")
@click.option(
    '--track-last', default=5, show_default=True, help="We want the SD of the last track-last to be less than tol.")
@click.option('--tol', default=0.005, show_default=True, help="Tolerance for tcregex accuracy.")
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('in_tcregex')
@click.argument('out_csv', type=click.File('w'))
def tcregex_pvae(nsamples, batch_size, max_iters, track_last, tol, params_json, model_weights, in_tcregex, out_csv):
    """
    Calculate Pvae for a TCR specified by a tcregex.

    A tcregex is specified as a string triple "v_gene,j_gene,cdr3_tcregex" where
    cdr3_tcregex uses regex symbols appropriate for amino acids.

    We keep on sampling sequences from the tcregex until the P_VAE converges.

    Note that the default number of importance samples is less than that for
    the usual pvae, because we're averaging out stochasticity anyhow.
    """
    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)

    # Accumulates the sequences and their P_VAEs across iters.
    generated_dfs = []
    # Accumulates the P_VAE means across iters.
    means = []

    for batch_i in range(max_iters):
        df_generated = tcregex.sample_tcregex(in_tcregex, batch_size)
        df_x = conversion.unpadded_tcrbs_to_onehot(df_generated, v.params['max_cdr3_len'])

        log_p_x = np.zeros((nsamples, len(df_x)))

        for i in range(nsamples):
            v.log_pvae_importance_sample(df_x, log_p_x[i])

        # Calculate log of mean of numbers given in log space.
        # This calculates the per-sequence log_p_x estimate.
        df_generated['log_p_x'] = special.logsumexp(log_p_x, axis=0) - np.log(nsamples)
        generated_dfs.append(df_generated)
        catted = pd.concat(generated_dfs)
        means.append(special.logsumexp(catted['log_p_x'], axis=0) - np.log(len(catted)))
        if len(means) > track_last:
            recent_sd = np.std(np.array(means[-track_last:]))
            click.echo("[Iter {}]\tmean: {:.6}\trecent SD: {:.5}\ttol: {}".format(batch_i, means[-1], recent_sd, tol))
            if recent_sd < tol:
                break
        else:
            click.echo("[Iter {}]\tmean: {:.6}".format(batch_i, means[-1]))

    click.echo("tcregex P_VAE estimate: {}".format(means[-1]))
    catted.to_csv(out_csv, index=False)


@cli.command()
@click.option('-n', '--nseqs', default=100, show_default=True, help='Number of sequences to generate.')
@click.argument('params_json', type=click.Path(exists=True))
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('out_csv', type=click.File('w'))
def generate(nseqs, params_json, model_weights, out_csv):
    """
    Generate some sequences and write them to a file.
    """
    v = TCRVAE.of_json_file(params_json)
    v.vae.load_weights(model_weights)
    v.generate(nseqs).to_csv(out_csv, index=False)


if __name__ == '__main__':
    cli()
