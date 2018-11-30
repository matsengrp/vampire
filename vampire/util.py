"""
Utilities, accessible via subcommands.
"""

import re

import click
import common
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


@click.group()
def cli():
    pass


@cli.command()
@click.option('--test-size', default=0.5, help="Proportion of sample to hold out for testing.")
@click.argument('in_csv', type=click.File('r'))
@click.argument('out1_csv', type=click.File('w'))
@click.argument('out2_csv', type=click.File('w'))
def split(test_size, in_csv, out1_csv, out2_csv):
    """
    Do a train/test split.
    """
    df = pd.read_csv(in_csv)
    (df1, df2) = train_test_split(df, test_size=test_size)
    df1.to_csv(out1_csv, index=False)
    df2.to_csv(out2_csv, index=False)


@cli.command()
@click.option('--out', type=click.File('w'), help='Output file path.', required=True)
@click.option('--idx', required=True, help='The row index for the summary output.')
@click.option('--idx-name', required=True, help='The row index name.')
@click.option(
    '--colnames', default='', help='Comma-separated column identifier names corresponding to the files that follow.')
@click.argument('in_paths', nargs=-1)
def summarize(out, idx, idx_name, colnames, in_paths):
    """
    Summarize results of a run as a single-row CSV. The input is of flexible
    length: each input file is associated with an identifier specified using
    the --colnames flag.
    """
    colnames = colnames.split(',')
    if len(colnames) != len(in_paths):
        raise Exception("The number of colnames is not equal to the number of input files.")
    input_d = {k: v for k, v in zip(colnames, in_paths)}

    index = pd.Index([idx], name=idx_name)
    if 'loss' in input_d:
        loss_df = pd.read_csv(input_d['loss'], index_col=0)
        df = pd.DataFrame(dict(zip(loss_df.index, loss_df['test'].transpose())), index=index)
    else:
        df = pd.DataFrame(index=index)

    def slurp_cols(path, prefix='', suffix=''):
        """
        Given a one-row CSV with summaries, add them to df with an optional
        prefix and suffix.
        """
        to_slurp = pd.read_csv(path)
        assert(len(to_slurp) == 1)
        for col in to_slurp:
            df[prefix+col+suffix] = to_slurp.loc[0, col]

    for name, path in input_d.items():
        if name == 'test_pvae':
            log_pvae = pd.read_csv(path)['log_p_x']
            df['test_median_log_pvae'] = np.median(log_pvae)
            df['test_log_mean_pvae'] = np.log(np.mean(np.exp(log_pvae)))
            # Yes, Vladimir, we are taking a standard deviation of something
            # that isn't normal. They look kinda gamma-ish after applying log.
            df['test_log_pvae_sd'] = np.std(log_pvae)
        elif name == 'vae_generated_sumrep':
            slurp_cols(path, prefix='sumrep_')
        elif re.search('sumrep_divergences', name):
            slurp_cols(path, prefix='sumdiv_')

    df.to_csv(out)


@cli.command()
@click.option('--out', type=click.File('w'), help='Output file path.', required=True)
@click.argument('in_paths', nargs=-1)
def csvstack(out, in_paths):
    """
    Like csvkit's csvstack, but can deal with varying columns.
    See https://github.com/wireservice/csvkit/issues/245 for details.

    Note that this sorts the columns by name (part of merging columns).
    """
    pd.concat([pd.read_csv(path) for path in in_paths], sort=True).to_csv(out, index=False)


@cli.command()
@click.option('--out', type=click.File('w'), help='Output file path.', required=True)
@click.argument('in_paths', nargs=-1)
def stackrows(out, in_paths):
    """
    Like csvkit's csvstack, but fancy.
    Assumes the first column is a semicolon-separted thing to split into a
    multi-index. Also runs strip_dirpath_extn on anything named "sample".

    Note that this sorts the columns by name (part of merging columns).
    """

    def read_row(path):
        row = pd.read_csv(path)
        assert len(row) == 1
        idx_names = row.columns[0].split(';')
        idx = row.iloc[0, 0].split(';')
        row.drop(row.columns[0], axis=1, inplace=True)

        for k, v in zip(idx_names, idx):
            if k == 'sample':
                v = common.strip_dirpath_extn(v)
            row[k] = v

        row.set_index(idx_names, inplace=True)

        return row

    pd.concat([read_row(path) for path in in_paths], sort=True).to_csv(out)


if __name__ == '__main__':
    cli()
