"""
Utilities, accessible via subcommands.
"""

import datetime
import os
import re
import shutil
import sys

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
        loss_df = pd.read_csv(input_d['loss'], index_col=0).reset_index()
        # The following 3 lines combine the data source and the metric into a
        # single id like `train_j_gene_output_loss`.
        loss_df = pd.melt(loss_df, id_vars='index')
        loss_df['id'] = loss_df['variable'] + '_' + loss_df['index']
        loss_df.set_index('id', inplace=True)
        df = pd.DataFrame(dict(zip(loss_df.index, loss_df['value'].transpose())), index=index)
    else:
        df = pd.DataFrame(index=index)

    def slurp_cols(path, prefix='', suffix=''):
        """
        Given a one-row CSV with summaries, add them to df with an optional
        prefix and suffix.
        """
        to_slurp = pd.read_csv(path)
        assert len(to_slurp) == 1
        for col in to_slurp:
            df[prefix + col + suffix] = to_slurp.loc[0, col]

    def add_p_summary(path, name):
        """
        Add a summary of something like `validation_pvae` where `validation` is
        the prefix and `pvae` is the statistic.
        """
        prefix, statistic = name.split('_')
        if statistic == 'pvae':
            log_statistic = pd.read_csv(path)['log_p_x']
        elif statistic == 'ppost':
            log_statistic = np.log(pd.read_csv(path)['Ppost'])
        else:
            raise Exception(f"Unknown statistic '{statistic}'")

        df[prefix + '_median_log_p'] = np.median(log_statistic)
        df[prefix + '_mean_log_p'] = np.mean(log_statistic)

    for name, path in input_d.items():
        if name in [
                'training_pvae', 'validation_pvae', 'test_pvae', 'training_ppost', 'validation_ppost', 'test_ppost'
        ]:
            add_p_summary(path, name)
        elif re.search('sumrep_divergences', name):
            slurp_cols(path, prefix='sumdiv_')
        elif re.search('auc_', name):
            slurp_cols(path)

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
    multi-index. Also runs strip_dirpath_extn on anything named 'sample'.

    Note that this sorts the columns by name (part of merging columns).
    """

    def read_row(path):
        row = pd.read_csv(path)
        assert len(row) == 1
        idx_names = row.columns[0].split(';')
        idx = row.iloc[0, 0].split(';')
        row.drop(row.columns[0], axis=1, inplace=True)

        for k, v in zip(idx_names, idx):
            if k in ['sample', 'test_set']:
                v = common.strip_dirpath_extn(v)
            row[k] = v

        row.set_index(idx_names, inplace=True)

        return row

    pd.concat([read_row(path) for path in in_paths], sort=True).to_csv(out)


@cli.command()
@click.option('--dest-path', type=click.Path(writable=True), required=True)
@click.argument('loss_paths', nargs=-1)
def copy_best_weights(dest_path, loss_paths):
    """
    Find the best weights according to validation loss and copy them to the specified path.
    """
    loss_paths = sorted(loss_paths)
    losses = [pd.read_csv(path, index_col=0).loc['loss', 'validation'] for path in loss_paths]
    smallest_loss_path = loss_paths[np.argmin(losses)]
    best_weights = os.path.join(os.path.dirname(smallest_loss_path), 'best_weights.h5')
    shutil.copyfile(best_weights, dest_path)


@cli.command()
@click.argument('l_csv_path', type=click.File('r'))
@click.argument('r_csv_path', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def sharedwith(l_csv_path, r_csv_path, out_csv):
    """
    Write out a TCR dataframe identical to that in `l_csv_path` (perhaps with reordered columns)
    but with an extra column `present` indicating if that TCR is present in r_csv_path.
    """
    avj_columns = ['amino_acid', 'v_gene', 'j_gene']

    def read_df(path):
        df = pd.read_csv(path)
        df.set_index(avj_columns, inplace=True)
        return df

    l_df = read_df(l_csv_path)
    r_df = read_df(r_csv_path)

    intersection = pd.merge(l_df, r_df, left_index=True, right_index=True, how='inner')
    l_df['present'] = 0
    l_df.loc[intersection.index, 'present'] = 1
    click.echo(
        f"{sum(l_df['present'])} of {len(l_df)} sequences in {l_csv_path.name} are shared with {r_csv_path.name}")
    l_df.to_csv(out_csv)


@cli.command()
@click.option('--ncols', default=17, show_default=True, help="Only take the first this many columns.")
@click.option('--out-prefix', metavar='PRE', type=click.Path(writable=True), help="Output prefix.", required=True)
@click.option('--test-size', default=1 / 3, show_default=True, help="Proportion of sample to hold out for testing.")
@click.argument('in_paths', nargs=-1)
def split_repertoires(ncols, out_prefix, test_size, in_paths):
    """
    Do a test-train split on the level of repertoires. Writes out

    PRE.train.tsv: a TSV with all of the sequences from the test set,
    PRE.test.txt: a text file with the test paths (one per line),
    PRE.log: information about this train-test split.
    """
    train_paths, test_paths = train_test_split(in_paths, test_size=test_size)

    with open(out_prefix + '.log', 'w') as fp:
        fp.write(' '.join(sys.argv) + '\n')
        fp.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\n')
        fp.write("train paths: " + str(train_paths) + '\n')
        fp.write("test paths: " + str(test_paths) + '\n')

    with open(out_prefix + '.test.txt', 'w') as fp:
        for path in test_paths:
            fp.write(os.path.abspath(path) + '\n')

    columns = None

    with open(out_prefix + '.train.tsv', 'w') as fp:
        for path in train_paths:
            df = pd.read_csv(path, sep='\t', usecols=range(ncols))
            if columns:
                # This is not our first file to write.
                # Make sure that colnames match.
                if columns != list(df.columns):
                    raise Exception("Column doesn't match!")
                df.to_csv(fp, sep='\t', header=False, index=False)
            else:
                # This is our first file to write.
                columns = list(df.columns)
                df.to_csv(fp, sep='\t', index=False)


if __name__ == '__main__':
    cli()
