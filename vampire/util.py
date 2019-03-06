"""
Utilities, accessible via subcommands.
"""

import datetime
import itertools
import json
import os
import re
import shutil
import sys

import click
import numpy as np
import pandas as pd

import vampire.common as common

from vampire import preprocess_adaptive
from vampire.gene_name_conversion import convert_gene_names
from vampire.gene_name_conversion import olga_to_adaptive_dict

from sklearn.model_selection import train_test_split


@click.group()
def cli():
    pass


@cli.command()
@click.option('--idx', required=True, help='The row index for the summary output.')
@click.option('--idx-name', required=True, help='The row index name.')
@click.argument('seq_path', type=click.Path(exists=True))
@click.argument('pvae_path', type=click.Path(exists=True))
@click.argument('ppost_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path(writable=True))
def merge_ps(idx, idx_name, seq_path, pvae_path, ppost_path, out_path):
    """
    Merge probability estimates from Pvae and Ppost into a single data frame and write to an output CSV.

    SEQ_PATH should be a path to sequences in canonical CSV format, with
    sequences in the same order as PVAE_PATH.
    """

    def prep_index(df):
        df.set_index(['amino_acid', 'v_gene', 'j_gene'], inplace=True)
        df.sort_index(inplace=True)

    pvae_df = pd.read_csv(seq_path)
    pvae_df['log_Pvae'] = pd.read_csv(pvae_path)['log_p_x']
    prep_index(pvae_df)
    ppost_df = convert_gene_names(pd.read_csv(ppost_path), olga_to_adaptive_dict())
    prep_index(ppost_df)

    # If we don't drop duplicates then merge will expand the number of rows.
    # See https://stackoverflow.com/questions/39019591/duplicated-rows-when-merging-dataframes-in-python
    # We deduplicate Ppost, which is guaranteed to be identical among repeated elements.
    merged = pd.merge(pvae_df, ppost_df.drop_duplicates(), how='left', left_index=True, right_index=True)
    merged['log_Ppost'] = np.log(merged['Ppost'])
    merged.reset_index(inplace=True)
    merged[idx_name] = idx
    merged.set_index(idx_name, inplace=True)
    merged.to_csv(out_path)


@cli.command()
@click.option('--train-size', default=1000, help="Data count to use for train.")
@click.argument('in_csv', type=click.File('r'))
@click.argument('out1_csv', type=click.File('w'))
@click.argument('out2_csv', type=click.File('w'))
def split(train_size, in_csv, out1_csv, out2_csv):
    """
    Do a train/test split.
    """
    df = pd.read_csv(in_csv)
    (df1, df2) = train_test_split(df, train_size=train_size)
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
def fancystack(out, in_paths):
    """
    Like csvkit's csvstack, but fancy.
    Assumes the first column is a semicolon-separted thing to split into a
    multi-index. Also runs strip_dirpath_extn on paths.

    Note that this sorts the columns by name (part of merging columns).
    """

    def read_df(path):
        df = pd.read_csv(path)
        idx_names = df.columns[0].split(';')
        idx_str = df.iloc[0, 0]
        # Make sure the index column is constant.
        assert (df.iloc[:, 0] == idx_str).all()
        idx = idx_str.split(';')
        df.drop(df.columns[0], axis=1, inplace=True)

        for k, v in zip(idx_names, idx):
            if k in ['train_data', 'test_set']:
                v = common.strip_dirpath_extn(v)
            df[k] = v

        df.set_index(idx_names, inplace=True)

        return df

    pd.concat([read_df(path) for path in in_paths], sort=True).to_csv(out)


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
@click.option('--out-prefix', metavar='PRE', type=click.Path(writable=True), help="Output prefix.", required=True)
@click.option('--test-size', default=1 / 3, show_default=True, help="Proportion of sample to hold out for testing.")
@click.option('--test-regex', help="A regular expression that is used to identify the test set.")
@click.option(
    '--limit-each',
    metavar='N',
    type=int,
    help="Limit the contribution of each repertoire to the training set "
    "to a random N sequences. Will fail if any repertoire has less than N seqs.")
@click.argument('in_paths', nargs=-1)
def split_repertoires(out_prefix, test_size, test_regex, limit_each, in_paths):
    """
    Do a test-train split on the level of repertoires.

    By default does a random split, but if --test-regex is supplied it uses
    samples matching the regex as test.

    The --limit-each flag can be handy to ensure equal contribution of each
    repertoire to the training set.

    Writes out
    PRE.json, information about this train-test split, and
    PRE.train.tsv, a TSV with all of the sequences from the train set.
    """
    if test_regex:
        regex = re.compile(test_regex)
        test_paths = list(filter(regex.search, in_paths))
        train_paths = list(itertools.filterfalse(regex.search, in_paths))
    else:
        train_paths, test_paths = train_test_split(in_paths, test_size=test_size)

    train_tsv_path = out_prefix + '.train.tsv'

    info = {
        'split_call': ' '.join(sys.argv),
        'split_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        'train_paths': train_paths,
        'test_paths': test_paths,
        'train_tsv_path': train_tsv_path,
    }

    json_path = out_prefix + '.json'
    with open(json_path, 'w') as fp:
        fp.write(json.dumps(info, indent=4))

    header_written = False

    with open(train_tsv_path, 'w') as fp:
        for path in train_paths:
            df = preprocess_adaptive.read_adaptive_tsv(path)
            if limit_each:
                if limit_each > len(df):
                    raise ValueError(f"--limit-each parameter is greater than the number of sequences in {path}")
                df = df.sample(n=limit_each)
            if header_written:
                df.to_csv(fp, sep='\t', header=False, index=False)
            else:
                # This is our first file to write.
                header_written = True
                df.to_csv(fp, sep='\t', index=False)

    click.echo("Check JSON file with")
    click.echo(f"cat {json_path}")


@cli.command()
@click.option('--train-size', default=0.5, help="Data fraction to use for train.")
@click.argument('in_csv')
@click.argument('out_train_csv_bz2')
@click.argument('out_test_csv_bz2')
def split_rows(train_size, in_csv, out_train_csv_bz2, out_test_csv_bz2):
    df = pd.read_csv(in_csv, index_col=0)
    (df1, df2) = train_test_split(df, train_size=train_size)
    df1.to_csv(out_train_csv_bz2, compression='bz2')
    df2.to_csv(out_test_csv_bz2, compression='bz2')


def to_fake_csv(seq_list, path, include_freq=False):
    """
    Write a list of our favorite triples to a file.
    """
    with open(path, 'w') as fp:
        if include_freq:
            fp.write('amino_acid,v_gene,j_gene,count,frequency\n')
        else:
            fp.write('amino_acid,v_gene,j_gene\n')

        for line in seq_list:
            fp.write(line + '\n')


@cli.command()
@click.option(
    '--include-freq',
    is_flag=True,
    help="Include frequencies from 'count' and the counts themselves as columns in CSV.")
@click.option(
    '--n-to-sample', default=100, help="Number of sequences to sample.")
@click.option(
    '--min-count',
    default=4,
    show_default=True,
    help="Only include sequences that are found in at least this number of subjects."
)
@click.option(
    '--column',
    default='count',
    help="Counts column to use for sampling probabilities.")
@click.argument('in_csv')
@click.argument('out_csv')
def sample_data_set(include_freq, n_to_sample, min_count, column, in_csv,
                    out_csv):
    """
    Sample sequences according to the counts given in the specified column and
    then output in a CSV file.

    Note that reported frequencies in --include-freq are from the 'count'
    column by design, irrespective of the --column argument.
    """
    df = pd.read_csv(in_csv, index_col=0)

    # This is the total number of occurrences of each sequence in selected_m.
    def seq_freqs_of_colname(colname, apply_min_count=False):
        seq_counts = np.array(df[colname])
        if apply_min_count:
            seq_counts[seq_counts < min_count] = 0
        count_sum = sum(seq_counts)
        if count_sum == 0:
            raise ZeroDivisionError
        return seq_counts / count_sum

    sampled_seq_v = np.random.multinomial(n_to_sample, seq_freqs_of_colname(column, apply_min_count=True))

    if include_freq:
        df.reset_index(inplace=True)
        out_vect = df['index'] + ',' + df['count'].astype('str') + ',' + seq_freqs_of_colname('count').astype('str')
    else:
        out_vect = df.index

    # In order to get the correct count, we take those that appear once or
    # more, then those twice or more, etc, until we exceed the maximum entry.
    sampled_seqs = []
    for i in range(np.max(sampled_seq_v)):
        sampled_seqs += list(out_vect[sampled_seq_v > i])

    to_fake_csv(sampled_seqs, out_csv, include_freq)


if __name__ == '__main__':
    cli()
