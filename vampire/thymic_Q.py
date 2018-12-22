"""
Computing in the thymic Q framework.
"""

import click
import delegator
import numpy as np
import os
import pandas as pd
import tempfile

# These are our "lvj" columns, which are the most common indices for
# probabilities and sequences in what follows.
lvj = ['length', 'v_gene', 'j_gene']


def tamp(x, m):
    """
    "Tamp down" the values in x with a "pseudo-maximum" m.
    """
    return m * np.arctan(x / m)


def bound(x, m):
    """
    Cut off the values in x with a hard maximum m.
    """
    return np.minimum(x, m)


def normalize_column(df, col_name):
    """
    Normalize a column so that it has sum one.
    """
    df[col_name] = df[col_name] / sum(df[col_name])


def add_pseudocount(df, col_name, pseudocount_multiplier):
    """
    Add a pseudocount to `col_name`.

    The pseudocount is pseudocount_multiplier times the smallest non-zero element.
    """
    zero_rows = (df[col_name] == 0)
    pseudocount = min(df.loc[~zero_rows, col_name]) * pseudocount_multiplier
    df.loc[zero_rows, col_name] += pseudocount


def read_olga_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None)
    if len(df.columns) == 4:
        df = df.iloc[:, 1:]
    assert len(df.columns) == 3
    df.columns = 'amino_acid v_gene j_gene'.split()
    return df


def read_olga_pgen_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None)
    assert len(df.columns) == 4
    df.columns = 'amino_acid v_gene j_gene Pgen'.split()
    return df


def lvj_frequency_of_olga_tsv(path, col_name):
    df = read_olga_tsv(path)
    df['length'] = df['amino_acid'].apply(len)
    df = df.loc[:, lvj]
    df[col_name] = 1.
    df = df.groupby(lvj).sum()
    normalize_column(df, col_name)
    return df


def p_lvj_of_Pgen_tsv(path, col_name='Pgen'):
    """
    Read a Pgen TSV and get a p_lvj by summing over sequences with the same lvj triple.

    col_name: the name given to the Pgen column.
    """
    df = read_olga_pgen_tsv(path)
    df['length'] = df['amino_acid'].apply(len)
    idx = 'length,v_gene,j_gene,Pgen'.split(',')
    df = df.loc[:, idx]
    df = df.groupby(idx[:-1]).sum()
    df.rename(columns={'Pgen': col_name}, inplace=True)
    normalize_column(df, col_name)
    return df


def set_lvj_index(df):
    df['length'] = df['amino_acid'].apply(len)
    df.set_index(lvj, inplace=True)


def merge_lvj_dfs(df1, df2, how='outer'):
    """
    Merge on the lvj columns.

    By default, uses the union of the keys (an "outer" join).
    """
    merged = pd.merge(df1, df2, how=how, left_index=True, right_index=True)
    merged.fillna(0, inplace=True)
    return merged


def q_of_train_and_model_pgen(train_pgen_tsv, model_p_lvj_csv, max_q=None, pseudocount_multiplier=0.5):
    """
    Fit a q distribution, but truncating q at max_q.
    """
    # Merge the p_lvj from the data and that from the model:
    df = merge_lvj_dfs(
        p_lvj_of_Pgen_tsv(train_pgen_tsv, col_name='data_P_lvj'), pd.read_csv(model_p_lvj_csv, index_col=lvj))
    # We need to do this merge so we can add a pseudocount:
    add_pseudocount(df, 'model_P_lvj', pseudocount_multiplier)
    normalize_column(df, 'model_P_lvj')
    q = df['data_P_lvj'] / df['model_P_lvj']
    if max_q:
        q = bound(q, max_q)
    return pd.DataFrame({'q': q})


def calc_Ppost(q_csv, data_pgen_tsv, pseudocount_multiplier=0.5):
    """
    Multiply Pgen by the corresponding q_lvj to get Ppost.
    """
    df = read_olga_pgen_tsv(data_pgen_tsv)
    set_lvj_index(df)
    # Merging on left means that we only use only keys from the left data
    # frame, i.e. the sequences for which we are interested in computing Ppost.
    df = merge_lvj_dfs(df, pd.read_csv(q_csv, index_col=lvj), how='left')
    add_pseudocount(df, 'q', pseudocount_multiplier)
    df['Ppost'] = df['Pgen'] * df['q']
    # Drop length
    df.reset_index(level=0, drop=True, inplace=True)
    df.reset_index(inplace=True)
    df.set_index(['amino_acid', 'v_gene', 'j_gene'], inplace=True)
    return df


def rejection_sample_Ppost(q_df, df_Pgen_sample, max_q):
    """
    Given a df_Pgen_sample that's sampled from Pgen, and a df of q's, perform
    rejection to obtain a sample from Ppost.

    max_q is the maximum theoretical q.
    """
    df = df_Pgen_sample
    set_lvj_index(df)
    df = merge_lvj_dfs(df, q_df, how='left')
    df['cutoff'] = df['q'] / 100
    df['random'] = np.random.uniform(size=len(df))
    df = df.loc[df['random'] < df['cutoff'], 'amino_acid']
    df = df.reset_index()[['amino_acid', 'v_gene', 'j_gene']]
    # Randomize the order of the sequences (they got sorted in the process of merging with Q.
    df = df.sample(frac=1)
    return df

    # c = delegator.run('/home/matsen/Downloads/miniconda3/envs/py36/bin/olga-generate.sh 100 ~/x.csv')


def sample_Ppost(q_csv, sample_size, max_q, max_iter=100, proposal_size=1e6):
    q_df = pd.read_csv(q_csv, index_col=lvj)
    out_df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        for iter_idx in range(max_iter):
            sample_path = os.path.join(tmpdir, str(iter_idx))
            c = delegator.run(' '.join(['olga-generate.sh', str(proposal_size), sample_path]))
            if c.return_code != 0:
                raise Exception("olga-generate.sh failed!")
            df_sample = rejection_sample_Ppost(q_df, read_olga_tsv(sample_path), max_q)
            print(f"Sampled {len(df_sample)} sequences.")
            out_df = out_df.append(df_sample)
            if len(out_df) >= sample_size:
                break

    if len(out_df) < sample_size:
        raise Exception("Did not obtain desired number of samples in the specified number of iterations.")

    out_df = out_df.head(sample_size)
    return out_df


@click.group()
def cli():
    pass


@cli.command()
@click.option('--col-name', required=True, help="Name for frequency column.")
@click.argument('in_tsv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def lvj_frequency(col_name, in_tsv, out_csv):
    """
    Calculate frequency of LVJ combinations in a given OLGA TSV.
    """
    lvj_frequency_of_olga_tsv(in_tsv, col_name).to_csv(out_csv)


@cli.command()
@click.option('--max-q', type=int, default=None, show_default=True, help="Limit q to be at most this value.")
@click.argument('train_pgen_tsv', type=click.File('r'))
# Here we use a click.Path so we can pass a .bz2'd CSV.
@click.argument('model_p_lvj_csv', type=click.Path(exists=True))
@click.argument('out_csv', type=click.File('w'))
def q(max_q, train_pgen_tsv, model_p_lvj_csv, out_csv):
    """
    Calculate q_lvj given training data and p_lvj obtained from the model.
    """
    q_of_train_and_model_pgen(train_pgen_tsv, model_p_lvj_csv, max_q=max_q).to_csv(out_csv)


@cli.command()
@click.argument('q_csv', type=click.File('r'))
@click.argument('data_pgen_tsv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def ppost(q_csv, data_pgen_tsv, out_csv):
    """
    Calculate Ppost given q_lvj and Pgen calculated for a data set.
    """
    calc_Ppost(q_csv, data_pgen_tsv).to_csv(out_csv)


@cli.command()
@click.option('--max-q', default=100, show_default=True, help="Limit q to be at most this value.")
@click.option(
    '--max-iter',
    default=100,
    show_default=True,
    help="Maximum number of iterations to try to achieve the desired number of samples.")
@click.option(
    '--proposal-size', default=1000000, show_default=True, help="Number of samples to take for proposal distribution.")
@click.argument('q_csv', type=click.File('r'))
@click.argument('sample_size', type=int)
@click.argument('out_csv', type=click.File('w'))
def sample(max_q, max_iter, proposal_size, q_csv, sample_size, out_csv):
    """
    Sample from the Ppost distribution via rejection sampling.
    """
    sample_Ppost(q_csv, sample_size, max_q, max_iter=max_iter, proposal_size=proposal_size).to_csv(out_csv)


if __name__ == '__main__':
    cli()
