"""
Computing in the thymic Q framework.
"""

import os
import tempfile

import click
import delegator
import numpy as np
import pandas as pd

# These are our "lvj" columns, which are the most common indices for
# probabilities and sequences in what follows.
lvj = ['length', 'v_gene', 'j_gene']


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
    """
    Read in a TSV in the format liked by OLGA. Four columns means that there is
    DNA data-- if so, drop it.
    """
    df = pd.read_csv(path, sep='\t', header=None)
    if len(df.columns) == 4:
        df = df.iloc[:, 1:]
    assert len(df.columns) == 3
    df.columns = 'amino_acid v_gene j_gene'.split()
    return df


def read_olga_pgen_tsv(path):
    """
    Read in a TSV output by OLGA's Pgen calculation.
    """
    df = pd.read_csv(path, sep='\t', header=None)
    assert len(df.columns) == 4
    df.columns = 'amino_acid v_gene j_gene Pgen'.split()
    return df


def lvj_frequency_of_olga_tsv(path, col_name):
    """
    Read in an OLGA TSV and calculate the frequency of the lvj triples
    contained in it.
    """
    df = read_olga_tsv(path)
    df['length'] = df['amino_acid'].apply(len)
    df = df.loc[:, lvj]
    df[col_name] = 1.
    df = df.groupby(lvj).sum()
    normalize_column(df, col_name)
    return df


def set_lvj_index(df):
    """
    Make an lvj index in place from the length, v_gene, and j_gene.
    """
    df['length'] = df['amino_acid'].apply(len)
    df.set_index(lvj, inplace=True)


def merge_lvj_dfs(df1, df2, how='outer'):
    """
    Merge two data frames on lvj indices.

    By default, uses the union of the keys (an "outer" join).
    """
    merged = pd.merge(df1, df2, how=how, left_index=True, right_index=True)
    merged.fillna(0, inplace=True)
    return merged


def q_of_train_and_model_pgen(model_p_lvj_csv, train_tsv, max_q=None, pseudocount_multiplier=0.5):
    """
    Fit a q distribution, but truncating q at max_q.
    """
    # Merge the p_lvj from the data and that from the model:
    df = merge_lvj_dfs(
        lvj_frequency_of_olga_tsv(train_tsv, col_name='data_P_lvj'), pd.read_csv(model_p_lvj_csv, index_col=lvj))
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
    # Add a pseudocount to Pgen and q to prevent infs in whole-data log likelihoods.
    add_pseudocount(df, 'q', pseudocount_multiplier)
    add_pseudocount(df, 'Pgen', pseudocount_multiplier)
    df['Ppost'] = df['Pgen'] * df['q']
    # Drop length from the index.
    df.reset_index(level=0, drop=True, inplace=True)
    # Turn the index columns into normal columns.
    df.reset_index(inplace=True)
    # Only keep our usual 3 columns.
    df.set_index(['amino_acid', 'v_gene', 'j_gene'], inplace=True)
    return df


def rejection_sample_Ppost(q_df, df_Pgen_sample):
    """
    Given a df_Pgen_sample that's sampled from Pgen, and a df of q's, perform
    rejection to obtain a sample from Ppost.

    Specifically, q is Ppost/Pgen, and so if we are proposing from Pgen, the
    maximum of q (i.e. q_max) is the bound on the likelihood ratio. Then each
    proposal x is accepted with probability (Ppost(x)/Pgen(x))/max_q, i.e.
    q(x)/max_q.
    """
    df = df_Pgen_sample
    set_lvj_index(df)
    # This merge is how we get the q value corresponding to each sequence x.
    df = merge_lvj_dfs(df, q_df, how='left')
    max_q = np.max(df['q'])
    df['acceptance_prob'] = df['q'] / max_q
    df['random'] = np.random.uniform(size=len(df))
    # We subselect the rows that should be accepted, and keep only the
    # amino_acid column (the v_gene and j_gene columns are kept as part of the
    # index).
    df = df.loc[df['random'] < df['acceptance_prob'], 'amino_acid']
    # Turn the index columns into normal columns, just keeping our usual 3
    # columns.
    df = df.reset_index()[['amino_acid', 'v_gene', 'j_gene']]
    # Randomize the order of the sequences (they got sorted in the process of
    # merging with Q).
    df = df.sample(frac=1)
    return df


def sample_Ppost(sample_size, q_csv, max_iter=100, proposal_size=1e6):
    """
    Repeatedly sample from Pgen to calculate Ppost using rejection sampling.
    """
    q_df = pd.read_csv(q_csv, index_col=lvj)
    out_df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        for iter_idx in range(max_iter):
            sample_path = os.path.join(tmpdir, str(iter_idx))
            c = delegator.run(' '.join(['olga-generate.sh', str(proposal_size), sample_path]))
            if c.return_code != 0:
                raise Exception("olga-generate.sh failed!")
            df_sample = rejection_sample_Ppost(q_df, read_olga_tsv(sample_path))
            out_df = out_df.append(df_sample)
            print(f"Sampled {len(out_df)} of {sample_size}")
            if len(out_df) >= sample_size:
                break

    if len(out_df) < sample_size:
        raise Exception("Did not obtain desired number of samples in the specified maximumn number of iterations.")

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
    Calculate frequency of lvj combinations in a given OLGA TSV.
    """
    lvj_frequency_of_olga_tsv(in_tsv, col_name).to_csv(out_csv)


@cli.command()
@click.option('--max-q', type=int, default=None, show_default=True, help="Limit q to be at most this value.")
# Here we use a click.Path so we can pass a .bz2'd CSV.
@click.argument('model_p_lvj_csv', type=click.Path(exists=True))
@click.argument('train_tsv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def q(max_q, model_p_lvj_csv, train_tsv, out_csv):
    """
    Calculate q_lvj given training data and p_lvj obtained from the model.
    """
    q_of_train_and_model_pgen(model_p_lvj_csv, train_tsv, max_q=max_q).to_csv(out_csv)


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
@click.option(
    '--max-iter',
    default=100,
    show_default=True,
    help="Maximum number of iterations to try to achieve the desired number of samples.")
@click.option(
    '--proposal-size', default=1000000, show_default=True, help="Number of samples to take for proposal distribution.")
@click.argument('sample_size', type=int)
@click.argument('q_csv', type=click.File('r'))
@click.argument('out_tsv', type=click.File('w'))
def sample(max_iter, proposal_size, sample_size, q_csv, out_tsv):
    """
    Sample from the Ppost distribution via rejection sampling.
    """
    sample_Ppost(
        sample_size, q_csv, max_iter=max_iter, proposal_size=proposal_size).to_csv(
            out_tsv, sep='\t', index=False)


if __name__ == '__main__':
    cli()
