"""
Using the thymic Q framework.
"""

import click

import numpy as np
import pandas as pd

lvj = ['length', 'v_gene', 'j_gene']


def tamp(x, m):
    """
    "Tamp down" the values in x with a "pseudo-maximum" m.
    """
    return m * np.arctan(x / m)


def normalize_column(df, col_name):
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


def merge_lvj_dfs(df1, df2):
    merged = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
    merged.fillna(0, inplace=True)
    return merged


def q_of_train_and_model_pgen(train_pgen_tsv, model_p_lvj_csv, tamp_m=1e5, pseudocount_multiplier=0.5):
    """
    See definition of tamp above to understand tamp_m.
    """
    # Merge the p_lvj from the data and that from the model:
    df = merge_lvj_dfs(
        p_lvj_of_Pgen_tsv(train_pgen_tsv, col_name='data_P_lvj'), pd.read_csv(model_p_lvj_csv, index_col=lvj))
    # We need to do this merge so we can add a pseudocount:
    add_pseudocount(df, 'model_P_lvj', pseudocount_multiplier)
    normalize_column(df, 'model_P_lvj')
    q = tamp(df['data_P_lvj'] / df['model_P_lvj'], tamp_m)
    return pd.DataFrame({'q': q})


def calc_Ppost(q_csv, data_pgen_tsv, pseudocount_multiplier=0.5):
    """
    Multiply Pgen by the corresponding q_lvj to get Ppost.
    """
    df = read_olga_pgen_tsv(data_pgen_tsv)
    df['length'] = df['amino_acid'].apply(len)
    df.set_index(lvj, inplace=True)
    df = merge_lvj_dfs(df, pd.read_csv(q_csv))
    add_pseudocount(df, 'q', pseudocount_multiplier)
    df['Ppost'] = df['Pgen'] * df['q']
    # Drop length
    df.reset_index(level=0, drop=True, inplace=True)
    df.reset_index(inplace=True)
    df.set_index(['amino_acid', 'v_gene', 'j_gene'], inplace=True)
    return df


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
@click.argument('train_pgen_tsv', type=click.File('r'))
@click.argument('model_p_lvj_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def q(train_pgen_tsv, model_p_lvj_csv, out_csv):
    """
    Calculate q_lvj given training data and p_lvj obtained from the model.
    """
    q_of_train_and_model_pgen(train_pgen_tsv, model_p_lvj_csv).to_csv(out_csv)


@cli.command()
@click.argument('q_csv', type=click.File('r'))
@click.argument('data_pgen_tsv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def ppost(q_csv, data_pgen_tsv, out_csv):
    """
    Calculate P_post given q_lvj and P_gen calculated for a data set.
    """
    calc_Ppost(q_csv, data_pgen_tsv).to_csv(out_csv)


if __name__ == '__main__':
    cli()
