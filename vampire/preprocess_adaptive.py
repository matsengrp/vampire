"""
Preprocess data coming from Adaptive immunoSEQ.

Reference:
https://clients.adaptivebiotech.com/assets/downloads/immunoSEQ_AnalyzerManual.pdf
"""

import click
import collections
import numpy as np
import pandas as pd
import random


def filter_and_drop_frame(df):
    """
    Select in-frame sequences and then drop that column.
    """
    return df.query('frame_type == "In"').drop('frame_type', axis=1)


def filter_on_cdr3_bounding_aas(df):
    """
    Only take sequences that have a C at the beginning and a F or a YV at the
    end of the `amino_acid` column.

    Note that according to the Adaptive docs the `amino_acid` column is indeed
    the CDR3 amino acid.
    """
    return df[df['amino_acid'].str.contains('^C.*F$')
              | df['amino_acid'].str.contains('^C.*YV$')]


def filter_on_TCRB(df):
    """
    Only take sequences that have a resolved TCRB gene for V and J.
    """
    return df[df['v_gene'].str.contains('^TCRB')
              & df['j_gene'].str.contains('^TCRB')]


def apply_all_filters(df):
    """
    Apply all filters.
    """
    click.echo(f"Original data: {len(df)} rows")
    df = filter_and_drop_frame(df)
    click.echo(f"Restricting to in-frame: {len(df)} rows")
    df = filter_on_cdr3_bounding_aas(df)
    click.echo(f"Requiring sane CDR3 bounding AAs: {len(df)} rows")
    df = filter_on_TCRB(df)
    click.echo(f"Requiring resolved TCRB genes: {len(df)} rows")
    return df.reset_index(drop=True)


def collect_protein_duplicates(df):
    """
    Build a dictionary mapping protein sequences to rows containing that
    protein sequence.
    """
    d = collections.defaultdict(list)

    for idx, row in df.iterrows():
        d[row['amino_acid']].append(idx)

    # nan means no protein sequence. We don't care about those.
    if np.nan in d:
        d.pop(np.nan)

    return d


def dedup_on_proteins(df):
    """
    Given a data frame of sequences, sample one representative per protein
    sequence uniformly.
    """
    dup_dict = collect_protein_duplicates(df)
    c = collections.Counter([len(v) for (_, v) in dup_dict.items()])
    click.echo("A count of the frequency of protein duplicates:")
    click.echo(c)
    indices = sum([random.sample(v, 1) for (_, v) in dup_dict.items()], [])
    indices.sort()
    return df.loc[indices].reset_index(drop=True)


def read_adaptive_tsv(f):
    """
    Read an Adaptive TSV file and extract the columns we use.
    """
    return pd.read_csv(f, delimiter='\t', usecols=[0, 1, 2, 10, 16])


@click.command()
@click.argument('in_tsv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def preprocess_tsv(in_tsv, out_csv):
    """
    Preprocess the Adaptive TSV at IN_TSV and output to OUT_CSV.

    This includes doing filters as well as deduplicating on proteins.
    """
    df = dedup_on_proteins(apply_all_filters(read_adaptive_tsv(in_tsv)))
    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    preprocess_tsv()
