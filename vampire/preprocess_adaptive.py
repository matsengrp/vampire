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

from vampire import gene_name_conversion as conversion

# Sometimes Adaptive uses one set of column names, and sometimes another.
HEADER_TRANSLATION_DICT = {
    'sequenceStatus': 'frame_type',
    'aminoAcid': 'amino_acid',
    'vGeneName': 'v_gene',
    'jGeneName': 'j_gene'
}


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
    return df[df['amino_acid'].str.contains('^C.*F$') | df['amino_acid'].str.contains('^C.*YV$')]


def filter_on_cdr3_length(df, max_len):
    """
    Only take sequences that have a CDR3 of at most `max_len` length.
    """
    return df[df['amino_acid'].apply(len) <= max_len]


def filter_on_TCRB(df):
    """
    Only take sequences that have a resolved TCRB gene for V and J.
    """
    return df[df['v_gene'].str.contains('^TCRB') & df['j_gene'].str.contains('^TCRB')]


def filter_on_olga(df):
    """
    Only take sequences with genes that are present in both the OLGA and the
    Adaptive gene sets.

    Also,
    * exclude TCRBJ2-5, which Adaptive annotates badly.
    * exclude TCRBJ2-7, which appears to be problematic for OLGA.
    """
    d = conversion.adaptive_to_olga_dict()
    del d['TRBJ']['TCRBJ02-05']
    del d['TRBJ']['TCRBJ02-07']
    return conversion.filter_by_gene_names(df, d)


def apply_all_filters(df, max_len=30, fail_fraction_remaining=None):
    """
    Apply all filters.

    Fail if less than `fail_fraction_remaining` of the sequences remain.
    """
    click.echo(f"Original data: {len(df)} rows")
    df = filter_and_drop_frame(df)
    original_count = len(df)
    click.echo(f"Restricting to in-frame: {len(df)} rows")
    df = filter_on_cdr3_bounding_aas(df)
    click.echo(f"Requiring sane CDR3 bounding AAs: {len(df)} rows")
    df = filter_on_cdr3_length(df, max_len)
    click.echo(f"Requiring CDR3 to be <= {max_len} amino acids: {len(df)} rows")
    df = filter_on_TCRB(df)
    click.echo(f"Requiring resolved TCRB genes: {len(df)} rows")
    df = filter_on_olga(df)
    click.echo(f"Requiring genes that are also present in the OLGA set: {len(df)} rows")
    if fail_fraction_remaining:
        if len(df) / original_count < fail_fraction_remaining:
            raise Exception(f"We started with {original_count} sequences and now we have {len(df)}. Failing.")
    return df.reset_index(drop=True)


def collect_vjcdr3_duplicates(df):
    """
    Define a vjcdr3 to be the concatenation of the V label,
    the J label, and the CDR3 protein sequence. Here we build
    a dictionary mapping vjcdr3 sequences to rows containing
    that vjcdr3 sequence.

    We only include sequences with a CDR3 amino acid sequence.
    """
    d = collections.defaultdict(list)

    for idx, row in df.iterrows():
        # nan means no CDR3 sequence. We don't want to include those.
        if row['amino_acid'] is not np.nan:
            key = '_'.join([row['v_gene'], row['j_gene'], row['amino_acid']])
            d[key].append(idx)

    return d


def dedup_on_vjcdr3(df):
    """
    Given a data frame of sequences, sample one
    representative per vjcdr3 uniformly.

    Note: not used in the current preprocessing step.
    """
    dup_dict = collect_vjcdr3_duplicates(df)
    c = collections.Counter([len(v) for (_, v) in dup_dict.items()])
    click.echo("A count of the frequency of vjcdr3 duplicates:")
    click.echo(c)
    indices = [random.choice(v) for (_, v) in dup_dict.items()]
    indices.sort()
    return df.loc[indices].reset_index(drop=True)


def read_adaptive_tsv(path):
    """
    Read an Adaptive TSV file and extract the columns we use, namely
    amino_acid, frame_type, v_gene, and j_gene.

    I have seen two flavors of the Adaptive header names, one of which uses
    snake_case and the other that uses camelCase.
    """

    test_bite = pd.read_csv(path, delimiter='\t', nrows=1)

    camel_columns = set(HEADER_TRANSLATION_DICT.keys())
    snake_columns = set(HEADER_TRANSLATION_DICT.values())

    if camel_columns.issubset(set(test_bite.columns)):
        take_columns = camel_columns
    elif snake_columns.issubset(set(test_bite.columns)):
        take_columns = snake_columns
    else:
        raise Exception("Unknown column names!")

    df = pd.read_csv(path, delimiter='\t', usecols=take_columns)
    df.rename(columns=HEADER_TRANSLATION_DICT, inplace=True)
    return df


@click.command()
# Below we use Path rather than File because we don't want to have to figure
# out whether a file is compressed or not-- Pandas will figure that out for us.
@click.option(
    '--fail-fraction-remaining',
    show_default=True,
    default=0.25,
    help="Fail if the post-filtration fraction is below this number.")
@click.option(
    '--sample',
    type=int,
    metavar='N',
    help="Sample N sequences without replacement from the preprocessed sequences for output.")
@click.argument('in_tsv', type=click.Path(exists=True))
@click.argument('out_csv', type=click.File('w'))
def preprocess_tsv(fail_fraction_remaining, sample, in_tsv, out_csv):
    """
    Preprocess the Adaptive TSV at IN_TSV and output to OUT_CSV.

    This includes doing filters as well as deduplicating on vjcdr3s.
    """
    df = apply_all_filters(read_adaptive_tsv(in_tsv), fail_fraction_remaining=fail_fraction_remaining)

    if sample:
        if len(df) < sample:
            raise Exception(
                f"We only have {len(df)} sequences so we can't sample {sample} of them without replacement. Failing.")
        df = df.sample(n=sample)
        click.echo(f"Sampling: {sample} rows")

    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    preprocess_tsv()
