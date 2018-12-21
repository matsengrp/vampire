import click
import pandas as pd

import vampire.common as common

translation_csv = 'adaptive-olga-translation.csv'


def adaptive_to_olga_dict():
    gb = common.read_data_csv(translation_csv).dropna().groupby('locus')
    return {locus: {row['adaptive']: row['olga'] for _, row in df.iterrows()} for locus, df in gb}


def olga_to_adaptive_dict():
    gb = common.read_data_csv(translation_csv).dropna().groupby('locus')
    return {locus: {row['olga']: row['adaptive'] for _, row in df.iterrows()} for locus, df in gb}


def filter_by_gene_names(df, conversion_dict):
    """
    Only allow through gene names that are present for both programs.
    """
    allowed = {locus: set(d.keys()) for locus, d in conversion_dict.items()}
    return df.loc[df['v_gene'].isin(allowed['TRBV']) & df['j_gene'].isin(allowed['TRBJ']), :]


def convert_gene_names(df, conversion_dict):
    converted = df.copy()
    converted['v_gene'] = df['v_gene'].map(conversion_dict['TRBV'].get)
    converted['j_gene'] = df['j_gene'].map(conversion_dict['TRBJ'].get)
    return converted


def convert_and_filter(df, conversion_dict):
    orig_len = len(df)
    converted = convert_gene_names(filter_by_gene_names(df, conversion_dict), conversion_dict)
    n_trimmed = orig_len - len(converted)
    if n_trimmed > 0:
        click.echo(f"Warning: couldn't convert {n_trimmed} sequences and trimmed them off.")
    return converted


# ### CLI ###


@click.group()
def cli():
    pass


@cli.command()
@click.argument('adaptive_csv', type=click.Path(exists=True))
@click.argument('olga_tsv', type=click.Path(writable=True))
def adaptive2olga(adaptive_csv, olga_tsv):
    df = pd.read_csv(adaptive_csv, usecols=['amino_acid', 'v_gene', 'j_gene'])
    convert_and_filter(df, adaptive_to_olga_dict()).to_csv(olga_tsv, sep='\t', index=False, header=False)


@cli.command()
@click.argument('olga_tsv', type=click.Path(exists=True))
@click.argument('adaptive_csv', type=click.Path(writable=True))
def olga2adaptive(olga_tsv, adaptive_csv):
    df = pd.read_csv(olga_tsv, sep='\t', header=None)
    if len(df.columns) == 4:
        df = df.iloc[:, 1:4]
    assert len(df.columns) == 3
    df.columns = ['amino_acid', 'v_gene', 'j_gene']
    convert_and_filter(df, olga_to_adaptive_dict()).to_csv(adaptive_csv, index=False)


if __name__ == '__main__':
    cli()
