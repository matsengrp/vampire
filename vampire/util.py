"""
Utilities, accessible via subcommands.
"""

import click
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


if __name__ == '__main__':
    cli()
