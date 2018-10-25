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
@click.argument('in_csv', type=click.File('r'))
@click.argument('out1_csv', type=click.File('w'))
@click.argument('out2_csv', type=click.File('w'))
def split(in_csv, out1_csv, out2_csv):
    """
    Do a 50/50 split.
    """
    df = pd.read_csv(in_csv)
    (df1, df2) = train_test_split(df, test_size=0.5)
    df1.to_csv(out1_csv, index=False)
    df2.to_csv(out2_csv, index=False)


if __name__ == '__main__':
    cli()
