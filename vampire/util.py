"""
Utilities, accessible via subcommands.
"""

import click
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
@click.option('--name', default='', help='The row entry for the summary output.')
@click.option('--prefix', default='', help='A string to prepend to loss column headers.')
@click.option('--pvae', type=click.File('r'), help='Path to a file with Pvae values.')
@click.option('--generated-pgen', type=click.File('r'), help='Path to a file with Pgen values for generated sequences.')
@click.argument('loss_csv', type=click.File('r'))
@click.argument('out_csv', type=click.File('w'))
def summarize(name, prefix, pvae, generated_pgen, loss_csv, out_csv):
    """
    Summarize results of a run as a single-row CSV.
    """
    loss_df = pd.read_csv(loss_csv, index_col=0)
    index = pd.Index([name], name='name')
    df = pd.DataFrame(dict(zip([prefix+i for i in loss_df.index], loss_df['test'].transpose())), index=index)
    if pvae:
        df['test_median_pvae'] = np.median(pd.read_csv(pvae)['log_p_x'])
    if generated_pgen:
        generated_pgen_df = pd.read_csv(generated_pgen, header=None)
        df['generated_median_pgen'] = np.median(np.log(generated_pgen_df[1]))
    df.to_csv(out_csv)


if __name__ == '__main__':
    cli()
