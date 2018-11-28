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
@click.option('--out', type=click.File('w'), help='Output file path.', required=True)
@click.option('--name', default='', help='The row entry for the summary output.')
@click.option(
    '--ids', default='', help='Comma-separated column identifier names corresponding to the files that follow.')
@click.argument('in_paths', nargs=-1)
def summarize(out, name, ids, in_paths):
    """
    Summarize results of a run as a single-row CSV. The input is of flexible
    length: each input file is associated with an identifier specified using
    the --ids flag.
    """
    headers = ids.split(',')
    if len(headers) != len(in_paths):
        raise Exception("The number of headers is not equal to the number of input files.")
    input_d = {k: v for k, v in zip(headers, in_paths)}

    index = pd.Index([name], name='name')
    if 'loss' in input_d:
        loss_df = pd.read_csv(input_d['loss'], index_col=0)
        df = pd.DataFrame(dict(zip(loss_df.index, loss_df['test'].transpose())), index=index)
    else:
        df = pd.DataFrame(index=index)

    for k, path in input_d.items():
        if k == 'test_pvae':
            log_pvae = pd.read_csv(path)['log_p_x']
            df['test_log_pvae_median'] = np.median(log_pvae)
            # Yes, Vladimir, we are taking a standard deviation of something
            # that isn't normal. They look kinda gamma-ish after applying log.
            df['test_log_pvae_sd'] = np.std(log_pvae)

    if name == '':
        df.to_csv(out, index=False)
    else:
        df.to_csv(out)


if __name__ == '__main__':
    cli()
