"""
Useful functions that don't deserve a CLI.

If they do, they belong in utils.py.
"""

import numpy as np
import pandas as pd
import pkg_resources
import os

# ### Math functions ###


def running_avg(x):
    """
    Given a 2D numpy array, calculate a running average of each of the columns.

    That is, the ijth entry is the average of the first i+1 entries of the jth
    column.

    Here's a handy plotting command for the output of running_avg:

    pd.DataFrame(running_avg(x)).plot.line(legend=False, figsize=(12,8))
    """
    running_avg = np.cumsum(x, axis=0)
    for i in range(x.shape[1]):
        running_avg[:, i] /= np.arange(1, len(x) + 1)
    return (running_avg)


def repeat_row(a, which_entry, n_repeats):
    """
    Repeat a specified row of the first axis a specified number of times.

    >>> x = np.random.rand(3, 2)
    >>> x
    array([[0.39592644, 0.92973981],
           [0.18207684, 0.54983777],
           [0.64938797, 0.01808416]])
    >>> repeat_row(x, 1, 2)
    array([[0.18207684, 0.54983777],
           [0.18207684, 0.54983777]])
    """
    assert which_entry in range(a.shape[0])

    if type(a) in [pd.Series, pd.DataFrame]:
        return a.iloc[np.full(n_repeats, which_entry)]

    assert type(a) == np.ndarray
    repeater_array = np.zeros((a.shape[0]), dtype=np.int64)
    repeater_array[which_entry] = n_repeats
    return np.repeat(a, repeater_array, axis=0)


def logspace(start, stop, num, decimals=3):
    """
    num evenly spaced numbers between start and stop, rounded to the given number of decimals.
    """
    return np.logspace(np.log10(start), np.log10(stop), num).round(decimals)


# ### Path functions ###


def strip_extn(in_path):
    """
    Strips the extension.
    """
    (path, extn) = os.path.splitext(str(in_path))
    if extn in ['.gz', '.bz2']:
        # Perhaps there is more to the suffix.
        return os.path.splitext(path)[0]
    else:
        return path


def strip_dirpath_extn(in_path):
    """
    Strips the directory path and the extension.
    """
    return os.path.basename(strip_extn(in_path))


def path_split_tail(in_path):
    """
    Give the farthest right object in a path, whether it be a directory ending
    with a `/` or a file.
    """
    return os.path.split(in_path.rstrip('/'))[1]


def cjoin(path, *paths):
    """
    This is os.path.join, but checks that the path exists.
    """
    joined = os.path.join(path, *paths)
    assert os.path.exists(joined)
    return joined


def read_data_csv(fname):
    """
    Read a CSV from our data path.
    """
    return pd.read_csv(pkg_resources.resource_filename('vampire', os.path.join('data', fname)))


# ### Misc functions ###


def zero_pad_list_func(l):
    """
    Make a function from a list of positive numbers to pad this list on the
    left with zeros according to length of the maximum number.
    """
    max_len = len(str(max(l)))
    return lambda i: str(i).zfill(max_len)


def cols_of_df(df):
    """
    Extract the data columns of a dataframe into a list of appropriately-sized
    numpy arrays.
    """
    return [np.stack(col.values) for _, col in df.items()]


def cluster_execution_string(command, localenv, prefix_position=1):
    """
    Apply this to your scons command string* to get it to execute on the
    cluster.

    *The command string but where $SOURCES is replaced by {sources} and
    $TARGETS is replaced by {targets}.

    prefix_position: from where in the command we should get the name of
    the script. 0 for scripts and 1 for subcommands.
    """
    script_prefix = strip_extn(command.split()[prefix_position])
    return (
        f"python3 execute.py --clusters='{localenv['clusters']}' --script-prefix={script_prefix} "
        f"'$SOURCES' '$TARGETS' '{command}'"
    )
