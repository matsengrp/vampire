"""
Useful functions that don't deserve a CLI.

If they do, they belong in utils.py.
"""

import numpy as np
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


# ### Misc functions ###

def zero_pad_list_func(l):
    """
    Make a function from a list of natural numbers to pad this list on the left
    with zeros according to maximum length.
    """
    max_len = len(str(max(l)))
    return lambda i: str(i).zfill(max_len)


def cols_of_df(df):
    """
    Extract the data columns of a dataframe into a list of appropriately-sized
    numpy arrays.
    """
    return [np.stack(col.values) for _, col in df.items()]
