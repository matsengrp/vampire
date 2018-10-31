"""
Useful functions that don't deserve a CLI.

If they do, they belong in utils.py.
"""

import numpy as np


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
