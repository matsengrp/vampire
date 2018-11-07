import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def heatmap_by_dirs(glob_str, d1_name, d2_name):
    """
    Aggregate data from a 2-deep integer directory hierarchy nested in one
    indexed by data sets and make an array of heatmaps.
    """
    dfs = []
    for path in glob.glob(glob_str):
        run_on, d1, d2 = path.split('/')[-4:][0:3]
        df = pd.read_csv(path, index_col=0).T
        df.index.names = ['train_vs_test']
        for k, v in [('run_on', run_on), (d1_name, int(d1)), (d2_name, int(d2))]:
            df[k] = v
            df.set_index(k, append=True, inplace=True)
        dfs.append(df / 1000)

    df = pd.concat(dfs).sort_index()
    n_data_sets = len(df.index.unique(level='run_on'))

    p, axs = plt.subplots(2, n_data_sets, figsize=(16, 16))
    for i, (train_vs_test, train_vs_test_df) in enumerate(df.groupby(level=0)):
        for j, (run_on, run_on_df) in enumerate(train_vs_test_df.groupby(level=1)):
            to_plot = run_on_df[['loss']].pivot_table(index=d1_name, columns=d2_name)
            to_plot.columns = to_plot.columns.droplevel(0)
            sns.heatmap(to_plot, ax=axs[i, j], cmap='cividis_r', annot=True)
            axs[i, j].set_title(f"{run_on} {train_vs_test}")
