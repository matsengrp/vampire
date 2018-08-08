import sys, os
import pandas as pd
from Bio.SeqUtils import GC
import matplotlib.pyplot as plt
import numpy as np
import random

# May need to move this to /fh/fast/
sys.path.insert(0, '/home/bolson2/Software/IGoR')

import pygor
from pygor import counters
from pygor.counters import bestscenarios
from pygor.counters.bestscenarios import bestscenarios

def get_seqs_from_tsv(input_file, \
                      output_file, \
                      subsample=True, \
                      sample_count=100000):
    df = pd.read_csv(input_file, \
                               sep='\t', \
                               index_col=None)
    df_samp = df.loc[random.sample(list(df.index), sample_count)] \
              if subsample \
              else df
    seqs = df_samp.ix[:, 0]
    seqs.to_csv(output_file, index=False, header=False)

def run_igor(input_file, \
             igor_wd, \
             num_scenarios=1, \
             eval_batch_name="igor", \
             gen_batch_name="igor", \
             chain="beta", \
             species="human" \
             ):
    df = pd.read_csv(input_file)
    df.columns = ['Sequence']
    generated_seq_count = df.shape[0]
    
    igor_command = "sh run_igor.sh" + \
            " -w " + igor_wd + \
            " -i " + str(input_file) + \
            " -n " + str(num_scenarios) + \
            " -g " + str(generated_seq_count) + \
            " -e " + eval_batch_name + \
            " -b " + gen_batch_name + \
            " -c " + chain + \
            " -s " + species 
    
    print igor_command
    os.system(igor_command)
    
    sim_df = pd.read_csv(os.path.join(igor_wd, \
                                      gen_batch_name + "_generated", \
                                      "generated_seqs_werr.csv"),
                         sep=";")

# Get and write annotation dataframe via pygor subpackage routine
def write_gene_info(igor_wd):
    scenarios_filename = "igor_generated/generated_realizations_werr.csv"
    scenarios_file = os.path.join(igor_wd, scenarios_filename)
    model_parms_filename = "igor_evaluate/final_parms.txt"
    model_parms_file = os.path.join(igor_wd, model_parms_filename)
    bs = bestscenarios.read_bestscenarios_values(scenarios_file, model_parms_file)
    output_file = "annotations.csv"
    bs.to_csv(os.path.join(igor_wd, output_file))

if __name__ == "__main__":
    cwd = os.getcwd()
    agg_seq_file = "agg_seqs.txt"
    get_seqs_from_tsv(
        "/fh/fast/matsen_e/kdavidse/data/dnnir/vampire/summary_stats/all_seshadri_data/all_TCRB_KD_cut.tsv",
        os.path.join(cwd, agg_seq_file),
        subsample=True)
    agg_dir = "igor_aggregate"
    agg_wd = os.path.join(cwd, agg_dir)
    if not os.path.exists(agg_wd):
        os.makedirs(agg_wd)
    run_igor(agg_seq_file, agg_wd)
    write_gene_info(agg_wd)

    cmv_seq_file = "cmv_seqs.txt"
    get_seqs_from_tsv( 
        "/fh/fast/matsen_e/kdavidse/data/dnnir/vampire/summary_stats/largest_CMV_sample/HIP13427_KD_cut.tsv",
        os.path.join(cwd, cmv_seq_file), 
        subsample=True)
    cmv_dir = "igor_cmv"
    cmv_wd = os.path.join(cwd, cmv_dir)
    if not os.path.exists(cmv_wd):
        os.makedirs(cmv_wd)
    run_igor(cmv_seq_file, cmv_wd)
    write_gene_info(cmv_wd)
