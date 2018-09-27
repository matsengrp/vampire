import sys, os
import pandas as pd
from Bio.SeqUtils import GC
import matplotlib.pyplot as plt
import numpy as np
import random
import re

# May need to move this to /fh/fast/
sys.path.insert(0, '/home/bolson2/Software/IGoR')

# import pygor
# from pygor import counters
# from pygor.counters import bestscenarios
# from pygor.counters.bestscenarios import bestscenarios

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
def get_annotations(igor_wd, chain):
    scenarios_filename = "igor_generated/generated_realizations_werr.csv"
    scenarios_file = os.path.join(igor_wd, scenarios_filename)
    model_parms_filename = "igor_evaluate/final_parms.txt"
    model_parms_file = os.path.join(igor_wd, model_parms_filename)
    annotation_dat = bestscenarios.read_bestscenarios_values(scenarios_file, model_parms_file)

    annotation_dat['v_gene'] = annotation_dat['v_choice'].apply( \
            lambda(x): extract_gene_from_full_string(x, chain, "V") )
    annotation_dat['j_gene'] = annotation_dat['j_choice'].apply( \
            lambda(x): extract_gene_from_full_string(x, chain, "J") )

    # Write annotations dataset to file
    output_file = "annotations.csv"
    annotation_dat.to_csv(os.path.join(igor_wd, output_file))
    return annotation_dat

def extract_gene_from_full_string(s, chain, gene):
    chain_letter = {"beta": "B",
                    "alpha": "A"}
    regex = "TR" + chain_letter[chain] + gene + "[0-9]+([-][0-9]+)?[*][0-9]+"
    regex_result = re.search(regex, s)
    gene_string = regex_result.group(0) if regex_result is not None else None
    return(gene_string)

def run_igor_analysis(input_file,
                      sequences_filename,
                      igor_dir_name,
                      subsample=False,
                      chain="beta"
                     ):
    cwd = os.getcwd()
    sequence_file = os.path.join(cwd, sequences_filename)
    get_seqs_from_tsv( 
        input_file,
        sequence_file,
        subsample=False)
    igor_directory = os.path.join(cwd, igor_dir_name)
    if not os.path.exists(igor_directory):
        os.makedirs(igor_directory)
    run_igor(sequences_filename, igor_directory)
    # annotations = get_annotations(igor_directory, chain)
    return 32


if __name__ == "__main__":
    do_aggregate = False
    if do_aggregate:
        agg_annotations = run_igor_analysis(
            "/fh/fast/matsen_e/kdavidse/data/dnnir/vampire/summary_stats/all_seshadri_data/all_TCRB_KD_cut.tsv",
            "agg_seqs.txt",
            "tmp_aggregate_quoll",
            True
        )

    do_individual = True
    if do_individual:
        cmv_annotations = run_igor_analysis(
            "/fh/fast/matsen_e/kdavidse/data/dnnir/vampire/summary_stats/largest_CMV_sample/HIP13427_BJO_train.tsv",
            "cmv_seqs.txt",
            "igor_cmv",
            False,
            "beta"
        )
