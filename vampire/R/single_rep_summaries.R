#!/usr/bin/env Rscript

# I'm not currently using this script, because I'm interested in sumdiv and
# these summaries aren't actually all that informative.
# But, if you want to use them the following targets may be helpful:

# @nest.add_target_with_env(localenv)
# def test_sumrep(env, outdir, c):
#     test_sumrep_path = common.strip_extn(c['test_head'])+'.sumrep.csv'
#     c['test_set_info_agg'][(str(c['test_head']), 'test_sumrep')] = test_sumrep_path
#     return env.Command(
#         test_sumrep_path,
#         c['test_head'],
#         'R/single_rep_summaries.R $SOURCE $TARGET')[0]

# @nest.add_target_with_env(localenv)
# def vae_generated_sumrep(env, outdir, c):
#     return env.Command(
#         common.strip_extn(c['vae_generated'])+'.sumrep.csv',
#         c['vae_generated'],
#         'R/single_rep_summaries.R $SOURCE $TARGET')[0]

# @nest.add_target_with_env(localenv)
# def olga_generated_sumrep(env, outdir, c):
#     """
#     Run univariate sumrep on the OLGA-generated sequences.
#     """
#     return env.Command(
#         common.strip_extn(c['olga_generated'])+'.sumrep.csv',
#         c['olga_generated'],
#         'R/single_rep_summaries.R $SOURCE $TARGET')[0]

# And in util.py summarize:

#         elif name == 'vae_generated_sumrep':
#             slurp_cols(path, prefix='sumrep_')


suppressMessages(library(argparse))
suppressMessages(library(devtools))

parser = ArgumentParser(description='Calculate single-repertoire summaries, writing out a single-row CSV.')
parser$add_argument('in_csv', help='A CSV with `amino_acid,v_gene,j_gene` columns.')
parser$add_argument('out_csv', help='Desired location for summary CSV. Will over-write with abandon.')

args = parser$parse_args()

suppressMessages(devtools::load_all('R/sumrep', quiet=TRUE))

in_df = read.csv(args$in_csv, stringsAsFactors=FALSE)

cdr3_len_distrib = sapply(in_df$amino_acid, nchar)
pwd_distrib = getPairwiseDistanceDistribution(in_df, column='amino_acid')
aliphatic_distrib = getAliphaticIndexDistribution(data.table(junction_aa=in_df$amino_acid))

make_distrib_summary = function(prefix, distrib) {
    summary_df = data.frame(
        mean=mean(pwd_distrib),
        median=median(pwd_distrib),
        sd=sd(pwd_distrib))
    colnames(summary_df) = sapply(colnames(summary_df), function(s) paste(prefix, s, sep='_'))
    summary_df
}

out_df = cbind(
    make_distrib_summary('cdr3_len', cdr3_len_distrib),
    make_distrib_summary('pwd', pwd_distrib),
    make_distrib_summary('alphatic', aliphatic_distrib)
)

write.csv(out_df, args$out_csv, row.names=FALSE)
