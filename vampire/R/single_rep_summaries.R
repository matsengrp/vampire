#!/usr/bin/env Rscript
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
