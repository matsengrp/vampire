#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(devtools))

parser = ArgumentParser(description='Calculate single-repertoire summaries, writing out a single-row CSV.')
parser$add_argument('in_csv', help='A CSV with `amino_acid,v_gene,j_gene` columns.')
parser$add_argument('out_csv', help='Desired location for summary CSV. Will over-write with abandon.')

args = parser$parse_args()

suppressPackageStartupMessages(devtools::load_all('R/sumrep', quiet=TRUE))

pwd_distrib = getPairwiseDistanceDistribution(read.csv(args$in_csv, stringsAsFactors=FALSE), column='amino_acid')

summary_df = data.frame(
    pwd_mean=mean(pwd_distrib),
    pwd_median=median(pwd_distrib),
    pwd_sd=sd(pwd_distrib))

write.csv(summary_df, args$out_csv, row.names=FALSE)
