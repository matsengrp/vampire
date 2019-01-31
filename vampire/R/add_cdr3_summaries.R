#!/usr/bin/env Rscript

suppressMessages(library(argparse))
suppressMessages(library(devtools))

parser = ArgumentParser(description='Calculate CDR3 summaries and add them as columns to the given data frame.')
parser$add_argument('in_csv', help='A CSV with an amino_acid column with CDR3 sequence.')
parser$add_argument('out_csv', help='Desired output location. Will over-write with abandon.')

args = parser$parse_args()

suppressMessages(devtools::load_all('R/sumrep', quiet=TRUE))

df = read.csv(args$in_csv, stringsAsFactors=FALSE, check.names=FALSE)
df$cdr3_length = nchar(df$amino_acid)

# sumrep likes the CDR3 column to be called junction_aa.
df$junction_aa = df$amino_acid
df$charge = getChargeDistribution(df)
df$gravy = getGRAVYDistribution(df)
df$aliphatic_index = getAliphaticIndexDistribution(df)
df$polarity = getPolarityDistribution(df)
df$charge = getChargeDistribution(df)
df$basicity = getBasicityDistribution(df)
df$acidity = getAcidityDistribution(df)
df$aromaticity = getAromaticityDistribution(df)
df$bulkiness = getBulkinessDistribution(df)
df$junction_aa <- NULL

write.csv(df, args$out_csv, row.names=FALSE)
