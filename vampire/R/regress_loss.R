#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(devtools))

parser = ArgumentParser(description='Perform regression to obtain optimal loss weights.')


parser$add_argument('in_pvae', help='A CSV with a column of per-sequence log Pvaes.')
parser$add_argument('in_per_seq_loss', help='A CSV with per-sequence losses.')
parser$add_argument('out_csv', help='Desired location for single-row regression weight CSV. Will over-write with abandon.')
parser$add_argument('--idx', required=TRUE, help='A row label.')
parser$add_argument('--idx-name', required=TRUE, help='The header for the row label.')


args = parser$parse_args()

pvae = read.csv(args$in_pvae)
losses = read.csv(args$in_per_seq_loss)
# Drop the total weighted loss: we want each one individually.
losses = losses[, !names(losses) == 'loss']
fit = lm(log_p_x ~ ., data = cbind(pvae, losses))
coeffs = t(data.frame(fit$coefficients))
row_label = data.frame(c(args$idx))
colnames(row_label) = args$idx_name
write.csv(cbind(row_label, coeffs), args$out_csv, row.names=FALSE)
