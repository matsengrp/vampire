#!/usr/bin/env Rscript

# Here's some R I used to process the aggregated results of this command:
#
#  df = read.csv(paste(dir, 'loss_regression.agg.csv', sep=''))
#  df = df[, colnames(df)!='X.Intercept.']
#  df = df[, colnames(df)!='beta']
#
#  ggplot(
#      drop_rows_with_na(melt(df, c('sample', 'model'))),
#          aes(model, value, color=model)
#          ) + geom_point() + facet_wrap(vars(variable), scales='free') + theme(axis.text.x=element_blank())
#
# medians = df[-1] %>% group_by(model) %>% summarize_all(funs(median))
# normalized = medians / medians$cdr3_output_loss
# normalized$model = medians$model
# write(toJSON(normalized, pretty=TRUE), file='normalized.json')

suppressMessages(library(argparse))
suppressMessages(library(devtools))

parser = ArgumentParser(description='Perform regression to obtain optimal loss weights.')


parser$add_argument('in_pvae', help='A CSV with a column of per-sequence log Pvaes.')
parser$add_argument('in_per_seq_loss', help='A CSV with per-sequence losses.')
parser$add_argument('out_csv', help='Desired location for single-row regression weight CSV. Will over-write with abandon.')
parser$add_argument('--idx', required=TRUE, help='A row label.')
parser$add_argument('--idx-name', required=TRUE, help='The header for the row label.')


args = parser$parse_args()

pvae = read.csv(args$in_pvae)
losses = read.csv(args$in_per_seq_loss)
# Despite my best efforts, on quokka row labels are getting drug in as X. Drop it
losses = losses[, !names(losses) == 'X']
# Drop the total weighted loss: we want each one individually.
losses = losses[, !names(losses) == 'loss']
fit = lm(log_p_x ~ ., data = cbind(pvae, losses))
coeffs = t(data.frame(fit$coefficients))
row_label = data.frame(c(args$idx))
colnames(row_label) = args$idx_name
write.csv(cbind(row_label, coeffs), args$out_csv, row.names=FALSE)
