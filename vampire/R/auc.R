#!/usr/bin/env Rscript

suppressMessages(library(argparse))
suppressMessages(library(devtools))
suppressMessages(library(pROC))

parser = ArgumentParser(description="Calculate AUC for Pgen or Pvae in a cross-fooling comparison.")

parser$add_argument('--pvae', action='store_true', help="We are processing Pvae, rather than default Pgen.")
parser$add_argument('in_real', help="A CSV with a column of per-sequence probabilities of real sequences.")
parser$add_argument('in_generated', help="A CSV with a column of per-sequence probabilities of synthetically generated sequences.")
parser$add_argument('out_csv', help="Desired location for single-row output CSV. Will over-write with abandon.")

args = parser$parse_args()

# Note that taking exp shouldn't make a difference for AUC, but I just wanted
# to be sure that we were making analyses as similar as possible.
if(args$pvae) {
    header=TRUE
    idx=1
    take_exp = TRUE
    col_name = 'auc_pvae'
} else {
    header = FALSE
    idx = 2
    take_exp = FALSE
    col_name = 'auc_pgen'
}


read_and_real = function(path, real) {
    P = read.csv(path, header=header)[[idx]]
    if(take_exp) {
        P = exp(P)
    }
    data.frame(P=P, real=real)
}

df = rbind(
    read_and_real(args$in_real, TRUE),
    read_and_real(args$in_generated, FALSE)
)

r = roc(df$real, df$P)
out_df = data.frame(r$auc)
colnames(out_df) = col_name
write.csv(out_df, args$out_csv, row.names=FALSE)
