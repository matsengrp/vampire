#!/usr/bin/env Rscript

suppressMessages(library(argparse))
suppressMessages(library(devtools))
suppressMessages(library(pROC))

parser = ArgumentParser(description="Calculate AUC for Pgen or Pvae in a cross-fooling comparison.")

parser$add_argument('--pvae', action='store_true', help="We are processing Pvae, rather than default Pgen.")
parser$add_argument('--plot', help="Optional destination path for a PDF ROC plot.")
parser$add_argument('in_real', help="A CSV with a column of per-sequence probabilities of real sequences.")
parser$add_argument('in_generated', help="A CSV with a column of per-sequence probabilities of synthetically generated sequences.")
parser$add_argument('out_csv', help="Desired location for single-row output CSV. Will over-write with abandon.")

args = parser$parse_args()

if(args$pvae) {
    idx = 'log_p_x'
    take_exp = TRUE
    col_name = 'auc_pvae'
} else {
    idx = 'Ppost'
    take_exp = FALSE
    col_name = 'auc_ppost'
}


# Note that taking exp shouldn't make a difference for AUC because it's
# monotonic, but I just wanted to be sure that we were making analyses between
# the cases identical.

read_and_real = function(path, real) {
    P = read.csv(path, header=TRUE)[[idx]]
    if(take_exp) {
        P = exp(P)
    }
    data.frame(P=P, real=real)
}

df = rbind(
    read_and_real(args$in_real, TRUE),
    read_and_real(args$in_generated, FALSE)
)

we_plot = (length(args$plot) > 0)
r = roc(df$real, df$P)
out_df = data.frame(r$auc)
colnames(out_df) = col_name
write.csv(out_df, args$out_csv, row.names=FALSE)

if(we_plot) {
    pdf(args$plot)
    title = paste('AUC =', r$auc)
    plot.roc(r, cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5, main=title)
    dev.off()
}
