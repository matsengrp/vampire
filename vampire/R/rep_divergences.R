#!/usr/bin/env Rscript
# Usage:
#   Rscript --vanilla GetDivergences.R file1.csv file2.csv [out_filename]
#   If out_filename is not specified, it saves the divergences to "divergences.csv"
# Note that this does full NN distance calculation, an O(n^2) operation, so don't
# try this for anything too big!
args <- commandArgs(TRUE)
if(length(args) < 2) {
    stop("Must supply two files for comparison.")
} else if(length(args) > 3) {
    stop("Unused arguments (at most 3 allowed).")
}

suppressMessages(library(devtools))
suppressMessages(devtools::load_all('R/sumrep', quiet=TRUE))

getDataTableFromCsv <- function(i) {
    dat <- args[i] %>% data.table::fread()
    names(dat) <- c("junction_aa", "v_call", "j_call")
    return(dat)
}

dat_a <- getDataTableFromCsv(1)
dat_b <- getDataTableFromCsv(2)

divs <- {}
divs$pairwise_distance <- comparePairwiseDistanceDistributions(dat_a,
                                                               dat_b,
                                                               column="junction_aa",
                                                               approximate=TRUE,
                                                               tol=1e-6
                                                              )
divs$nn_distance <- compareNNDistanceDistributions(dat_a,
                                                   dat_b,
                                                   column="junction_aa",
                                                   approximate=TRUE,
                                                   tol=1e-6
                                                  )
divs$cdr3_length <- compareCDR3LengthDistributions(dat_a, dat_b, by_amino_acid=TRUE)
divs$aliphatic_index <- compareAliphaticIndexDistributions(dat_a, dat_b)
divs$gravy_index <- compareGRAVYDistributions(dat_a, dat_b)
divs$polarity <- comparePolarityDistributions(dat_a, dat_b)
divs$charge <- compareChargeDistributions(dat_a, dat_b)
divs$basicity <- compareBasicityDistributions(dat_a, dat_b)
divs$acidity <- compareAcidityDistributions(dat_a, dat_b)
divs$aromaticity <- compareAromaticityDistributions(dat_a, dat_b)
divs$bulkiness <- compareBulkinessDistributions(dat_a, dat_b)
divs$v_gene_freq <- compareVGeneDistributions(dat_a, dat_b)
divs$j_gene_freq <- compareJGeneDistributions(dat_a, dat_b)
divs$aa_freq <- compareAminoAcidDistributions(dat_a, dat_b)
divs$aa_2mer_freq <- compareAminoAcid2merDistributions(dat_a, dat_b)

out_filename <- ifelse(length(args) == 3, args[3], "divergences.csv")
divs %>%
    as.data.table %>%
    data.table::fwrite(file=out_filename)
