#!/usr/bin/env Rscript

suppressMessages(library(cowplot))
suppressMessages(library(latex2exp))


### Utilities ###

# Restrict the rows of df to ones in which sample is not a trimmed version of test_set.
restrict_to_true_test = function(df) {
    # We use a convention in which processing information is added after the sample name with periods.
    # Here we trim that off.
    df$trimmed_test_set_name = sapply(strsplit(as.character(df$test_set), '\\.'), `[`, 1)
    df = df[complete.cases(df$trimmed_test_set_name), ]
    df[df$sample != df$trimmed_test_set_name, ]
}

# Do a little gymnastics to fill in fake values for OLGA (which is missing numerical_col)
# so that it will draw a line when relevant.
fake_extra_entries = function(df, numerical_col) {
    to_duplicate = df[is.na(df[,numerical_col]), ]
    to_duplicate[ , numerical_col] = max(df[,numerical_col], na.rm=TRUE)
    df[is.na(df[,numerical_col]), numerical_col] = min(df[,numerical_col], na.rm=TRUE)
    rbind(df, to_duplicate)
}

add_model_class = function(df) {
    df$class = 'dnn'
    df[df$model == 'olga', ]$class = 'graphical'
    df
}

drop_rows_with_na = function(df) { df[complete.cases(df), ] }


### Plotting ###

plot_likelihoods = function(df, numerical_col, out_path=NULL, x_trans='identity') {
    id_vars = c('test_set', 'model', numerical_col)
    measure_vars = c('test_log_mean_pvae', 'test_log_pvae_sd', 'test_median_log_pvae')

    df = df[df$model != 'olga', ]
    df = restrict_to_true_test(df)
    df = df[c(id_vars, measure_vars)]
    df$ymin = df$test_log_mean_pvae - 0.5*df$test_log_pvae_sd
    df$ymax = df$test_log_mean_pvae + 0.5*df$test_log_pvae_sd

    mean = ggplot(df, aes_string(numerical_col, 'test_log_mean_pvae', color='model')) +
           geom_line() +
           geom_errorbar(aes(ymin=ymin, ymax=ymax), alpha=0.3) +
           facet_wrap(vars(test_set), scales='free') +
           scale_x_continuous(trans=x_trans)

    medn = ggplot(df, aes_string(numerical_col, 'test_median_log_pvae', color='model')) +
           geom_line() +
           facet_wrap(vars(test_set), scales='free') +
           scale_x_continuous(trans=x_trans)

    p = plot_grid(mean, medn, labels = c("mean", "median"), nrow = 2, align = 'v')

    if(length(out_path)) ggsave(out_path, height=6, width=8)
    p
}

plot_divergences = function(df, numerical_col, out_path=NULL, x_trans='identity') {
    id_vars = c('test_set', 'model', 'class', numerical_col)
    measure_vars = grep('sumdiv_', colnames(df), value=TRUE)

    df = restrict_to_true_test(df)
    df = fake_extra_entries(df, numerical_col)
    df = add_model_class(df)
    df = df[c(id_vars, measure_vars)]

    p = ggplot(
        melt(df, id_vars, measure_vars, variable.name='divergence'),
        aes_string(numerical_col, 'value', color='model', linetype='class')
    ) + geom_line() +
        facet_grid(vars(divergence), vars(test_set), scales='free') +
        scale_x_continuous(trans=x_trans) +
        scale_y_log10() +
        theme(strip.text.y = element_text(angle = 0))

    if(length(out_path)) ggsave(out_path, height=8, width=8)
    p
}

plot_fooling = function(df, numerical_col, out_path=NULL, x_trans='identity') {
    colnames(df)[colnames(df) == 'auc_pgen'] <- 'graphical'
    colnames(df)[colnames(df) == 'auc_pvae'] <- 'dnn'
    id_vars = c('test_set', 'model', numerical_col)
    measure_vars = c('dnn', 'graphical')

    df = restrict_to_true_test(df)
    df = df[c(id_vars, measure_vars)]
    df = drop_rows_with_na(df)

    p = ggplot(
        melt(df, id_vars, measure_vars, variable.name='AUC'),
        aes_string(numerical_col, 'value', linetype='AUC', color='model')
    ) + geom_line() +
        facet_wrap(vars(test_set), scales='free') +
        scale_x_continuous(trans=x_trans)

    if(length(out_path)) ggsave(out_path, height=4, width=7)
    p
}

# For comparing the distributions of probabilities estimated by OLGA and the
# VAE.

distribution_comparison = function(df, x_label) {
    ggplot(df, aes(probability, color=source, fill=source)) +
        geom_density(alpha=0.5) + scale_x_log10() +
        xlab(TeX(x_label)) + theme(legend.position=c(0.2, 0.9))
}

pvae_distribution_comparison = function(data_path, sim_path) {
    data = read.csv(data_path)
    colnames(data) = c('probability')
    data$source = 'data'
    sim = read.csv(sim_path)
    colnames(sim) = c('probability')
    sim$source = 'OLGA'
    df = rbind(sim, data)
    df$probability = exp(df$probability)
    distribution_comparison(df, 'P_{VAE}')
}

read_olga_pgen_csv = function(path) {
    df = read.csv(path, header=FALSE)
    data.frame(probability = df[,2])
}

pgen_distribution_comparison = function(data_path, sim_path) {
    data = read_olga_pgen_csv(data_path)
    data$source = 'data'
    sim = read_olga_pgen_csv(sim_path)
    sim$source = 'VAE'
    df = rbind(sim, data)
    df$probability
    distribution_comparison(df, 'P_{OLGA}')
}
