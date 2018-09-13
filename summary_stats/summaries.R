devtools::load_all("/fh/fast/matsen_e/bolson2/sumrep")

library(plyr)
library(xtable)

printTable <- function(dat, filename, digits=4, hline_pos=0) {
    sink(filename)
    cat("\\begin{center}", '\n')
    dat %>%
        xtable::xtable(digits=digits) %>%
        print(
              floating=FALSE,
              include.rownames=FALSE,
              latex.environments="center",
              hline.after=c(0, hline_pos),
              sanitize.text.function=function(x){x}
             )
    cat("\\end{center}", '\n')
    sink()
}
    
multiplot <- function(..., plotlist=NULL, file, cols=1, rows=1, layout=NULL) {
    library(grid)
    plots <- c(list(...), plotlist)
    numPlots = length(plots)

    if(is.null(layout)) {
        layout <- matrix(seq(1, cols * rows),
                         ncol = cols, nrow = rows)
    }
    if (numPlots==1) {
       print(plots[[1]])
    } else {
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
        for(i in 1:numPlots) {
            matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
            print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                            layout.pos.col = matchidx$col))
        }   
    }     
} 

agg_file <- "/fh/fast/matsen_e/data/dnnir/vampire/summary_stats/all_seshadri_data/all_TCRB_KD_cut.tsv"
igor_agg_file <- "/fh/fast/matsen_e/data/dnnir/vampire/summary_stats/agg_igor_gen_seqs.csv"
vae_agg_file <- "/fh/fast/matsen_e/data/dnnir/vampire/summary_stats/all_seshadri_data/all_TCRB_KD_cut_predictions.tsv"
cmv_file <- "/fh/fast/matsen_e/data/dnnir/vampire/summary_stats/largest_CMV_sample/HIP13427_KD_cut.tsv"
igor_cmv_file <- "/fh/fast/matsen_e/data/dnnir/vampire/summary_stats/cmv_igor_gen_seqs.csv"
vae_cmv_file <- "/fh/fast/matsen_e/data/dnnir/vampire/summary_stats/largest_CMV_sample/all_TCRB_KD_cut_predictions_HIP13427.tsv"


read_files <- FALSE
if(read_files) {
    agg_dat <- data.table::fread(agg_file, header=FALSE)
    names(agg_dat) <- c("cdr3s", "cdr3_aa", "v_gene", "j_gene")
    agg_dat$cdr3s <- agg_dat$cdr3s %>% tolower

    igor_agg_dat <- data.table::fread(igor_agg_file)
    igor_agg_dat$cdr3s <- igor_agg_dat$nt_CDR3 %>% tolower
    igor_agg_dat$cdr3_aa <- igor_agg_dat$cdr3s %>% 
        convertNucleobasesToAminoAcids

    vae_agg_dat <- data.table::fread(vae_agg_file, header=FALSE)
    names(vae_agg_dat) <- c("cdr3_aa", "v_gene", "j_gene")

    cmv_dat <- data.table::fread(cmv_file, header=FALSE)
    names(cmv_dat) <- c("cdr3s", "cdr3_aa", "v_gene", "j_gene")
    cmv_dat$cdr3s <- cmv_dat$cdr3s %>% tolower

    igor_cmv_dat <- data.table::fread(igor_cmv_file)
    igor_cmv_dat$cdr3s <- igor_cmv_dat$nt_CDR3 %>% tolower
    igor_cmv_dat$cdr3_aa <- igor_cmv_dat$cdr3s %>% 
        convertNucleobasesToAminoAcids

    vae_cmv_dat <- data.table::fread(vae_cmv_file, header=FALSE)
    names(vae_cmv_dat) <- c("cdr3_aa", "v_gene", "j_gene")
    vae_cmv_dat <- vae_cmv_dat %>% 
        subsample(nrow(cmv_dat))
}

dat_list <- list(agg_dat, 
                 igor_agg_dat, 
                 vae_agg_dat,
                 cmv_dat,
                 igor_cmv_dat,
                 vae_cmv_dat
                 )
dat_types <- rep(c("Observed", "IGoR sim", "VAE sim"), 2)
dat_labels <- c(rep("Aggregate", 3), rep("Individual", 3))
ggplot_text_size <- theme_get() %$%
    text %$%
    size

getSummaryDat <- function(dat,
                          dat_name,
                          dat_label,
                          summary_function
) {
    print(dat_name)
    dat_summary <- dat %>% 
        summary_function
    summary_dat <- data.table(Summary=dat_summary,
                              Dataset=dat_name,
                              Label=dat_label
                              )
    return(summary_dat)
}

getSequenceLengthDistribution <- function(sequences) {
    lengths <- sequences %>%
        sapply(nchar) %>%
        sapply(function(x) { ifelse(x < 30, x, NA) })
    return(lengths)
}

compareSequenceLengthDistributions <- function(dat_a,
                                               dat_b) {
    dist_a <- dat_a %$% 
        cdr3_aa %>% 
        getSequenceLengthDistribution
    dist_b <- dat_b %$%
        cdr3_aa %>% 
        getSequenceLengthDistribution
    divergence <- getJSDivergence(dist_a, dist_b)
    return(divergence)
}

do_full <- FALSE
if(do_full) { 
    summary_functions <- list("getSequenceLengthDistribution",
                              "getHydrophobicityDistribution",
                              "getAliphaticIndexDistribution",
                              "getGRAVYDistribution"
                             )
    
    summary_labels <- list("CDR3 length (AA)",
                           "Hydrophobicity",
                           "Aliphatic Index",
                           "GRAVY index"
                          )
    
    smoothness <- list(6, 1, 1.5, 1)
    
    plots <- {}
    agg_plots <- {}
    cmv_plots <- {}
    i <- 1
    for(summary_function in summary_functions) {
        summary_string <- paste0(summary_function, "_dists")
        assign(summary_string,
               dat_list %>% 
                   lapply(function(x) { x %$% cdr3_aa }) %>% 
                   mapply(getSummaryDat,
                          .,
                          dat_types,
                          dat_labels,
                          MoreArgs=list(eval(parse(text=summary_function))),
                          SIMPLIFY=FALSE
                          ) 
                   )
        summary_dat <- 
            do.call(rbind, eval(parse(text=summary_string)))
        
        plots[[i]] <- ggplot(summary_dat, 
                             aes(x=Summary, 
                                 colour=Dataset,
                                 lty=Label
                                 )
                             ) + 
            geom_density(adjust=smoothness[[i]]) +
            xlab(summary_labels[[i]]) +
            ylab("Density")

        agg_plots[[i]] <- ggplot(summary_dat[summary_dat$Label == "Aggregate", ],
                             aes(x=Summary, 
                                 colour=Dataset
                                 )
                             ) + 
            geom_density(adjust=smoothness[[i]]) +
            xlab(summary_labels[[i]]) +
            ylab("Density")

        cmv_plots[[i]] <- ggplot(summary_dat[summary_dat$Label == "Individual", ],
                             aes(x=Summary, 
                                 colour=Dataset
                                 )
                             ) + 
            geom_density(adjust=smoothness[[i]]) +
            xlab(summary_labels[[i]]) +
            ylab("Density") +
            theme(axis.title=element_text(size=1.8*ggplot_text_size),
                  legend.title=element_text(size=1.8*ggplot_text_size),
                  axis.text=element_text(size=1.5*ggplot_text_size),
                  legend.text=element_text(size=1.5*ggplot_text_size))


        i <- i + 1
    }
    pdf("physiochem.pdf", width=10, height=6)
    multiplot(plotlist=plots, cols=2, rows=2)
    dev.off()

    pdf("physiochem_agg.pdf", width=10, height=6)
    multiplot(plotlist=agg_plots, cols=2, rows=2)
    dev.off()

    pdf("physiochem_cmv.pdf", width=10, height=6)
    multiplot(plotlist=cmv_plots, cols=2, rows=2)
    dev.off()
}

do_aa <- TRUE
if(do_aa) {
    aa_dat <- dat_list %>%
        lapply(getAminoAcidDistribution) %>%
        lapply(data.frame) %>%
        Map(cbind, ., dat_types) %>%
        Map(cbind, ., dat_labels) %>%
        lapply(setNames, c("AA", "Frequency", "Dataset", "Label")) %>%
        do.call(rbind, .)
}

p1 <- ggplot(aa_dat[aa_dat$Label == "Aggregate", ], 
             aes(x=AA, y=Frequency, fill=Dataset)) +
        geom_bar(stat="identity", position="dodge") +
        xlab("Amino acid") +
        theme(axis.title=element_text(size=1.8*ggplot_text_size),
              legend.title=element_text(size=1.8*ggplot_text_size),
              axis.text=element_text(size=1.5*ggplot_text_size),
              legend.text=element_text(size=1.5*ggplot_text_size))
p2 <- ggplot(aa_dat[aa_dat$Label == "Individual", ], 
             aes(x=AA, y=Frequency, fill=Dataset)) +
        geom_bar(stat="identity", position="dodge") +
        xlab("Amino acid") +
        theme(axis.title=element_text(size=1.8*ggplot_text_size),
              legend.title=element_text(size=1.8*ggplot_text_size),
              axis.text=element_text(size=1.5*ggplot_text_size),
              legend.text=element_text(size=1.5*ggplot_text_size))
pdf("aa.pdf", width=10, height=6)
multiplot(plotlist=list(p1, p2), cols=1, rows=2)
dev.off()

do_divergences <- TRUE
if(do_divergences) {
    div_functions <- list("compareSequenceLengthDistributions",
                          "compareAliphaticIndexDistributions",
                          "compareGRAVYDistributions",
                          "compareAminoAcidDistributions",
                          "compareAminoAcid2merDistributions"
                         )
    agg_igor_divs <- {}
    cmv_igor_divs <- {}

    agg_vae_divs <- {}
    cmv_vae_divs <- {}

    agg_dat_list <- list(agg_dat,
                     igor_agg_dat,
                     vae_agg_dat) 

    cmv_dat_list <- list(cmv_dat,
                     igor_cmv_dat,
                     vae_cmv_dat) 

    for(div_function in div_functions) {
        # Get aggregate dataset divergences
        agg_igor_div <- doComparison(div_function, agg_dat_list[1:2]) %$% 
            Divergence
        agg_vae_div <- doComparison(div_function, agg_dat_list[c(1, 3)]) %$% 
            Divergence

        agg_igor_divs <- c(agg_igor_divs, agg_igor_div)
        agg_vae_divs <- c(agg_vae_divs, agg_vae_div)

        # Get individual dataset divergences
        cmv_igor_div <- doComparison(div_function, cmv_dat_list[1:2]) %$% 
            Divergence
        cmv_vae_div <- doComparison(div_function, cmv_dat_list[c(1, 3)]) %$% 
            Divergence

        cmv_igor_divs <- c(cmv_igor_divs, cmv_igor_div)
        cmv_vae_divs <- c(cmv_vae_divs, cmv_vae_div)
    }
}

div_labels <- c("CDR3 lengths (AA)",
                "Aliphatic indices",
                "GRAVY indices",
                "Amino acid frequencies",
                "Amino acid 2mer frequencies"
               )

div_types <- c("JS",
               "JS",
               "JS",
               "$\\ell_1$",
               "$\\ell_1$"
              )

agg_table <- cbind.data.frame(div_labels,
                              agg_igor_divs,
                              agg_vae_divs,
                              div_types)

cmv_table <- cbind.data.frame(div_labels,
                              cmv_igor_divs,
                              cmv_vae_divs,
                              div_types)

table_names <- c("Summary", 
                 "Div to IGoR", 
                 "Div to VAE", 
                 "Div"
                )
agg_table %>%
    setNames(table_names) %>%
    printTable("agg_div.tex")

cmv_table %>%
    setNames(table_names) %>%
    printTable("cmv_div.tex")
