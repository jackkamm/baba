source("code/baba.R")

## baba.output <- "data/scratch/newhumori_18pops/decomposition/inferred_components.txt"
## sorted.pops.fname <- "data/scratch/newhumori_18pops/sorted_pops.French.Europe_LNBA.Chimp.txt"
## plot.fname <- "data/scratch/newhumori_18pops/decomposition/plot_baba.pdf"
plot.baba <- function(baba.output, sorted.pops.fname, plot.fname){
  sorted.pops <- scan(sorted.pops.fname, what = character())

  read.table(baba.output, head=T, stringsAsFactors=F) %>%
    mutate(Population = factor(Population, levels=sorted.pops)) ->
    baba.df

  baba.df %>%
    ggplot(aes(y=Population, x=Component, fill=Value)) +
    geom_tile() +
    scale_fill_gradient(low="gray", high="red") +
    facet_grid(Mode ~ .) ->
    p

  ggsave(plot.fname, p, width=7, height=10)
}

## df.filename <- "data/scratch/newhumori_18pops/all_quartets_df.txt"
## x <- "French"
## y <- "Europe_LNBA"
## a <- "Chimp"
run.sort.pops.bbaa <- function(df.filename, x, y, a){
  read.table(df.filename, head=T, stringsAsFactors = F) %>%
    sort.pops.bbaa(x, y, a) %>%
    write("")
}

extract_qpDstats_quartets_all <- function(qpDstats_cleaned_output, quartets_df_filename){
  qp.df.all <- read.table(qpDstats_cleaned_output,
                      col.names = c("Pop1","Pop2","Pop3","Pop4", "ABBA_BABA", "Z.score", "is_best", "BABA", "ABBA", "n.snps"),
                      stringsAsFactors = F) %>%
    ## drop is_best parameter, it is not a good indication of topology (allows "topologies" BBAA < ABBA,BABA)
    select(Pop1, Pop2, Pop3, Pop4, Z.score, BABA, ABBA, n.snps)

  ## duplicate rows for all permutations
  qp.df.all <- qp.df.all %>%
    bind_rows(qp.df.all %>%
                transform(Pop1=Pop2, Pop2=Pop1, Z.score=-Z.score, BABA=ABBA, ABBA=BABA))
  qp.df.all <- qp.df.all %>%
    bind_rows(qp.df.all %>%
                transform(Pop3=Pop4, Pop4=Pop3, Z.score=-Z.score, BABA=ABBA, ABBA=BABA))
  qp.df.all <- qp.df.all %>%
    bind_rows(qp.df.all %>%
                transform(Pop1=Pop3, Pop3=Pop1, Pop2=Pop4, Pop4=Pop2))

  ## add BBAA column and return
  qp.df.all %>%
    select(Pop1, Pop2, Pop3, Pop4, BABA) %>%
    transform(Pop2=Pop3, Pop3=Pop2, BBAA=BABA) %>%
    select(-BABA) %>%
    group_by(Pop1,Pop2,Pop3,Pop4) %>% summarize(BBAA=unique(BBAA)) %>%
    inner_join(x=qp.df.all) %>%
    write.table(quartets_df_filename, row.names=F, quote=F)
}

args <- commandArgs(trailingOnly=TRUE)
do.call(args[1], as.list(args[-1]))
