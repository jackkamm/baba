source("code/baba.R")

remove.quartet.pops <- function(input.quartets.file, output.quartets.file, ...){
  to.remove <- c(...)
  read.table(input.quartets.file, head=T) %>%
    filter(!(Pop1 %in% to.remove)) %>%
    filter(!(Pop2 %in% to.remove)) %>%
    filter(!(Pop3 %in% to.remove)) %>%
    filter(!(Pop4 %in% to.remove)) %>%
    write.table(output.quartets.file, quote=F)
}

plot.baba.vectors <- function(baba.output, sorted.pops.fname, plot.fname){
  sorted.pops <- scan(sorted.pops.fname, what = character())

  read.table(baba.output, head=T, stringsAsFactors=F) %>%
    mutate(Population = factor(Population, levels=sorted.pops)) ->
    baba.df

  baba.df %>%
    select(Component, Weight, Mode, Population, Value) %>%
    ggplot(aes(x=Component, y=Population, fill=Value, alpha=Weight)) +
    geom_tile() +
    scale_alpha_continuous(range=c(.65,1)) +
    facet_grid(Mode ~ .) -> p

  ggsave(plot.fname, p)
}

## baba.output <- "data/scratch/newhumori_18pops/decomposition/inferred_components.txt"
## sorted.pops.fname <- "data/scratch/newhumori_18pops/sorted_pops.French.Europe_LNBA.Chimp.txt"
## plot.fname <- "data/scratch/newhumori_18pops/decomposition/plot_baba.pdf"
plot.baba.matrices <- function(baba.output, sorted.pops.fname, plot.fname, X.1, Y.1, X.2, Y.2){
  sorted.pops <- scan(sorted.pops.fname, what = character())

  read.table(baba.output, head=T, stringsAsFactors=F) %>%
    mutate(Population = factor(Population, levels=sorted.pops)) ->
    baba.df

  get.outer.df.helper <- function(df, X.mode, Y.mode){
    ret <- df[[X.mode]] %o% df[[Y.mode]]
    colnames(ret) <- df$Population
    ret %>%
      as.data.frame() %>%
      mutate(Y.pop = df$Population, Weight = df$Weight) %>%
      gather(X.pop, Value, -Y.pop, -Weight) %>%
      mutate(X.pop = factor(X.pop, levels=sorted.pops)) %>%
      mutate(X.mode = X.mode, Y.mode = Y.mode)
  }
  get.outer.df <- function(df){
    rbind(
      df %>% get.outer.df.helper(X.1, Y.1),
      df %>% get.outer.df.helper(X.2, Y.2)
    )
  }
  n.components <- length(levels(factor(baba.df$Component)))
  for (i in seq(1, n.components, by=4)){
    baba.df %>%
      filter(Component %in% i:(i+3)) %>%
      select(Component, Weight, Mode, Population, Value) %>%
      spread(Mode, Value) %>%
      group_by(Component) %>%
      do(get.outer.df(.)) %>%
      ungroup() %>%
      mutate(Facet = paste("X =", X.mode, ", Y =", Y.mode)) %>%
      ggplot(aes(x=X.pop, y=Y.pop, fill=Value, alpha=Weight)) +
      geom_tile() +
      #scale_alpha_continuous(breaks=(0:5)/5*max(baba.df$Weight), limits=c(0, max(baba.df$Weight))) +
      #scale_alpha_continuous(trans="log", limits=c(min(baba.df$Weight), max(baba.df$Weight))) +
      scale_alpha_continuous(range=c(.65,1), limits=c(baba.df %>% filter(Weight > 0) %>% with(min(Weight)),
                                                      max(baba.df$Weight))) +
      scale_fill_gradient(low="gray", high="red") +
      facet_wrap(~ Component + Facet, ncol=4) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) ->
      p

    ggsave(paste(plot.fname, i, "-", i+3, ".pdf", sep=""),
           p, width=14, height=7)
  }
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
