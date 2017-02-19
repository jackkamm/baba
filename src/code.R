library(dplyr)

extract_qpDstats_quartets <- function(qpDstats_table_filename){
  qp.df.all <- read.table(qpDstats_table_filename,
                      col.names = c("X","Y","Z","A", "ABBA_BABA", "Z.score", "is_best", "BABA", "ABBA", "n.snps"),
                      stringsAsFactors = F) %>%
    ## drop is_best parameter, it is not a good indication of topology (allows "topologies" BBAA < ABBA,BABA)
    select(X, Y, Z, A, Z.score, BABA, ABBA, n.snps)

  ## duplicate rows for all permutations
  qp.df.all <- qp.df.all %>%
    bind_rows(qp.df.all %>%
                transform(X=Y, Y=X, Z.score=-Z.score, BABA=ABBA, ABBA=BABA))
  qp.df.all <- qp.df.all %>%
    bind_rows(qp.df.all %>%
                transform(Z=A, A=Z, Z.score=-Z.score, BABA=ABBA, ABBA=BABA))
  qp.df.all <- qp.df.all %>%
    bind_rows(qp.df.all %>%
                transform(X=Z, Z=X, Y=A, A=Y))

  ## add BBAA column and return
  qp.df.all %>%
    select(X, Y, Z, A, BABA) %>%
    transform(Y=Z, Z=Y, BBAA=BABA) %>%
    select(-BABA) %>%
    group_by(X,Y,Z,A) %>% summarize(BBAA=unique(BBAA)) %>%
    inner_join(x=qp.df.all)
}

sort.pops.bbaa <- function(df, x, y, a){
  c(x, y,
    df %>%
      filter(BBAA > BABA, BBAA > ABBA, X == x, Y == y, A == a) %>%
      with(structure(BBAA, names=Z)) %>%
      sort() %>% names(),
    a)
}
