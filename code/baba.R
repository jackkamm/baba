library(dplyr)
library(tidyr)
library(ggplot2)

sort.pops.bbaa <- function(quartets.df, pop1, pop2, pop4){
  c(pop1, pop2,
    quartets.df %>%
      filter(BBAA > BABA, BBAA > ABBA, Pop1 == pop1, Pop2 == pop2, Pop4 == pop4) %>%
      with(structure(BBAA, names=Pop3)) %>%
      sort() %>% names(),
    pop4)
}
