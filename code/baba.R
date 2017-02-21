library(dplyr)
library(ggplot2)

sort.pops.bbaa <- function(quartets.df, x, y, a){
  c(x, y,
    quartets.df %>%
      filter(BBAA > BABA, BBAA > ABBA, X == x, Y == y, A == a) %>%
      with(structure(BBAA, names=Z)) %>%
      sort() %>% names(),
    a)
}
