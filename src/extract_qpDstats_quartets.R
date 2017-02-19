source("src/abba/abba.R")
args <- commandArgs(trailingOnly=TRUE)

args[1] %>%
  extract_qpDstats_quartets() %>%
  write.table(args[2], row.names=F, quote=F)
