library(baba)

args <- commandArgs(trailingOnly=TRUE)
do.call(args[1], as.list(args[-1]))
