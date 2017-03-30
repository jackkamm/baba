# meant to be run from file directory
# if hanging, try uninstalling with remove.packages("baba") first
library(devtools)
setwd("baba")
build()
document()
install()
