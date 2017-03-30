# meant to be run from file directory
library(devtools)
if (!require("baba")){
	remove.packages("baba")
}
setwd("baba")
build()
document()
install()
