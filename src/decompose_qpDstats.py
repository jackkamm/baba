from baba import quartet_tensor

infile = "data/scratch/newhumori_18pops.qpDstats.dataframe.with_BBAA.txt"

with open(infile) as f:
    qt = quartet_tensor(f)
