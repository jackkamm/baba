import pandas as pd
import autograd.numpy as np

class qpDstats_output(object):
    def __init__(self, instream):
        self.lines = list(instream)

    def write_data_frame(self, outstream):
        for line in self.lines:
            if line.startswith("result:"):
                print(*line.split()[1:], sep="\t",
                    file=outstream)

class quartet_tensor(object):
    def __init__(self, instream):
        self.df = pd.read_table(instream, sep=None)

        self.pops = set(self.df["X"])
        assert all(set(qt.df[k]) == self.pops
                   for k in ("Y","Z","A"))
        self.pops = sorted(self.pops)
        self.pop2idx = {k:v for v,k in enumerate(self.pops)}

    def baba_abba_array(self):
        pass


