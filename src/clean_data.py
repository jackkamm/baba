class qpDstats_output(object):
    def __init__(self, instream):
        self.lines = list(instream)

    def write_data_frame(self, outstream):
        print(*("X Y Z A BABA-ABBA Zscore " +
                "least.significant BABA ABBA Total").split(),
            sep="\t", file=outstream)
        for line in self.lines:
            if line.startswith("result:"):
                print(*line.split()[1:], sep="\t",
                    file=outstream)

