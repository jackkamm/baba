class qpDstats_output(object):
    def __init__(self, instream):
        self.lines = list(instream)

    def write_data_frame(self, outstream):
        for line in self.lines:
            if line.startswith("result:"):
                print(*line.split()[1:], sep="\t",
                    file=outstream)
