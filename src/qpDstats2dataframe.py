#!/usr/bin/env python3
import sys
from abba import qpDstats_output
qpDstats_output(sys.stdin).write_data_frame(None)
