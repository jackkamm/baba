import pytest
import baba
import os
import os.path as osp
import autograd.numpy as np
import pandas as pd

def test_quartet_decomp_df():
    path = osp.join(osp.dirname(osp.abspath(__file__)), "inferred_components.txt")
    qd = baba.quartet_decomposition.from_dataframe(path)
    qd_df = qd.data_frame()
    qd_df2 = pd.read_table(path)
    assert qd_df.equals(qd_df2)


def test_baba_from_qpDstat():
    path = osp.join(osp.dirname(osp.abspath(__file__)), "../../data/newhumori_18pops.qpDstats.output")
    #path = "../data/newhumori_18pops.qpDstats.output"
    with open(path) as f:
        baba_stats = baba.baba.from_qpDstat(f)

    path2 = osp.join(osp.dirname(osp.abspath(__file__)), "../../data/scratch/newhumori_18pops/all_quartets_df.txt")
    #path2 = "../data/scratch/newhumori_18pops/all_quartets_df.txt"
    df = pd.read_table(path2, sep=" ")
    baba_stats2 = baba.baba.from_dataframe(df)

    assert np.all(baba_stats.baba == baba_stats2.baba)
    assert np.all(baba_stats.z_score == baba_stats2.z_score)
    assert np.all(baba_stats.n_snps == baba_stats2.n_snps)
