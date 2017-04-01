import pytest
import baba
import os
import os.path as osp
import autograd.numpy as np
import pandas as pd

def test_quartet_decomp_df():
    try:
        path = osp.join(osp.dirname(osp.abspath(__file__)), "inferred_components.txt")
    except NameError:
        path = "inferred_components.txt"
    qd = baba.quartet_decomposition.from_dataframe(path)
    qd_df = qd.data_frame()
    qd_df2 = pd.read_table(path)
    assert qd_df.equals(qd_df2)
