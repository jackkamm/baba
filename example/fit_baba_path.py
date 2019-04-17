from baba import quartet_stats, quartet_decomposition
import logging
import autograd.numpy as np
import pandas as pd
import json
import sys
import argparse as ap
import os
import os.path as osp

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
    fit_baba_path("probgen17_qpDstat_results5.txt", 16, .1, 1000, 20, "fit_baba_path_results")

def fit_baba_path(qpDstat_out_f, n_components, l1_start, l1_end, l1_steps, outdir, log_l1_steps=True, start_df=None):
    os.makedirs(outdir)

    with open(qpDstat_out_f) as f:
        qstats = quartet_stats.from_qpDstat(f)

    if log_l1_steps:
        l1_list = l1_start * np.exp(np.log(l1_end / l1_start) * np.arange(l1_steps+1) / l1_steps)
    else:
        l1_list = l1_start + (l1_end - l1_start) * np.arange(l1_steps+1) / l1_steps

    if start_df:
        start_decomp = quartet_decomposition.from_dataframe(start_df)
    else:
        start_decomp = None

    for decomp in qstats.decompose_z_scores(n_components, l1_list, start_decomp = start_decomp):
        df = decomp.data_frame()
        sparsity = decomp.fit_info["sparsity"]
        df["sparsity"] = sparsity
        df["objective"] = decomp.fit_info["fun"]
        df["l2_err"] = decomp.fit_info["l2_err"]
        df["run"] = osp.basename(outdir)
        df.to_csv(f"{outdir}/{sparsity}_decomposition.txt", index=False, sep="\t")
        with open(f"{outdir}/{sparsity}_info.json", "w") as f:
            json.dump(decomp.fit_info, f)


if __name__ == "__main__":
    main()
