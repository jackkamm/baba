import sys
import subprocess
import pandas as pd
import autograd
import scipy
import autograd.numpy as np
import os
import scipy.stats
import json
from collections import OrderedDict
from baba import quartet_decomposition, baba
import subprocess


def make_z_baba_abba(in_file, sorted_pops_file, n_components, l1_penalty,
                     outdir, seed=None):
    inferred_components_file = os.path.join(outdir, "inferred_components.txt")
    decompose_z_baba_abba(in_file, n_components, l1_penalty,
                          os.path.join(outdir, "optimization_result.json"),
                          inferred_components_file)
    subprocess.check_call(["Rscript", "code/run_baba.R", "plot.baba.matrices",
                           inferred_components_file, sorted_pops_file,
                           os.path.join(outdir, "z_baba_abba_squares."),
                           "Pop1", "Pop3", "Pop2", "Pop4"])
    subprocess.check_call(["Rscript", "code/run_baba.R", "plot.baba.vectors",
                           inferred_components_file, sorted_pops_file,
                           os.path.join(outdir, "z_baba_abba_vectors.pdf")])


# in_file = "../data/scratch/newhumori_18pops/all_quartets_df.txt"
# n_components = 5
# n_components = 10
# l1_penalty = 100
# l1_penalty = 10
# outdir = "../data/scratch/newhumori_18pops/decomposition_5_100"
# optimization_result_file = os.path.join(outdir, "optimization_result.json")
# inferred_components_file = os.path.join(outdir, "inferred_components.txt")
def decompose_z_baba_abba(in_file, n_components, l1_penalty,
                          optimization_result_file,
                          inferred_components_file, seed=None):
    if seed:
        np.random.seed(int(seed))
    n_components = int(n_components)
    l1_penalty = float(l1_penalty)

    df = pd.read_table(in_file, sep=None)
    ab = baba.from_dataframe(df)

    components_size = (4, n_components, len(ab.populations))
    random_baba = quartet_decomposition(
        ab.populations, scipy.stats.uniform.rvs(size=components_size))
    res = random_baba.optimize(ab.make_z_baba_abba_objective(l1_penalty),
                               jac_maker=autograd.grad,
                               bounds=[0, None])

    inferred = res.quartet_decomposition
    inferred = inferred.reweight(norm_order=float('inf'))

    with open(optimization_result_file, "w") as f:
        json.dump(OrderedDict(
            [(k, str(res[k])) for k in ["success", "status", "message"]] +
            [(k, int(res[k])) for k in ["nfev", "nit"]] +
            [(k, float(res[k])) for k in ["fun"]] +
            [(k, list(res[k])) for k in ["x", "jac"]]
        ), f, indent=True)

    with open(inferred_components_file, "w") as f:
        inferred.dump(f)


def clean_qpDstats_output():
    for line in sys.stdin:
        if line.startswith("result:"):
            print(*line.split()[1:], sep="\t")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
