import sys
import os
import pandas as pd
import autograd
import autograd.numpy as np
import scipy
import scipy.stats
import json
from collections import OrderedDict
from baba import baba_decomposition

# in_file = "../data/scratch/newhumori_18pops/all_quartets_df.txt"
# n_components = 5
# n_components = 10
# l1_penalty = 100
# l1_penalty = 10
# outdir = "../data/scratch/newhumori_18pops/decomposition_5_100"
# optimization_result_file = os.path.join(outdir, "optimization_result.json")
# inferred_components_file = os.path.join(outdir, "inferred_components.txt")
def decompose_qpdstats(in_file, n_components, l1_penalty,
                       optimization_result_file,
                       inferred_components_file):
    n_components = int(n_components)
    l1_penalty = float(l1_penalty)

    df = pd.read_table(in_file, sep=None)
    pops = set(df["X"])
    assert all(set(df[k]) == pops for k in "Y Z A".split())
    assert df.shape[0] == np.prod(len(pops) - np.arange(4))
    pops = sorted(pops)
    pop2idx = {k: v for v, k in enumerate(pops)}

    df_bbaa = df.query("BBAA >= BABA and BBAA >= ABBA")

    x, y, z, a = ([pop2idx[v_i]
                  for v_i in df_bbaa[k]]
                  for k in "XYZA")

    z_score = df_bbaa["Z.score"]
    z_score_arr = np.zeros([len(pops)]*4)
    z_score_arr[x, y, z, a] = z_score

    symmetries = [[0, 1, 2, 3]]
    symmetries += [[z, a, x, y] for x, y, z, a in symmetries]
    symmetries += [[y, x, a, z] for x, y, z, a in symmetries]
    assert all(np.all(z_score_arr == np.transpose(z_score_arr, s))
               for s in symmetries)

    antisymmetries = [[y, x, z, a] for x, y, z, a in symmetries]
    assert all(np.all(z_score_arr == -np.transpose(z_score_arr, s))
               for s in antisymmetries)

    def objective(baba_decomp):
        arr = baba_decomp.array
        components = baba_decomp.components
        symmetrized_arr = 0
        for s in symmetries:
            symmetrized_arr = symmetrized_arr + np.transpose(arr, s)
        ## TODO: zero out antisymmetries!
        ## so their error on diagonal isn't zerod out
        for s in antisymmetries:
            symmetrized_arr = symmetrized_arr - np.transpose(arr, s)
        return (np.sum((symmetrized_arr - z_score_arr)**2)
                + np.sum(l1_penalty * components *
                         (1 +
                          l1_penalty * (components[[1, 0, 3, 2], :, :] +
                                        components[[2, 3, 0, 1], :, :] +
                                        components[[3, 2, 3, 0], :, :]))))

    components_size = (4, n_components, len(pops))
    random_baba = baba_decomposition(pops,
        scipy.stats.uniform.rvs(size=components_size))
    res = random_baba.optimize(objective,
                               jac_maker = autograd.grad,
                               bounds = [0, None])

    inferred = res.baba_decomposition
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
