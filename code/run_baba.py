import sys
import subprocess
import pandas as pd
import autograd
import scipy
import scipy.stats
import json
from collections import OrderedDict
from baba import quartet_decomposition, baba


# in_file = "../data/scratch/newhumori_18pops/all_quartets_df.txt"
# n_components = 5
# n_components = 10
# l1_penalty = 100
# l1_penalty = 10
# outdir = "../data/scratch/newhumori_18pops/decomposition_5_100"
# optimization_result_file = os.path.join(outdir, "optimization_result.json")
# inferred_components_file = os.path.join(outdir, "inferred_components.txt")
def decompose_qpdstats(in_file, pop_order_file,
                       n_components, l1_penalty,
                       optimization_result_file,
                       inferred_components_file,
                       plot_file):
    n_components = int(n_components)
    l1_penalty = float(l1_penalty)

    df = pd.read_table(in_file, sep=None)
    ab = baba.from_dataframe(df)

    components_size = (4, n_components, len(ab.populations))
    random_baba = quartet_decomposition(
        ab.populations, scipy.stats.uniform.rvs(size=components_size))
    res = random_baba.optimize(ab.make_baba_abba_objective(l1_penalty),
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

    subprocess.check_call(["Rscript", "code/run_baba.R", "plot.baba",
                           inferred_components_file, pop_order_file,
                           plot_file])


def clean_qpDstats_output():
    for line in sys.stdin:
        if line.startswith("result:"):
            print(*line.split()[1:], sep="\t")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
