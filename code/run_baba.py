import sys
import pandas as pd
import autograd
import autograd.numpy as np
import scipy
import scipy.stats
import json
from collections import OrderedDict


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

    components_size = (4, n_components, len(pops))
    init_components = scipy.stats.uniform.rvs(size=components_size)
    init_x = np.reshape(init_components, -1)
    assert np.all(np.reshape(init_x, components_size) == init_components)

    symmetries = [[0, 1, 2, 3]]
    symmetries += [[z, a, x, y] for x, y, z, a in symmetries]
    symmetries += [[y, x, a, z] for x, y, z, a in symmetries]
    assert all(np.all(z_score_arr == np.transpose(z_score_arr, s))
               for s in symmetries)

    antisymmetries = [[y, x, z, a] for x, y, z, a in symmetries]
    assert all(np.all(z_score_arr == -np.transpose(z_score_arr, s))
               for s in antisymmetries)

    def objective(x):
        components = np.reshape(x, components_size)
        arr = np.einsum("ia,ib,ic,id->abcd",
                        *(components[i, :, :] for i in range(4)))
        symmetrized_arr = 0
        for s in symmetries:
            symmetrized_arr = symmetrized_arr + np.transpose(arr, s)
        for s in antisymmetries:
            symmetrized_arr = symmetrized_arr - np.transpose(arr, s)
        return (np.sum((symmetrized_arr - z_score_arr)**2)
                + np.sum(l1_penalty * x))

    res = scipy.optimize.minimize(
        objective, init_x, jac=autograd.grad(objective),
        bounds=[(0, None)] * len(init_x))

    with open(optimization_result_file, "w") as f:
        json.dump(OrderedDict(
            [(k, str(res[k])) for k in ["success", "status", "message"]] +
            [(k, int(res[k])) for k in ["nfev", "nit"]] +
            [(k, float(res[k])) for k in ["fun"]] +
            [(k, list(res[k])) for k in ["x", "jac"]]
        ), f, indent=True)

    inferred_components = np.reshape(res.x, components_size)
    with open(inferred_components_file, "w") as f:
        print("Component", "Subcomponent", "Population", "Value",
              sep="\t", file=f)
        for i, subcomponent in enumerate(inferred_components):
            for j, row in enumerate(subcomponent):
                for k, value in enumerate(row):
                    print(j+1, "XYZA"[i], pops[k], value, sep="\t", file=f)


def clean_qpDstats_output():
    for line in sys.stdin:
        if line.startswith("result:"):
            print(*line.split()[1:], sep="\t")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
