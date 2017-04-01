import itertools as it
import autograd.numpy as np
import scipy
from cached_property import cached_property
import json
import subprocess
import pandas as pd
import autograd
import os
import scipy.stats
from collections import OrderedDict


class quartet_decomposition(object):
    @classmethod
    def from_dataframe(cls, df):
        if isinstance(df, str):
            df = pd.read_table(df)
        populations = sorted(set(df["Population"]))

        def get_idxs(column):
            sorted_values = sorted(set(column))
            idx_dict = {val: idx for idx, val in enumerate(
                sorted_values)}
            return np.array([
                idx_dict[val] for val in column
            ], dtype=int)

        axes = ["Mode", "Component", "Population"]
        size = [len(set(df[a])) for a in axes]
        components = np.zeros(size)

        for i, j, k, v in zip(*([get_idxs(df[col]) for col in axes] + [df["Value"]])):
            components[i, j, k] = v

        weights = [w for c, w in sorted(set(zip(df["Component"], df["Weight"])))]
        return cls(populations, components, weights = weights)

    def __eq__(self, other):
        return self.populations == other.populations and np.all(self.components == other.components) and np.all(self.weights == other.weights)

    def __init__(self, populations, components,
                 weights=None):
        """
        components[i,j,k], i=mode, j=component, k=population
        """
        self.populations = populations

        self.components = np.array(components)
        if not len(self.components.shape) == 3:
            raise ValueError(
                "components has wrong shape {}".format(
                    self.components.shape))

        if weights is None:
            weights = np.ones(self.components.shape[1])
        self.weights = weights
        if len(self.weights) != self.components.shape[1]:
            raise ValueError(
                "{} {} != {} {}".format(
                    "n_components", self.components.shape[1],
                    "n_weights", len(self.weights)))

    @cached_property
    def array(self):
        return np.einsum("i,ia,ib,ic,id->abcd", self.weights,
                         *self.components)

    @cached_property
    def flattened_components(self):
        return np.reshape(self.components, -1)

    def data_frame(self):
        df = []
        for index, value in np.ndenumerate(self.components):
            mode, component, population = index
            df.append((component + 1,
                       self.weights[component],
                       "Pop{}".format(mode + 1),
                       self.populations[population],
                       value))
        return pd.DataFrame(df, columns = (
            "Component", "Weight", "Mode",
            "Population", "Value"))

    def dump(self, f):
        self.data_frame().to_csv(f, sep="\t", index=False)

    def reweight(self, norm_order):
        """
        Reweights every component to have norm 1,
        and then sorts by the component weights
        """
        norms = np.linalg.norm(self.components,
                               ord=norm_order, axis=2)
        all0 = norms == 0
        assert np.all(np.max(np.abs(self.components),
                             axis=2)[all0] == 0)
        norms[all0] = 1

        components = np.einsum("ijk,ij->ijk",
                               self.components,
                               1. / norms)
        norms[all0] = 0
        weights = self.weights * np.prod(norms, axis=0)

        sort_components = np.argsort(weights)[::-1]
        weights = weights[sort_components]
        components = components[:, sort_components, :]
        return quartet_decomposition(self.populations,
                                     components,
                                     weights=weights)

    def optimize(self, objective,
                 jac_maker=None,
                 hess_maker=None,
                 hessp_maker=None,
                 bounds=None, **kwargs):
        kwargs = dict(kwargs)

        def fun(flattened_components):
            return objective(
                quartet_decomposition(
                    self.populations,
                    np.reshape(flattened_components,
                               self.components.shape),
                    weights=self.weights))
        for key, val_maker in (
                ("jac", jac_maker),
                ("hess", hess_maker),
                ("hessp", hessp_maker)):
            if val_maker:
                kwargs[key] = val_maker(fun)

        if bounds:
            bounds = np.array(bounds)
            if bounds.shape == (2,):
                bounds = [tuple(bounds)] * len(
                    self.flattened_components)
            kwargs["bounds"] = bounds

        res = scipy.optimize.minimize(
            fun, self.flattened_components, **kwargs)

        res.quartet_decomposition = quartet_decomposition(
            self.populations,
            np.reshape(res.x, self.components.shape))

        return res


def build_tensor(shape, idxs, vals):
    ret = np.zeros(shape)
    ret[idxs[:, 0], idxs[:, 1], idxs[:, 2], idxs[:, 3]] = vals
    return ret


class baba(object):
    @classmethod
    def from_dataframe(cls, df):
        pops = set(df["Pop1"])
        assert all(set(df["Pop2"]) == pops for k in "Pop2 Pop3 Pop4".split())
        assert df.shape[0] == np.prod(len(pops) - np.arange(4))
        pops = sorted(pops)
        pop2idx = {k: v for v, k in enumerate(pops)}
        idxs = np.array([[pop2idx[pop] for pop in df[k]]
                         for k in "Pop1 Pop2 Pop3 Pop4".split()]).T
        assert np.all(idxs.shape == np.array([np.prod(
            len(pops) - np.arange(4)), 4]))
        shape = [len(pops)] * 4
        ret = cls(pops, *[build_tensor(shape, idxs, df[k])
                          for k in "BABA Z.score n.snps".split()])

        assert np.all(build_tensor(shape, idxs, df["ABBA"]) == ret.abba)
        assert np.all(build_tensor(shape, idxs, df["BBAA"]) == ret.bbaa)
        return ret

    def __init__(self, populations, baba, z_score, n_snps):
        for arr in (baba, z_score, n_snps):
            if np.any(arr.shape != np.array([len(populations)] * 4)):
                raise ValueError("array has wrong dimension")

        symmetries = [[0, 1, 2, 3]]
        symmetries += [[x, w, z, y] for w, x, y, z in symmetries]
        if any(np.any(z_score != np.transpose(z_score, s))
               for s in symmetries):
            raise ValueError("Z-scores not symmetric")

        symmetries += [[y, x, w, z] for w, x, y, z in symmetries]
        symmetries += [[w, z, y, x] for w, x, y, z in symmetries]
        assert len(symmetries) == 8
        if any(np.any(baba != np.transpose(baba, s))
               for s in symmetries):
            raise ValueError("BABA not symmetric")

        if any(np.any(n_snps != np.transpose(n_snps, s))
               for s in it.permutations(range(4))):
            raise ValueError("n_snps not symmetric")

        self.populations = populations
        self.baba = baba
        self.n_snps = n_snps
        self.z_score = z_score

        if np.any((self.z_score < 0) & (self.baba > self.abba)) or np.any(
                (self.z_score > 0) & (self.baba < self.abba)):
            raise ValueError("z_score should have same sign as baba-abba")

    @cached_property
    def abba(self):
        return np.transpose(self.baba, [1, 0, 2, 3])

    @cached_property
    def bbaa(self):
        return np.transpose(self.baba, [0, 2, 1, 3])

    def make_z_baba_abba_objective(self, l1_penalty):
        z_scores = np.array(self.z_score)
        # only keep bbaa > baba > abba
        z_scores[(self.bbaa < self.baba) | (self.baba < self.abba)] = 0

        symmetries = set(get_permutations("BABA", "BABA")) & set(
            get_permutations("ABBA", "ABBA"))
        assert len(symmetries) == 4

        def objective(baba_decomp):
            assert baba_decomp.populations == self.populations
            arr = baba_decomp.array
            components = baba_decomp.components
            symmetrized_arr = 0
            for s in symmetries:
                symmetrized_arr = symmetrized_arr + np.transpose(arr, s)
            return (np.sum((symmetrized_arr - z_scores)**2)
                    + np.sum(l1_penalty * components))

        return objective

    # def make_baba_abba_objective(self, l1_penalty):
    #     symmetries = set(get_permutations("BBAA", "BBAA")) & set(
    #         get_permutations("ABBA", "BABA"))
    #     assert len(symmetries) == 4

    #     antisymmetries = set([])
    #     for non_bbaa in ("ABBA", "BABA"):
    #         antisymmetries |= (set(get_permutations("BBAA", non_bbaa)) & set(
    #             get_permutations(non_bbaa, "BBAA")))

    #     # convert z_score from BABA-ABBA to BBAA-ABBA
    #     z_score = np.transpose(self.z_score, [0, 2, 1, 3])

    #     repeated_idx = np.zeros(self.z_score.shape)
    #     for i, j, k, l in it.product(*[range(n) for n in repeated_idx.shape]):
    #         if len(set([i, j, k, l])) != 4:
    #             repeated_idx[i, j, k, l] = 1
    #     assert np.all(repeated_idx * z_score == 0)

    #     def objective(decomp):
    #         bbaa_abba = decomp.array
    #         arr = bbaa_abba
    #         for s in symmetries:
    #             arr = arr + np.transpose(arr, s)
    #         for s in antisymmetries:
    #             arr = arr - np.transpose(arr, s)

    #         return (np.sum((arr - z_score)**2)
    #                 # add penalty for repeated populations
    #                 + np.sum(repeated_idx * decomp.array**2)
    #                 + np.sum(l1_penalty * decomp.components))

    #     return objective


def get_permutations(from_ABs, to_ABs):
    def recode(ABs):
        return np.array([{"A": -1, "B": 1}[c] for c in ABs.upper()], dtype=int)
    from_ABs = recode(from_ABs)
    to_ABs = recode(to_ABs)
    for permutation in it.permutations(range(len(from_ABs))):
        curr = from_ABs[np.array(permutation)]
        if np.all(curr == to_ABs) or np.all(curr == -1 * to_ABs):
            yield permutation



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
