import itertools as it
import autograd.numpy as np
import scipy
from cached_property import cached_property
import subprocess
import pandas as pd
import autograd
import os
import scipy.stats
import io
import re
import json
import logging
import collections as co
from .quartet_decomposition import quartet_decomposition


def build_tensor(shape, idxs, vals):
    ret = np.zeros(shape)
    ret[idxs[:, 0], idxs[:, 1], idxs[:, 2], idxs[:, 3]] = vals
    return ret


class quartet_stats(object):
    @classmethod
    def from_qpDstat(cls, qpDstat_out):
        lines = []
        line_re = re.compile(r"result:(.+)")
        for line in qpDstat_out:
            matched = line_re.match(line)
            if matched:
                lines.append(
                    matched.group(1).split())

        if len(lines[0]) == 10:
            columns = ("Pop1","Pop2","Pop3","Pop4", "ABBA_BABA", "Z.score", "is_best", "BABA", "ABBA", "n.snps")
        else:
            columns = ("Pop1","Pop2","Pop3","Pop4", "ABBA_BABA", "Z.score", "BABA", "ABBA", "n.snps")

        stringio = io.StringIO()
        for line in lines:
            print(*line, file=stringio, sep="\t")
        stringio.seek(0)

        df = pd.read_table(stringio, names=columns)[[
            "Pop1", "Pop2", "Pop3", "Pop4",
            "Z.score", "BABA", "ABBA", "n.snps"
        ]]

        populations = sorted(set(
            p for i in range(1, 5)
            for p in df[f"Pop{i}"]))

        pop2idx = {pop: idx
                for idx, pop in enumerate(populations)}

        idxs = np.array([[pop2idx[pop] for pop in df[f"Pop{i}"]] for i in range(1, 5)], dtype=int).T

        def get_raw_array(vals):
            ret = np.zeros((len(populations),) * 4)
            ret[idxs[:,0], idxs[:,1], idxs[:,2], idxs[:,3]] = vals
            return ret

        equiv_perms = get_permutations(
            "BABA", "BABA") & get_permutations("ABBA", "ABBA")
        raw_z_score = get_raw_array(df["Z.score"])
        z_score = get_symmetrized_array(
            get_raw_array(df["Z.score"]), equiv_perms
        )
        assert np.all((raw_z_score == 0) | (raw_z_score == z_score))
        z_score2 = np.transpose(-z_score, (1, 0, 2, 3))
        assert np.all((z_score == 0) | (z_score2 == 0))
        z_score += z_score2

        raw_n_snps = get_raw_array(df["n.snps"])
        n_snps = get_symmetrized_array(
            raw_n_snps,
            list(it.permutations(range(4))),
            accum_fun=np.maximum
        )
        assert np.all((raw_n_snps == 0) | (raw_n_snps == n_snps))

        raw_baba1 = get_raw_array(df["BABA"])
        raw_baba2 = np.transpose(get_raw_array(df["ABBA"]),
                                (1, 0, 2, 3))

        baba1 = get_symmetrized_array(
            raw_baba1,
            get_permutations("BABA", "BABA"),
            accum_fun=np.maximum)
        baba2 = get_symmetrized_array(
            raw_baba2,
            get_permutations("BABA", "BABA"),
            accum_fun=np.maximum)
        assert np.all((baba1 == 0) | (baba2 == 0) | (baba1 == baba2))
        baba = np.maximum(baba1, baba2)
        assert np.all((baba1 == 0) | (baba1 == baba))
        assert np.all((baba2 == 0) | (baba2 == baba))

        abba = np.transpose(baba, (1,0,2,3))
        raw_abba = get_raw_array(df["ABBA"])
        assert np.all((raw_abba == 0) | (raw_abba == abba))

        n_pops = len(populations)
        n_pop_perms = n_pops * (n_pops-1) * (n_pops-2) * (n_pops - 3)
        assert np.sum(n_snps != 0) == n_pop_perms
        assert np.sum(baba != 0) == n_pop_perms
        assert np.sum(z_score != 0) == n_pop_perms - len(get_permutations("BBAA", "BBAA")) * sum(df["Z.score"] == 0)

        return cls(populations, baba,
                   z_score, n_snps)

    @classmethod
    def from_dataframe(cls, df):
        """DEPRACATED"""
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

        equiv_perms = get_permutations("ABBA", "ABBA") & get_permutations("BABA", "BABA")
        opposite_perms = get_permutations("ABBA", "BABA") & get_permutations("BABA", "ABBA")

        if not is_symmetric(z_score, equiv_perms) or not is_symmetric(z_score, opposite_perms, antisymm=True):
            raise ValueError("Z-scores not symmetric")

        if not is_symmetric(n_snps,
                            list(it.permutations(range(4)))):
            raise ValueError("n_snps not symmetric")

        if not is_symmetric(baba, get_permutations("BABA",
                                                   "BABA")):
            raise ValueError("BABA not symmetric")

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
            symmetrized_arr = get_symmetrized_array(
                arr, symmetries)

            return (np.sum((symmetrized_arr - z_scores)**2)
                    + np.sum(l1_penalty * components))

        return objective

    def decompose_z_scores(self, n_components, l1_penalty, start_decomp = None):
        if start_decomp is None:
            components_size = (4, n_components, len(self.populations))
            start_decomp = quartet_decomposition(
                self.populations, scipy.stats.uniform.rvs(size=components_size))

        def fit(start, l1):
            ret =  start.optimize(self.make_z_baba_abba_objective(l1),
                                  jac_maker=autograd.grad,
                                  bounds=[0, None])
            ret.fit_info["sparsity"] = l1
            ret.fit_info["l2_err"] = self.make_z_baba_abba_objective(0)(ret)
            ret.fit_info.move_to_end("l2_err", last=False)
            ret.fit_info.move_to_end("sparsity", last=False)
            return ret

        try:
            l1_penalty_list = list(l1_penalty)
        except TypeError:
            return fit(start_decomp, l1_penalty)

        prev_decomp = start_decomp
        for l1 in l1_penalty_list:
            logging.info(f"Fitting decomposition at sparsity = {l1}")
            prev_decomp = fit(prev_decomp, l1)
            yield prev_decomp


def decompose_z_baba_abba(in_file, n_components, l1_penalty,
                          optimization_result_file,
                          inferred_components_file,
                          seed=None):
    """DEPRACATED"""
    if seed:
        np.random.seed(int(seed))
    n_components = int(n_components)
    l1_penalty = float(l1_penalty)

    with open(in_file) as f:
        ab = quartet_stats.from_qpDstat(f)

    decomposition = ab.decompose_z_scores(n_components, l1_penalty)

    with open(optimization_result_file, "w") as f:
        json.dump(decomposition.fit_info, f, indent=True)

    with open(inferred_components_file, "w") as f:
        decomposition.dump(f)


def get_permutations(from_ABs, to_ABs):
    return set(_get_permutations(from_ABs, to_ABs))

def _get_permutations(from_ABs, to_ABs):
    def recode(ABs):
        return np.array([{"A": -1, "B": 1}[c] for c in ABs.upper()], dtype=int)
    from_ABs = recode(from_ABs)
    to_ABs = recode(to_ABs)
    for permutation in it.permutations(range(len(from_ABs))):
        curr = from_ABs[np.array(permutation)]
        if np.all(curr == to_ABs) or np.all(curr == -1 * to_ABs):
            yield permutation


def get_symmetrized_array(arr, symmetries,
                          accum_fun=lambda x,y: x+y):
    symmetrized_arr = np.zeros(arr.shape)
    for s in symmetries:
         symmetrized_arr = accum_fun(
             symmetrized_arr, np.transpose(arr, s))
    return symmetrized_arr

def is_symmetric(arr, symmetries, antisymm = False):
    arr2 = get_symmetrized_array(arr, symmetries) / len(symmetries)
    if antisymm:
        arr2 = -1 * arr2
    return np.allclose(arr, arr2)
