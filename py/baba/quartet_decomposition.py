import itertools as it
import autograd
import autograd.numpy as np
import scipy
from cached_property import cached_property
import pandas as pd
import os
import scipy.stats
from .empirical_quartets import baba
import json
from collections import OrderedDict


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
                          inferred_components_file,
                          seed=None):
    if seed:
        np.random.seed(int(seed))
    n_components = int(n_components)
    l1_penalty = float(l1_penalty)

    #df = pd.read_table(in_file, sep=None)
    #ab = baba.from_dataframe(df)
    with open(in_file) as f:
        ab = baba.from_qpDstat(f)

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
