
import itertools as it
import autograd.numpy as np
import scipy
from cached_property import cached_property


class quartet_decomposition(object):
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

    def dump(self, f):
        print("Component", "Weight", "Mode", "Population",
              "Value", sep="\t", file=f)
        for index, value in np.ndenumerate(self.components):
            mode, component, population = index
            print(component, self.weights[component],
                  mode, self.populations[population], value,
                  sep="\t", file=f)

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

    def make_baba_abba_objective(self, l1_penalty):
        z_scores = np.array(self.z_score)
        # TODO: only keep bbaa > baba > abba
        z_scores[(self.bbaa < self.abba) | (self.bbaa < self.baba)] = 0

        symmetries = [[0, 1, 2, 3]]
        symmetries += [[y, z, w, x] for w, x, y, z in symmetries]
        symmetries += [[x, w, z, y] for w, x, y, z in symmetries]
        assert all(np.all(z_scores == np.transpose(z_scores, s))
                   for s in symmetries)

        antisymmetries = [[x, w, y, z] for w, x, y, z in symmetries]
        assert all(np.all(z_scores == -np.transpose(z_scores, s))
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
            return (np.sum((symmetrized_arr - z_scores)**2)
                    + np.sum(l1_penalty * components *
                            (1 +
                             l1_penalty * (components[[1, 0, 3, 2], :, :] +
                                           components[[2, 3, 0, 1], :, :] +
                                           components[[3, 2, 3, 0], :, :]))))
        return objective
