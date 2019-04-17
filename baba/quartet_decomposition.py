import autograd
import autograd.numpy as np
import scipy
from cached_property import cached_property
import pandas as pd
import scipy.stats
import collections as co
import logging
from .symmetries import get_symmetrized_array


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

        axes = ["Leaf", "Component", "Population"]
        size = [len(set(df[a])) for a in axes]
        components = np.zeros(size)

        for i, j, k, v in zip(*([get_idxs(df[col]) for col in axes] + [df["PopulationWeight"]])):
            components[i, j, k] = v

        weights = [w for c, w in sorted(set(zip(df["Component"], df["ComponentWeight"])))]
        return cls(populations, components, weights = weights)

    @classmethod
    def random_uniform(cls, populations, n_components):
            components_size = (4, n_components, len(populations))
            return cls(populations,
                       scipy.stats.uniform.rvs(size=components_size))

    def __eq__(self, other):
        return self.populations == other.populations and np.all(self.components == other.components) and np.all(self.weights == other.weights)

    def __init__(self, populations, components,
                 weights=None, fit_info=None):
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
        self.weights = np.array(weights)
        if len(self.weights) != self.components.shape[1]:
            raise ValueError(
                "{} {} != {} {}".format(
                    "n_components", self.components.shape[1],
                    "n_weights", len(self.weights)))

        self.fit_info = fit_info

    @cached_property
    def array(self):
        return np.einsum("i,ia,ib,ic,id->abcd", self.weights,
                         *self.components)

    @cached_property
    def flattened_components(self):
        return np.reshape(self.components, -1)

    def data_frame(self):
        return self.reweight()._data_frame()
    def _data_frame(self):
        df = []
        for index, value in np.ndenumerate(self.components):
            mode, component, population = index
            df.append((component + 1,
                       self.weights[component],
                       "Pop{}".format(mode + 1),
                       self.populations[population],
                       value))
        return pd.DataFrame(df, columns = (
            "Component", "ComponentWeight", "Leaf",
            "Population", "PopulationWeight"))

    def dump(self, f):
        self.data_frame().to_csv(f, sep="\t", index=False)

    def reweight(self, norm_order=float('inf'), eps=1e-8):
        """
        Reweights every component to have norm 1,
        and then sorts by the component weights
        """
        components = np.array(self.components)
        components[np.isclose(components, 0, atol=eps)] = 0
        norms = np.linalg.norm(components,
                               ord=norm_order, axis=2)
        all0 = norms == 0
        assert np.all(np.max(np.abs(components),
                             axis=2)[all0] == 0)
        norms[all0] = 1

        components = np.einsum("ijk,ij->ijk",
                               components,
                               1. / norms)
        norms[all0] = 0
        weights = self.weights * np.prod(norms, axis=0)

        sort_components = np.argsort(weights)[::-1]
        weights = weights[sort_components]
        components = components[:, sort_components, :]
        return quartet_decomposition(self.populations,
                                     components,
                                     weights=weights,
                                     fit_info=self.fit_info)

    def reset_weights(self):
        """
        Returns equivalent decomposition whose weights are all 1
        """
        return quartet_decomposition(
            self.populations,
            np.einsum("ijk,j->ijk",
                      self.components,
                      self.weights**(.25)),
            fit_info = self.fit_info
        )

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
                               self.components.shape)))
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
            fun, self.reset_weights().flattened_components, **kwargs)

        # store fit info in a format that is easily converted to json
        fit_info = co.OrderedDict(
            [(k, str(res[k])) for k in ["success", "status", "message"]] +
            [(k, int(res[k])) for k in ["nfev", "nit"]] +
            [(k, float(res[k])) for k in ["fun"]] +
            [(k, list(res[k])) for k in ["x", "jac"]]
        )

        return quartet_decomposition(
            self.populations,
            np.reshape(res.x, self.components.shape),
            fit_info = fit_info
        )

    def fit_decomposition(self, arr_to_decompose, symmetries, l1_penalty):
        def make_objective(l1):
            def objective(decomp):
                assert decomp.populations == self.populations
                arr = decomp.array
                components = decomp.components
                symmetrized_arr = get_symmetrized_array(
                    arr, symmetries)

                return (np.sum((symmetrized_arr - arr_to_decompose)**2)
                        + np.sum(l1 * components))
            return objective

        def fit(start, l1):
            ret =  start.optimize(make_objective(l1),
                                  jac_maker=autograd.grad,
                                  bounds=[0, None])
            ret.fit_info["sparsity"] = l1
            ret.fit_info["l2_err"] = make_objective(0)(ret)
            ret.fit_info.move_to_end("l2_err", last=False)
            ret.fit_info.move_to_end("sparsity", last=False)
            return ret

        try:
            l1_penalty_list = list(l1_penalty)
        except TypeError:
            return fit(self, l1_penalty)

        prev_decomp = self
        for l1 in l1_penalty_list:
            logging.info(f"Fitting decomposition at sparsity = {l1}")
            prev_decomp = fit(prev_decomp, l1)
            yield prev_decomp
