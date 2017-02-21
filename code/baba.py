import autograd.numpy as np
import pandas as pd
import scipy
import itertools as it
from cached_property import cached_property
import autograd

class baba_decomposition(object):
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
                               ord = norm_order, axis = 2)
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
        return baba_decomposition(self.populations,
                                  components,
                                  weights = weights)

    def optimize(self, objective,
                 jac_maker = None,
                 hess_maker = None,
                 hessp_maker = None,
                 bounds = None, **kwargs):
        kwargs = dict(kwargs)
        def fun(flattened_components):
            return objective(
                baba_decomposition(
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

        res.baba_decomposition = baba_decomposition(
            self.populations,
            np.reshape(res.x, self.components.shape))

        return res
