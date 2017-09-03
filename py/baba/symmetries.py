import autograd.numpy as np
import itertools as it


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
