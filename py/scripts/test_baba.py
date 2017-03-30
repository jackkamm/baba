import baba
import pytest


def test_get_permutations():
    bbaa_symmetries = [tuple(range(4))]
    bbaa_symmetries += [(x, w, y, z) for w, x, y, z in bbaa_symmetries]
    bbaa_symmetries += [(w, x, z, y) for w, x, y, z in bbaa_symmetries]
    bbaa_symmetries += [(y, z, w, x) for w, x, y, z in bbaa_symmetries]

    assert set(bbaa_symmetries) == set(baba.get_permutations("BBAA", "BBAA"))


def test_get_permutations2():
    symmetries = set(baba.get_permutations("ABBA", "BABA"))
    assert len(symmetries) == 8
    for s in symmetries:
        assert "".join(["ABBA"[i] for i in s]) in ("BABA", "ABAB")


def test_get_permutations3():
    symmetries = [(0, 1, 2, 3)]
    symmetries += [(y, z, w, x) for w, x, y, z in symmetries]
    symmetries += [(x, w, z, y) for w, x, y, z in symmetries]

    assert set(symmetries) == set(baba.get_permutations("BABA", "BABA")) & set(
        baba.get_permutations("ABBA", "ABBA"))
