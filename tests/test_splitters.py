import numpy as np
import pytest

from utilities import splitters


def _check_disjoint(splits):
    train = set(splits["train"])
    val = set(splits.get("val", []))
    test = set(splits.get("test", []))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_random_split_indices_sizes():
    splits = splitters.random_split_indices(
        n_samples=100,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify=None,
    )
    _check_disjoint(splits)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 100


def test_random_split_indices_stratified():
    y = np.array([0] * 50 + [1] * 50)
    splits = splitters.random_split_indices(
        n_samples=100,
        test_size=0.2,
        val_size=0.1,
        random_state=0,
        stratify=y,
    )
    _check_disjoint(splits)


@pytest.mark.skipif(splitters.Chem is None, reason="RDKit not installed")
def test_scaffold_split_indices():
    smiles = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)O",
        "CC(=O)N",
        "CCS",
        "CCCl",
        "CCBr",
    ]
    splits = splitters.scaffold_split_indices(
        smiles_list=smiles,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )
    _check_disjoint(splits)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(smiles)
