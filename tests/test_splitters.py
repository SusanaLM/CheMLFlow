import sys
import types

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize(
    ("strategy", "expected_method"),
    [
        ("tdc_random", "random"),
        ("tdc_scaffold", "scaffold"),
    ],
)
def test_tdc_split_indices_passes_explicit_method(monkeypatch, strategy, expected_method):
    calls = {"name": None, "method": None}

    class _FakeADME:
        def __init__(self, name):
            calls["name"] = name

        def get_split(self, method="random"):
            calls["method"] = method
            return {
                "train": pd.DataFrame({"Drug": ["CCO"]}),
                "valid": pd.DataFrame({"Drug": ["CCN"]}),
                "test": pd.DataFrame({"Drug": ["CCC"]}),
            }

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.ADME = _FakeADME
    fake_tdc.single_pred = fake_single_pred
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    splits = splitters.tdc_split_indices(
        group="ADME",
        name="Pgp_Broccatelli",
        strategy=strategy,
        curated_smiles=["CCO", "CCN", "CCC"],
    )
    assert calls["name"] == "Pgp_Broccatelli"
    assert calls["method"] == expected_method
    assert splits["train"] == [0]
    assert splits["valid"] == [1]
    assert splits["test"] == [2]


@pytest.mark.parametrize(
    ("strategy", "expected_method", "expected_train", "expected_valid", "expected_test"),
    [
        ("tdc_random", "random", [0], [1], [2]),
        ("tdc_scaffold", "scaffold", [2], [1], [0]),
    ],
)
def test_tdc_split_indices_supports_nested_strategy_payload(
    monkeypatch,
    strategy,
    expected_method,
    expected_train,
    expected_valid,
    expected_test,
):
    calls = {"name": None, "method": None}

    class _FakeADME:
        def __init__(self, name):
            calls["name"] = name

        def get_split(self, method="random"):
            calls["method"] = method
            return {
                "random": {
                    "train": pd.DataFrame({"Drug": ["CCO"]}),
                    "valid": pd.DataFrame({"Drug": ["CCN"]}),
                    "test": pd.DataFrame({"Drug": ["CCC"]}),
                },
                "scaffold": {
                    "train": pd.DataFrame({"Drug": ["CCC"]}),
                    "valid": pd.DataFrame({"Drug": ["CCN"]}),
                    "test": pd.DataFrame({"Drug": ["CCO"]}),
                },
            }

    fake_tdc = types.ModuleType("tdc")
    fake_single_pred = types.ModuleType("tdc.single_pred")
    fake_single_pred.ADME = _FakeADME
    fake_tdc.single_pred = fake_single_pred
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)
    monkeypatch.setitem(sys.modules, "tdc.single_pred", fake_single_pred)

    splits = splitters.tdc_split_indices(
        group="ADME",
        name="Pgp_Broccatelli",
        strategy=strategy,
        curated_smiles=["CCO", "CCN", "CCC"],
    )
    assert calls["name"] == "Pgp_Broccatelli"
    assert calls["method"] == expected_method
    assert splits["train"] == expected_train
    assert splits["valid"] == expected_valid
    assert splits["test"] == expected_test
