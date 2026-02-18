import json
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    MurckoScaffold = None


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _scaffold_smiles(smiles: str) -> Optional[str]:
    if Chem is None or MurckoScaffold is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _find_smiles_column(columns: Iterable[str]) -> Optional[str]:
    candidates = ["canonical_smiles", "smiles", "SMILES", "Smiles", "Drug", "drug"]
    for name in candidates:
        if name in columns:
            return name
    return None


def _map_split_indices(
    split_smiles: Iterable[str],
    canonical_to_indices: Dict[str, List[int]],
) -> List[int]:
    indices: List[int] = []
    missing = 0
    for smi in split_smiles:
        canonical = _canonicalize_smiles(str(smi))
        if canonical is None:
            missing += 1
            continue
        bucket = canonical_to_indices.get(canonical)
        if not bucket:
            missing += 1
            continue
        indices.append(bucket.pop(0))
    if missing:
        logging.warning("Split mapping skipped %s rows that were not found in curated data.", missing)
    return indices


def _build_canonical_index(smiles_list: Iterable[str]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_list):
        canonical = _canonicalize_smiles(str(smi))
        if canonical is None:
            continue
        index.setdefault(canonical, []).append(i)
    return index


def random_split_indices(
    n_samples: int,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: Optional[np.ndarray] = None,
) -> Dict[str, List[int]]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1); got {test_size!r}")
    if not 0 <= val_size < 1:
        raise ValueError(f"val_size must be in [0, 1); got {val_size!r}")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    indices = np.arange(n_samples)
    if stratify is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_pos, test_pos = next(splitter.split(indices, stratify))
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_pos, test_pos = next(splitter.split(indices))
    train_idx = indices[train_pos]
    test_idx = indices[test_pos]
    if val_size > 0:
        val_fraction = val_size / (1 - test_size)
        if not 0 < val_fraction < 1:
            raise ValueError(
                f"Derived val_fraction={val_fraction!r} from val_size/test_size is invalid."
            )
        if stratify is not None:
            strat_train = stratify[train_idx]
            splitter_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=random_state,
            )
            train_pos2, val_pos = next(splitter_val.split(train_idx, strat_train))
        else:
            splitter_val = ShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=random_state,
            )
            train_pos2, val_pos = next(splitter_val.split(train_idx))
        val_idx = train_idx[val_pos]
        train_idx = train_idx[train_pos2]
    else:
        val_idx = np.array([], dtype=int)
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def scaffold_split_indices(
    smiles_list: Iterable[str],
    test_size: float,
    val_size: float,
    random_state: int,
) -> Dict[str, List[int]]:
    if Chem is None or MurckoScaffold is None:
        raise RuntimeError("RDKit is required for scaffold splitting.")

    smiles = list(smiles_list)
    scaffold_groups: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        scaffold = _scaffold_smiles(str(smi))
        if scaffold is None:
            scaffold = ""
        scaffold_groups.setdefault(scaffold, []).append(idx)

    rng = np.random.RandomState(random_state)
    groups = list(scaffold_groups.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_samples = len(smiles)
    n_test = int(round(test_size * n_samples))
    n_val = int(round(val_size * n_samples))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for group in groups:
        if len(test_idx) + len(group) <= n_test:
            test_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def tdc_split_indices(
    group: str,
    name: str,
    strategy: str,
    curated_smiles: Iterable[str],
) -> Dict[str, List[int]]:
    if group.upper() != "ADME":
        raise ValueError(f"Unsupported TDC group: {group}")
    from tdc.single_pred import ADME

    desired = "scaffold" if strategy.endswith("scaffold") else "random"
    data = ADME(name=name)
    split = data.get_split(method=desired)
    if not isinstance(split, dict):
        raise ValueError(f"TDC get_split(method={desired!r}) returned an invalid payload.")

    split_keys_lower = {str(k).lower() for k in split.keys()}
    if {"train", "test"}.issubset(split_keys_lower):
        selected = split
    else:
        strategy_key = None
        for key in split.keys():
            if desired in str(key).lower():
                strategy_key = key
                break
        if strategy_key is None:
            raise ValueError(
                f"TDC split does not expose a '{desired}' key. Available: {list(split.keys())}"
            )
        selected = split[strategy_key]
        if not isinstance(selected, dict):
            raise ValueError(
                f"TDC split key {strategy_key!r} does not contain split tables."
            )

    canonical_map = _build_canonical_index(curated_smiles)
    split_dict: Dict[str, List[int]] = {}
    for split_name, df in selected.items():
        smiles_col = _find_smiles_column(df.columns)
        if smiles_col is None:
            raise ValueError("TDC split frame missing SMILES column.")
        split_dict[split_name] = _map_split_indices(df[smiles_col].astype(str).tolist(), canonical_map)
    return split_dict


def save_split_indices(split_indices: Dict[str, List[int]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_indices, f, indent=2)


def build_split_indices(
    strategy: str,
    curated_df,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_column: Optional[str] = None,
    tdc_group: Optional[str] = None,
    tdc_name: Optional[str] = None,
) -> Dict[str, List[int]]:
    if strategy.startswith("tdc"):
        if not tdc_group or not tdc_name:
            raise ValueError("TDC split requires tdc_group and tdc_name.")
        if "canonical_smiles" not in curated_df.columns:
            raise ValueError("Curated data must include canonical_smiles for TDC split mapping.")
        return tdc_split_indices(tdc_group, tdc_name, strategy, curated_df["canonical_smiles"])

    stratify = None
    if stratify_column and stratify_column in curated_df.columns:
        stratify = curated_df[stratify_column].values

    if strategy == "scaffold":
        if "canonical_smiles" not in curated_df.columns:
            raise ValueError("Curated data must include canonical_smiles for scaffold splitting.")
        return scaffold_split_indices(
            curated_df["canonical_smiles"].astype(str).tolist(),
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )
    return random_split_indices(
        n_samples=len(curated_df),
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
