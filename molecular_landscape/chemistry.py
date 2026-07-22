"""RDKit molecular preparation and representation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import FingerprintConfig


@dataclass
class MoleculeCohort:
    df: pd.DataFrame
    mols: List[Chem.Mol]
    exclusions: pd.DataFrame


def parse_molecules(
    df: pd.DataFrame,
    smiles_col: str,
    id_col: str,
) -> MoleculeCohort:
    valid_rows = []
    mols: List[Chem.Mol] = []
    exclusions = []
    for source_row, row in df.iterrows():
        identifier = row[id_col]
        smiles = row[smiles_col]
        reason = None
        mol = None
        if pd.isna(smiles) or not str(smiles).strip():
            reason = "missing SMILES"
        else:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
            except Exception as exc:
                reason = f"SMILES parsing exception: {type(exc).__name__}: {exc}"
            if mol is None and reason is None:
                reason = "RDKit could not parse or sanitize SMILES"

        if reason is not None:
            exclusions.append(
                {
                    "source_row": int(source_row),
                    "compound_id": identifier,
                    "exclusion_stage": "molecule_parsing",
                    "reason": reason,
                }
            )
            continue

        out = row.to_dict()
        out["_source_row"] = int(source_row)
        out["_compound_id"] = identifier
        out["_input_smiles"] = str(smiles)
        out["_canonical_smiles"] = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        valid_rows.append(out)
        mols.append(mol)

    if len(mols) < 4:
        raise ValueError("At least four valid molecules are required for PCA and UMAP.")
    return MoleculeCohort(
        df=pd.DataFrame(valid_rows).reset_index(drop=True),
        mols=mols,
        exclusions=pd.DataFrame(exclusions),
    )


def build_fingerprints(
    mols: Sequence[Chem.Mol],
    config: FingerprintConfig,
):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.radius,
        fpSize=config.n_bits,
        includeChirality=config.include_chirality,
        useBondTypes=True,
        atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            if config.use_features
            else None
        ),
    )
    return [generator.GetFingerprint(mol) for mol in mols]


def fingerprint_bit_matrix(fingerprints, n_bits: int) -> np.ndarray:
    matrix = np.zeros((len(fingerprints), n_bits), dtype=np.float32)
    for idx, fingerprint in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[idx])
    return matrix


def tanimoto_distance_matrix(fingerprints) -> np.ndarray:
    n_items = len(fingerprints)
    distances = np.zeros((n_items, n_items), dtype=np.float32)
    for idx in range(n_items - 1):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[idx],
                fingerprints[idx + 1 :],
            ),
            dtype=np.float32,
        )
        row_distances = 1.0 - similarities
        distances[idx, idx + 1 :] = row_distances
        distances[idx + 1 :, idx] = row_distances
    return distances


def build_typed_fingerprints(
    mols: Sequence[Chem.Mol],
    fp_type: str,
    n_bits: int = 2048,
    radius: int = 2,
):
    """Build one of several fingerprint families for representation comparison."""
    from rdkit.Chem import MACCSkeys

    if fp_type == "maccs":
        return [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    if fp_type == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=True
        )
    elif fp_type == "fcfp":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=True,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
    elif fp_type == "rdkit":
        generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    elif fp_type == "atompair":
        generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    elif fp_type == "torsion":
        generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    else:
        raise ValueError(f"Unsupported fingerprint representation: {fp_type}")
    return [generator.GetFingerprint(mol) for mol in mols]


def representation_sensitivity(
    reference_distance: np.ndarray,
    mols: Sequence[Chem.Mol],
    representations: Sequence[str],
    n_neighbors: int,
    random_state: int,
    n_bits: int = 2048,
    radius: int = 2,
    max_pairs: int = 200_000,
) -> pd.DataFrame:
    """Quantify how stable the chemical-space organization is to the fingerprint choice.

    For each alternative representation, reports the mean k-nearest-neighbour overlap
    and the sampled pairwise-distance rank correlation against the default (Morgan)
    Tanimoto geometry. High agreement means the local chemistry survives the
    representation choice; low agreement means map interpretation is representation
    dependent and must not be over-read.
    """
    from scipy.stats import spearmanr

    n_items = len(mols)
    k = max(2, min(n_neighbors, n_items - 1))

    def knn_sets(distance: np.ndarray) -> List[set]:
        order = np.argsort(distance, axis=1, kind="stable")
        result = []
        for idx in range(n_items):
            neighbours = order[idx][order[idx] != idx][:k]
            result.append(set(neighbours.tolist()))
        return result

    reference_neighbours = knn_sets(reference_distance)
    rng = np.random.default_rng(random_state)
    pair_budget = min(max_pairs, n_items * (n_items - 1) // 2)
    left = rng.integers(0, n_items, size=pair_budget)
    right = rng.integers(0, n_items, size=pair_budget)
    keep = left != right
    left, right = left[keep], right[keep]
    reference_pairs = reference_distance[left, right]

    rows = []
    for representation in representations:
        fingerprints = build_typed_fingerprints(mols, representation, n_bits, radius)
        distance = tanimoto_distance_matrix(fingerprints)
        neighbours = knn_sets(distance)
        overlap = float(
            np.mean(
                [
                    len(reference_neighbours[i] & neighbours[i]) / k
                    for i in range(n_items)
                ]
            )
        )
        spearman = spearmanr(reference_pairs, distance[left, right]).statistic
        rows.append(
            {
                "representation": representation,
                f"mean_knn_overlap_at_{k}": overlap,
                "distance_spearman_to_default": float(spearman),
            }
        )
    return pd.DataFrame(rows)


def fingerprint_collision_summary(
    df: pd.DataFrame,
    fingerprints,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    groups: Dict[str, List[int]] = {}
    for idx, fingerprint in enumerate(fingerprints):
        groups.setdefault(fingerprint.ToBitString(), []).append(idx)

    rows = []
    collision_group = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        collision_group += 1
        for idx in indices:
            rows.append(
                {
                    "collision_group": collision_group,
                    "group_size": len(indices),
                    "source_row": int(df.iloc[idx]["_source_row"]),
                    "compound_id": df.iloc[idx]["_compound_id"],
                    "canonical_smiles": df.iloc[idx]["_canonical_smiles"],
                }
            )
    collision_frame = pd.DataFrame(
        rows,
        columns=[
            "collision_group",
            "group_size",
            "source_row",
            "compound_id",
            "canonical_smiles",
        ],
    )
    return collision_frame, {
        "n_fingerprints": len(fingerprints),
        "n_unique_fingerprints": len(groups),
        "n_collision_member_rows": sum(
            len(items) for items in groups.values() if len(items) > 1
        ),
        "n_collision_excess_rows": len(fingerprints) - len(groups),
        "n_collision_groups": sum(len(items) > 1 for items in groups.values()),
    }


def identity_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Report identity concerns without silently removing scientifically valid rows."""
    rows: List[Dict[str, object]] = []

    def add_groups(issue_type: str, values: pd.Series, mask: pd.Series) -> int:
        group_count = 0
        for value in sorted(values[mask].dropna().unique().tolist(), key=str):
            members = values.eq(value) & mask
            if int(members.sum()) < 2:
                continue
            group_count += 1
            for idx in np.where(members.to_numpy())[0].tolist():
                rows.append(
                    {
                        "issue_type": issue_type,
                        "issue_group": group_count,
                        "group_size": int(members.sum()),
                        "source_row": int(df.iloc[idx]["_source_row"]),
                        "compound_id": df.iloc[idx]["_compound_id"],
                        "canonical_smiles": df.iloc[idx]["_canonical_smiles"],
                    }
                )
        return group_count

    identifiers = df["_compound_id"].map(
        lambda value: None if pd.isna(value) or not str(value).strip() else str(value).strip()
    )
    missing = identifiers.isna()
    for idx in np.where(missing.to_numpy())[0].tolist():
        rows.append(
            {
                "issue_type": "missing_compound_id",
                "issue_group": idx + 1,
                "group_size": 1,
                "source_row": int(df.iloc[idx]["_source_row"]),
                "compound_id": df.iloc[idx]["_compound_id"],
                "canonical_smiles": df.iloc[idx]["_canonical_smiles"],
            }
        )
    duplicate_id_groups = add_groups(
        "duplicate_compound_id",
        identifiers,
        identifiers.duplicated(keep=False) & identifiers.notna(),
    )
    canonical = df["_canonical_smiles"].astype(str)
    duplicate_structure_groups = add_groups(
        "duplicate_canonical_smiles",
        canonical,
        canonical.duplicated(keep=False),
    )
    columns = [
        "issue_type",
        "issue_group",
        "group_size",
        "source_row",
        "compound_id",
        "canonical_smiles",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame, {
        "n_missing_compound_ids": int(missing.sum()),
        "n_duplicate_id_groups": duplicate_id_groups,
        "n_duplicate_canonical_smiles_groups": duplicate_structure_groups,
        "n_identity_audit_rows": int(len(frame)),
    }


def assign_scaffold_families(
    df: pd.DataFrame,
    mols: Sequence[Chem.Mol],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaffold_smiles = [
        MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=True,
        )
        for mol in mols
    ]
    counts = pd.Series(scaffold_smiles).value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda item: (-counts[item], item))
    label_by_smiles = {smiles: idx for idx, smiles in enumerate(ordered)}

    out = df.copy()
    out["scaffold_smiles"] = scaffold_smiles
    out["scaffold_id"] = [label_by_smiles[item] for item in scaffold_smiles]
    out["scaffold_size"] = [int(counts[item]) for item in scaffold_smiles]

    summary = pd.DataFrame(
        [
            {
                "scaffold_id": label_by_smiles[smiles],
                "scaffold_smiles": smiles,
                "size": int(counts[smiles]),
            }
            for smiles in ordered
        ]
    )
    return out, summary


def molecular_descriptor_frame(mols: Sequence[Chem.Mol]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MolWt": [float(Descriptors.MolWt(mol)) for mol in mols],
            "TPSA": [float(rdMolDescriptors.CalcTPSA(mol)) for mol in mols],
            "LogP": [float(Crippen.MolLogP(mol)) for mol in mols],
            "HBA": [int(Lipinski.NumHAcceptors(mol)) for mol in mols],
            "HBD": [int(Lipinski.NumHDonors(mol)) for mol in mols],
            "RotBonds": [int(Lipinski.NumRotatableBonds(mol)) for mol in mols],
            "RingCount": [int(Lipinski.RingCount(mol)) for mol in mols],
            "HeavyAtoms": [int(mol.GetNumHeavyAtoms()) for mol in mols],
            "FractionCSP3": [float(rdMolDescriptors.CalcFractionCSP3(mol)) for mol in mols],
        }
    )


def descriptor_audit(
    df: pd.DataFrame,
    descriptor_cols: Sequence[str],
    robust_z_threshold: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize descriptors and flag robust univariate outliers."""
    summaries = []
    outliers = []
    for name in descriptor_cols:
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Descriptor '{name}' contains non-finite values.")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = 1.4826 * mad
        robust_z = (
            np.zeros_like(values)
            if robust_scale <= 1e-12
            else (values - median) / robust_scale
        )
        outlier_mask = np.abs(robust_z) > robust_z_threshold
        summaries.append(
            {
                "descriptor": name,
                "count": int(len(values)),
                "min": float(np.min(values)),
                "q1": float(np.quantile(values, 0.25)),
                "median": median,
                "mean": float(np.mean(values)),
                "q3": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "mad": mad,
                "robust_z_threshold": robust_z_threshold,
                "n_robust_outliers": int(outlier_mask.sum()),
            }
        )
        for idx in np.where(outlier_mask)[0].tolist():
            outliers.append(
                {
                    "source_row": int(df.iloc[idx]["_source_row"]),
                    "compound_id": df.iloc[idx]["_compound_id"],
                    "canonical_smiles": df.iloc[idx]["_canonical_smiles"],
                    "descriptor": name,
                    "value": float(values[idx]),
                    "robust_z": float(robust_z[idx]),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(
        outliers,
        columns=[
            "source_row",
            "compound_id",
            "canonical_smiles",
            "descriptor",
            "value",
            "robust_z",
        ],
    )
