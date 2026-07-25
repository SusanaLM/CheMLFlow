"""Expanded, failure-tolerant molecular descriptor calculations for EDA."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


DESCRIPTOR_NAMES = [
    "MolWt",
    "HeavyAtomCount",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "FormalCharge",
    "NumHeteroatoms",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "NumChiralCenters",
]


def _descriptor_values(mol: Chem.Mol) -> dict[str, float]:
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "RingCount": float(Lipinski.RingCount(mol)),
        "AromaticRingCount": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "FormalCharge": float(Chem.GetFormalCharge(mol)),
        "NumHeteroatoms": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "NumSaturatedRings": float(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "NumAliphaticRings": float(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "NumChiralCenters": float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
    }


def calculate_molecular_descriptors(
    structure_df: pd.DataFrame,
    mols: Sequence[Chem.Mol],
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate descriptors per molecule; isolated failures become NaN warnings."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for position, mol in enumerate(mols):
        row: dict[str, Any] = {
            "structure_index": int(structure_df.iloc[position]["_structure_index"]),
            "source_row": int(structure_df.iloc[position]["_source_row"]),
            "compound_id": structure_df.iloc[position]["_compound_id"],
        }
        try:
            row.update(_descriptor_values(mol))
        except Exception as exc:
            row.update({name: np.nan for name in DESCRIPTOR_NAMES})
            warnings.append(
                f"Descriptor calculation failed for {row['compound_id']}: "
                f"{type(exc).__name__}: {exc}"
            )
        rows.append(row)
    return pd.DataFrame(rows), warnings


def summarize_descriptors(
    descriptor_df: pd.DataFrame,
    robust_z_threshold: float = 3.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize descriptors and detect robust univariate outliers."""
    summaries: dict[str, Any] = {}
    outliers: list[dict[str, Any]] = []
    for name in DESCRIPTOR_NAMES:
        values = pd.to_numeric(descriptor_df[name], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            summaries[name] = {"count": 0, "n_outliers": 0}
            continue
        median = float(valid.median())
        mad = float((valid - median).abs().median())
        q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
        iqr = q3 - q1
        if mad > 1e-12:
            robust_z = 0.6745 * (values - median) / mad
            method = "MAD"
        elif iqr > 1e-12:
            robust_z = 0.7413 * (values - median) / iqr
            method = "IQR fallback"
        else:
            robust_z = pd.Series(np.zeros(len(values)), index=values.index)
            method = "constant; no robust z-score"
        mask = robust_z.abs().ge(robust_z_threshold) & values.notna()
        summaries[name] = {
            "count": int(valid.count()),
            "missing": int(values.isna().sum()),
            "min": float(valid.min()),
            "q1": q1,
            "median": median,
            "mean": float(valid.mean()),
            "q3": q3,
            "max": float(valid.max()),
            "std": float(valid.std(ddof=0)),
            "mad": mad,
            "outlier_method": method,
            "robust_z_threshold": robust_z_threshold,
            "n_outliers": int(mask.sum()),
        }
        for idx in values.index[mask]:
            outliers.append(
                {
                    "structure_index": int(descriptor_df.loc[idx, "structure_index"]),
                    "source_row": int(descriptor_df.loc[idx, "source_row"]),
                    "compound_id": descriptor_df.loc[idx, "compound_id"],
                    "descriptor": name,
                    "value": float(values.loc[idx]),
                    "robust_z": float(robust_z.loc[idx]),
                    "method": method,
                }
            )
    return summaries, pd.DataFrame(
        outliers,
        columns=[
            "structure_index",
            "source_row",
            "compound_id",
            "descriptor",
            "value",
            "robust_z",
            "method",
        ],
    )
