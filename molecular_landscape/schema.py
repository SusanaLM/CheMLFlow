"""Conservative schema detection and scientific column classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd


SMILES_ALIASES = (
    "canonical_smiles",
    "smiles",
    "SMILES",
    "mol_smiles",
)
ID_ALIASES = (
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "mol_id",
    "compound_id",
    "id",
    "name",
    "__row_index",
)
PROPERTY_ALIASES = (
    "pchembl_value",
    "pIC50",
    "pic50",
    "pKi",
    "pki",
    "pKd",
    "pkd",
    "activity",
    "activity_value",
    "response",
    "target_value",
)
LOG_ACTIVITY_NAMES = {
    "pchembl_value",
    "pic50",
    "pki",
    "pkd",
    "pec50",
    "pactivity",
}
ACTIVITY_NAMES = LOG_ACTIVITY_NAMES | {
    "ic50",
    "ki",
    "kd",
    "ec50",
    "activity",
    "activity_value",
    "response",
    "target_value",
}


@dataclass(frozen=True)
class SchemaSelection:
    smiles_col: str
    id_col: str
    property_cols: List[str]
    property_detection: str


def _first_present(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    available = set(columns)
    return next((alias for alias in aliases if alias in available), None)


def is_activity_property(name: str) -> bool:
    normalized = name.lower().replace("-", "").replace("_", "")
    activity = {item.replace("_", "").lower() for item in ACTIVITY_NAMES}
    return normalized in activity or normalized.startswith("pchembl")


def is_log_activity_property(name: str) -> bool:
    normalized = name.lower().replace("-", "").replace("_", "")
    log_names = {item.replace("_", "").lower() for item in LOG_ACTIVITY_NAMES}
    return normalized in log_names or normalized.startswith("pchembl")


def detect_schema(
    df: pd.DataFrame,
    smiles_col: Optional[str],
    id_col: Optional[str],
    property_cols: Optional[Sequence[str]],
) -> SchemaSelection:
    columns = list(df.columns)

    selected_smiles = smiles_col or _first_present(columns, SMILES_ALIASES)
    if selected_smiles is None:
        raise ValueError(
            "Could not conservatively identify a SMILES column. "
            "Pass --smiles-col explicitly."
        )
    if selected_smiles not in df.columns:
        raise ValueError(f"SMILES column not found: {selected_smiles}")

    selected_id = id_col or _first_present(columns, ID_ALIASES)
    if selected_id is None:
        raise ValueError(
            "Could not conservatively identify a compound identifier column. "
            "Pass --id-col explicitly."
        )
    if selected_id not in df.columns:
        raise ValueError(f"Identifier column not found: {selected_id}")

    if property_cols is not None:
        selected_properties = list(property_cols)
        duplicates = sorted(
            {
                name
                for name in selected_properties
                if selected_properties.count(name) > 1
            }
        )
        if duplicates:
            raise ValueError(f"Duplicate property columns are not allowed: {duplicates}")
        missing = [name for name in selected_properties if name not in df.columns]
        if missing:
            raise ValueError(f"Selected property columns not found: {missing}")
        if not selected_properties:
            return SchemaSelection(
                selected_smiles,
                selected_id,
                [],
                "explicit-none",
            )
        return SchemaSelection(
            selected_smiles,
            selected_id,
            selected_properties,
            "explicit",
        )

    candidates = [name for name in PROPERTY_ALIASES if name in df.columns]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple plausible property columns were detected. "
            f"Pass --property-cols explicitly: {candidates}"
        )
    return SchemaSelection(
        selected_smiles,
        selected_id,
        candidates,
        "auto-single" if candidates else "auto-none",
    )
