"""Dataset-level EDA profile and interpretation notes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..schema import SchemaSelection, is_log_activity_property


def _value_counts(df: pd.DataFrame, columns: Iterable[str], limit: int = 20) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for column in columns:
        if column not in df.columns:
            continue
        counts = df[column].fillna("<missing>").astype(str).value_counts().head(limit)
        summaries[column] = {str(key): int(value) for key, value in counts.items()}
    return summaries


def _class_column(df: pd.DataFrame) -> Optional[str]:
    return next(
        (name for name in ("class", "activity_class", "label") if name in df.columns),
        None,
    )


def build_dataset_profile(
    *,
    schema: SchemaSelection,
    input_rows: int,
    structure_df: pd.DataFrame,
    property_df: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    descriptor_outliers: pd.DataFrame,
    invalid_molecules: int = 0,
) -> Dict[str, Any]:
    """Build machine-readable statistics plus cautious human interpretation."""
    warnings = []
    notes = []
    property_summary: Dict[str, Dict[str, Any]] = {}
    for name in schema.property_cols:
        raw_values = property_df[name]
        values = pd.to_numeric(raw_values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        categorical = int(values.count()) == 0 and int(raw_values.notna().sum()) > 0
        available = int(raw_values.notna().sum()) if categorical else int(values.count())
        property_summary[name] = (
            {
                "available": available,
                "missing_or_excluded": int(len(structure_df) - available),
                "kind": "categorical",
                "unique_values": int(raw_values.dropna().nunique()),
                "value_counts": {
                    str(key): int(value)
                    for key, value in raw_values.dropna().astype(str).value_counts().items()
                },
            }
            if categorical
            else {
                "available": available,
                "missing_or_excluded": int(len(structure_df) - available),
                "kind": "numeric",
                "min": float(values.min()) if values.count() else None,
                "median": float(values.median()) if values.count() else None,
                "mean": float(values.mean()) if values.count() else None,
                "max": float(values.max()) if values.count() else None,
                "std": float(values.std(ddof=0)) if values.count() else None,
            }
        )
        if len(structure_df) > available:
            warnings.append(
                f"{name}: {len(structure_df) - available} valid structures lack a "
                "scientifically usable property value."
            )
        if is_log_activity_property(name):
            notes.append(
                f"{name} is a negative-log molar activity measure; larger values "
                "usually indicate stronger potency and a difference of 1 is "
                "approximately ten-fold."
            )

    duplicate_ids = int(structure_df["_compound_id"].duplicated(keep=False).sum())
    duplicate_structures = int(
        structure_df["_canonical_smiles"].duplicated(keep=False).sum()
    )
    n_scaffolds = int(len(scaffold_summary))
    singleton_scaffolds = int((scaffold_summary["size"] == 1).sum())
    singleton_fraction = singleton_scaffolds / n_scaffolds if n_scaffolds else 0.0
    largest_scaffold_size = (
        int(scaffold_summary["size"].max()) if n_scaffolds else 0
    )
    if invalid_molecules:
        warnings.append(f"{invalid_molecules} input rows could not be parsed as valid molecules.")
    if duplicate_ids:
        warnings.append(f"{duplicate_ids} rows belong to duplicated compound identifiers.")
    if duplicate_structures:
        warnings.append(
            f"{duplicate_structures} rows belong to duplicated canonical structures."
        )
    if not descriptor_outliers.empty:
        warnings.append(
            f"{descriptor_outliers['source_row'].nunique()} molecules have at least "
            "one robust univariate descriptor outlier."
        )
    if singleton_fraction >= 0.5:
        warnings.append(
            f"Scaffold fragmentation is high: {singleton_fraction:.1%} of scaffold "
            "families are singletons."
        )
        notes.append(
            "Many singleton scaffolds make random train/test splits optimistic; "
            "use scaffold-aware validation before modelling."
        )

    relation_summaries = _value_counts(
        structure_df,
        ("standard_relation", "relation"),
    )
    censored = 0
    for column in ("standard_relation", "relation"):
        if column in structure_df.columns:
            values = structure_df[column].fillna("").astype(str).str.strip()
            censored += int((~values.isin(["", "="])).sum())
    if censored:
        warnings.append(
            f"{censored} relation-field observations are censored or non-equality records."
        )

    class_col = _class_column(structure_df)
    class_counts: Dict[str, int] = {}
    if class_col:
        class_counts = {
            str(key): int(value)
            for key, value in structure_df[class_col]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .items()
        }
        nonzero = [value for key, value in class_counts.items() if key != "<missing>"]
        if len(nonzero) >= 2 and min(nonzero) / sum(nonzero) < 0.2:
            warnings.append("The detected activity/class labels are substantially imbalanced.")
            notes.append(
                "Class imbalance makes naive accuracy misleading; use balanced and "
                "class-specific performance metrics."
            )

    notes.extend(
        [
            "The structure-only map is the least circular view for exploratory SAR.",
            "PCA and UMAP are projections and do not prove a mechanistic structure-activity relationship.",
            "Property-aware maps are supervised by the selected property and require cautious interpretation.",
        ]
    )
    return {
        "schema": {
            "smiles_column": schema.smiles_col,
            "id_column": schema.id_col,
            "property_columns": list(schema.property_cols),
            "property_detection": schema.property_detection,
            "class_column": class_col,
        },
        "counts": {
            "input_rows": int(input_rows),
            "valid_molecules": int(len(structure_df)),
            "invalid_molecules": int(invalid_molecules),
            "duplicate_id_rows": duplicate_ids,
            "duplicate_canonical_structure_rows": duplicate_structures,
            "scaffolds": n_scaffolds,
            "singleton_scaffolds": singleton_scaffolds,
            "largest_scaffold_size": largest_scaffold_size,
            "descriptor_outlier_molecules": int(
                descriptor_outliers["source_row"].nunique()
                if not descriptor_outliers.empty
                else 0
            ),
        },
        "property_summary": property_summary,
        "activity_class_counts": class_counts,
        "endpoint_summaries": _value_counts(
            structure_df,
            ("standard_type", "type", "bao_endpoint", "assay_type"),
        ),
        "unit_summaries": _value_counts(
            structure_df,
            ("standard_units", "units", "qudt_units"),
        ),
        "relation_summaries": relation_summaries,
        "warnings": warnings,
        "interpretation_notes": list(dict.fromkeys(notes)),
    }


def _warning(severity: str, category: str, message: str) -> Dict[str, str]:
    return {"severity": severity, "category": category, "message": message}


def build_dataset_health(
    *,
    input_df: pd.DataFrame,
    schema: SchemaSelection,
    structure_df: pd.DataFrame,
    molecule_exclusions: pd.DataFrame,
    descriptor_df: pd.DataFrame,
    descriptor_outliers: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    singleton_warning_fraction: float,
) -> tuple[Dict[str, Any], list[Dict[str, str]]]:
    """Build advanced chemistry/data health diagnostics and structured warnings."""
    warnings: list[Dict[str, str]] = []
    invalid_examples = (
        molecule_exclusions.head(10).to_dict(orient="records")
        if molecule_exclusions is not None and not molecule_exclusions.empty
        else []
    )
    identifiers = input_df[schema.id_col]
    duplicate_id_mask = identifiers.duplicated(keep=False) & identifiers.notna()
    duplicate_structure_mask = structure_df["_canonical_smiles"].duplicated(keep=False)
    property_health: Dict[str, Any] = {}
    for name in schema.property_cols:
        raw = input_df[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        nonnumeric_mask = raw.notna() & numeric.isna()
        property_health[name] = {
            "missing_count": int(raw.isna().sum()),
            "missing_fraction": float(raw.isna().mean()),
            "non_numeric_count": int(nonnumeric_mask.sum()),
            "non_numeric_examples": raw[nonnumeric_mask].astype(str).head(10).tolist(),
        }
        if raw.isna().any():
            warnings.append(
                _warning(
                    "warning",
                    "property_missingness",
                    f"{name} is missing for {int(raw.isna().sum())} input rows.",
                )
            )
        if nonnumeric_mask.any():
            warnings.append(
                _warning(
                    "warning",
                    "non_numeric_property",
                    f"{name} contains {int(nonnumeric_mask.sum())} non-numeric values.",
                )
            )

    relation_counts: Dict[str, Dict[str, int]] = {}
    censored_count = 0
    for column in ("standard_relation", "relation"):
        if column not in input_df.columns:
            continue
        normalized = input_df[column].fillna("<missing>").astype(str).str.strip()
        relation_counts[column] = {
            str(key): int(value) for key, value in normalized.value_counts().items()
        }
        censored_count += int((~normalized.isin(["", "=", "<missing>"])).sum())
    if censored_count:
        warnings.append(
            _warning(
                "warning",
                "censored_records",
                f"{censored_count} relation-field observations are censored or non-equality records.",
            )
        )

    def value_counts(columns: Iterable[str]) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = {}
        for column in columns:
            if column in input_df.columns:
                result[column] = {
                    str(key): int(value)
                    for key, value in input_df[column]
                    .fillna("<missing>")
                    .astype(str)
                    .value_counts()
                    .head(30)
                    .items()
                }
        return result

    unit_columns: tuple[str, ...] = ()
    for candidate in ("standard_units", "qudt_units", "units"):
        if candidate in input_df.columns and input_df[candidate].notna().any():
            unit_columns = (candidate,)
            break
    units = value_counts(unit_columns)
    for column, counts in units.items():
        nonmissing = [key for key in counts if key != "<missing>"]
        if len(nonmissing) > 1:
            warnings.append(
                _warning(
                    "warning",
                    "mixed_units",
                    f"{column} contains multiple units; harmonise units before quantitative modelling.",
                )
            )
    endpoints = value_counts(("standard_type", "type", "bao_endpoint", "assay_type"))

    formal_charge = pd.to_numeric(descriptor_df.get("FormalCharge"), errors="coerce")
    molecular_weight = pd.to_numeric(descriptor_df.get("MolWt"), errors="coerce")
    chiral_centers = pd.to_numeric(descriptor_df.get("NumChiralCenters"), errors="coerce")
    singleton_scaffolds = int((scaffold_summary["size"] == 1).sum())
    singleton_fraction = (
        float(singleton_scaffolds / len(scaffold_summary)) if len(scaffold_summary) else 0.0
    )
    if singleton_fraction >= singleton_warning_fraction:
        warnings.append(
            _warning(
                "warning",
                "scaffold_fragmentation",
                "A high fraction of scaffolds are singletons. Scaffold-aware validation "
                "may be difficult and random splits may be optimistic.",
            )
        )
    if not descriptor_outliers.empty:
        warnings.append(
            _warning(
                "warning",
                "descriptor_outliers",
                f"{descriptor_outliers['structure_index'].nunique()} molecules have at least one robust descriptor outlier.",
            )
        )
    if duplicate_id_mask.any():
        warnings.append(
            _warning(
                "warning",
                "duplicate_identifiers",
                f"{int(duplicate_id_mask.sum())} input rows have duplicated molecule identifiers.",
            )
        )
    if duplicate_structure_mask.any():
        warnings.append(
            _warning(
                "warning",
                "duplicate_structures",
                f"{int(duplicate_structure_mask.sum())} valid rows belong to duplicated canonical structures.",
            )
        )
    dot_disconnected = int(structure_df["_input_smiles"].astype(str).str.contains(".", regex=False).sum())
    if dot_disconnected:
        warnings.append(
            _warning(
                "warning",
                "salt_or_mixture_like",
                f"{dot_disconnected} valid records contain dot-disconnected SMILES and may be salts or mixtures.",
            )
        )
    health = {
        "plain_language_summary": (
            f"The dataset contains {len(input_df)} rows and {len(structure_df)} "
            f"RDKit-valid molecules across {len(scaffold_summary)} Bemis-Murcko scaffolds."
        ),
        "n_rows": int(len(input_df)),
        "n_valid_molecules": int(len(structure_df)),
        "n_invalid_molecules": int(len(input_df) - len(structure_df)),
        "invalid_row_examples": invalid_examples,
        "duplicate_id_rows": int(duplicate_id_mask.sum()),
        "duplicate_id_examples": identifiers[duplicate_id_mask].astype(str).head(10).tolist(),
        "duplicate_canonical_structure_rows": int(duplicate_structure_mask.sum()),
        "duplicate_canonical_structure_examples": structure_df.loc[
            duplicate_structure_mask, "_canonical_smiles"
        ].head(10).tolist(),
        "property_health": property_health,
        "censored_relation_observations": censored_count,
        "relation_summaries": relation_counts,
        "unit_summaries": units,
        "endpoint_summaries": endpoints,
        "charge_distribution": {
            "min": float(formal_charge.min()) if formal_charge.count() else None,
            "median": float(formal_charge.median()) if formal_charge.count() else None,
            "max": float(formal_charge.max()) if formal_charge.count() else None,
            "charged_fraction": float(formal_charge.ne(0).mean()) if formal_charge.count() else None,
        },
        "molecular_weight_extremes": {
            "min": float(molecular_weight.min()) if molecular_weight.count() else None,
            "median": float(molecular_weight.median()) if molecular_weight.count() else None,
            "max": float(molecular_weight.max()) if molecular_weight.count() else None,
            "below_100_count": int(molecular_weight.lt(100).sum()),
            "above_800_count": int(molecular_weight.gt(800).sum()),
        },
        "dot_disconnected_smiles_count": dot_disconnected,
        "stereochemistry": {
            "molecules_with_chiral_centers": int(chiral_centers.gt(0).sum()),
            "fraction_with_chiral_centers": float(chiral_centers.gt(0).mean())
            if chiral_centers.count()
            else None,
            "max_chiral_centers": int(chiral_centers.max()) if chiral_centers.count() else None,
        },
        "scaffold_fragmentation": {
            "n_scaffolds": int(len(scaffold_summary)),
            "singleton_scaffolds": singleton_scaffolds,
            "singleton_scaffold_fraction": singleton_fraction,
            "largest_scaffold_size": int(scaffold_summary["size"].max())
            if len(scaffold_summary)
            else 0,
        },
        "descriptor_outlier_molecules": int(
            descriptor_outliers["structure_index"].nunique()
            if not descriptor_outliers.empty
            else 0
        ),
    }
    return health, warnings
