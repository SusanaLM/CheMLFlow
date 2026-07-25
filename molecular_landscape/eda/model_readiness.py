"""Practical, transparent model-readiness diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew

from .property_registry import PropertyProfile


def build_model_readiness(
    molecule_table: pd.DataFrame,
    property_profile: PropertyProfile | None,
    nearest_neighbors: pd.DataFrame,
    discontinuities: pd.DataFrame,
    descriptor_outliers: pd.DataFrame,
) -> dict[str, Any]:
    recommendations: list[str] = []
    n_rows = len(molecule_table)
    scaffold_sizes = molecule_table.groupby("scaffold_id").size()
    singleton_fraction = float((scaffold_sizes == 1).mean())
    largest_fraction = float(scaffold_sizes.max() / n_rows)
    duplicate_groups = molecule_table.groupby("canonical_smiles")
    duplicate_structure_groups = int((duplicate_groups.size() > 1).sum())
    duplicate_conflicts = 0

    diagnostics: dict[str, Any] = {
        "n_molecules": n_rows,
        "duplicate_structure_groups": duplicate_structure_groups,
        "descriptor_outlier_fraction": float(
            descriptor_outliers["structure_index"].nunique() / n_rows
            if n_rows and not descriptor_outliers.empty
            else 0.0
        ),
        "singleton_scaffold_fraction": singleton_fraction,
        "largest_scaffold_fraction": largest_fraction,
        "local_discontinuity_count": int(len(discontinuities)),
    }
    if duplicate_structure_groups:
        recommendations.append(
            "Review and deduplicate repeated canonical structures before splitting."
        )
    if largest_fraction >= 0.10:
        recommendations.append(
            "Large scaffold families are present; random splits may overestimate generalisation."
        )
    if singleton_fraction >= 0.30:
        recommendations.append(
            "Many singleton scaffolds are present; scaffold splits may be severe or unstable."
        )
    if len(discontinuities):
        recommendations.append(
            "Local property discontinuities are present; smooth similarity-based models may struggle in these neighbourhoods."
        )
    if diagnostics["descriptor_outlier_fraction"] > 0:
        recommendations.append("Review descriptor outliers before training.")

    if property_profile is not None:
        name = property_profile.column
        values = molecule_table[name]
        if property_profile.semantic_type in {"classification", "generic_categorical"}:
            labels = values.dropna().astype(str)
            counts = labels.value_counts()
            minority_fraction = float(counts.min() / counts.sum()) if len(counts) else None
            duplicate_conflicts = sum(
                group[name].dropna().astype(str).nunique() > 1
                for _, group in duplicate_groups
            )
            same_neighbor = np.nan
            if not nearest_neighbors.empty:
                left = nearest_neighbors["query_property"].astype(str)
                right = nearest_neighbors["neighbor_property"].astype(str)
                same_neighbor = float((left == right).mean())
            diagnostics.update(
                {
                    "property_kind": "categorical",
                    "n_labelled": int(labels.count()),
                    "class_counts": {str(key): int(value) for key, value in counts.items()},
                    "minority_class_fraction": minority_fraction,
                    "duplicate_structure_label_conflicts": int(duplicate_conflicts),
                    "nearest_neighbor_label_consistency": same_neighbor,
                }
            )
            if minority_fraction is not None and minority_fraction < 0.20:
                recommendations.append(
                    "Class imbalance is substantial; use balanced and class-specific metrics rather than accuracy alone."
                )
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            valid = numeric.dropna()
            duplicate_conflicts = sum(
                pd.to_numeric(group[name], errors="coerce").dropna().nunique() > 1
                for _, group in duplicate_groups
            )
            consistency = None
            if not nearest_neighbors.empty:
                deltas = pd.to_numeric(
                    nearest_neighbors["absolute_property_difference"], errors="coerce"
                ).dropna()
                consistency = float(deltas.median()) if len(deltas) else None
            dynamic_range = float(valid.max() - valid.min()) if len(valid) else None
            property_skew = float(skew(valid, bias=False)) if len(valid) >= 3 else None
            if property_skew is not None and not np.isfinite(property_skew):
                property_skew = None
            diagnostics.update(
                {
                    "property_kind": "numeric",
                    "n_labelled": int(valid.count()),
                    "property_missing_fraction": float(numeric.isna().mean()),
                    "property_dynamic_range": dynamic_range,
                    "property_std": float(valid.std(ddof=0)) if len(valid) else None,
                    "property_skew": property_skew,
                    "duplicate_structure_property_conflicts": int(duplicate_conflicts),
                    "nearest_neighbor_median_absolute_property_difference": consistency,
                }
            )
            if dynamic_range is not None and dynamic_range < 1.0:
                recommendations.append(
                    "The property range is narrow; continuous regression may be difficult."
                )
    if duplicate_conflicts:
        recommendations.append(
            "Duplicate structures with conflicting labels/properties require review before modelling."
        )
    recommendations.append("Use scaffold-aware validation alongside a random-split baseline.")
    diagnostics["recommendations"] = list(dict.fromkeys(recommendations))
    return diagnostics
