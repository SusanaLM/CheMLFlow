"""Property distribution artifacts and structure-property summaries."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from .property_registry import PropertyProfile


def _numeric_property_distribution(
    molecule_table: pd.DataFrame,
    property_profile: PropertyProfile,
    descriptor_cols: Sequence[str],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    name = property_profile.column
    numeric = pd.to_numeric(molecule_table[name], errors="coerce")
    valid = numeric.dropna()
    quantiles = {
        str(key): float(value)
        for key, value in valid.quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items()
    }
    bins = pd.DataFrame(
        {
            "structure_index": molecule_table["structure_index"],
            "compound_id": molecule_table["compound_id"],
            name: numeric,
        }
    )
    bins["property_bin"] = pd.qcut(
        numeric,
        q=min(10, max(2, int(valid.nunique()))),
        duplicates="drop",
    ).astype(str)
    bins.loc[numeric.isna(), "property_bin"] = "missing"
    q1, q3 = valid.quantile([0.25, 0.75]) if len(valid) else (np.nan, np.nan)
    iqr = float(q3 - q1) if len(valid) else np.nan
    lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    outlier_mask = numeric.lt(lower) | numeric.gt(upper)
    outliers = molecule_table.loc[
        outlier_mask,
        ["structure_index", "compound_id", "canonical_smiles", "svg_path"],
    ].copy()
    outliers[name] = numeric[outlier_mask]
    outliers["outlier_direction"] = np.where(
        numeric[outlier_mask] < lower, "low", "high"
    )
    outliers["iqr_lower_bound"] = lower
    outliers["iqr_upper_bound"] = upper

    relationships = []
    for descriptor in descriptor_cols:
        descriptor_values = pd.to_numeric(molecule_table[descriptor], errors="coerce")
        mask = numeric.notna() & descriptor_values.notna()
        if int(mask.sum()) < 3:
            continue
        pearson = numeric[mask].corr(descriptor_values[mask], method="pearson")
        rank = spearmanr(numeric[mask], descriptor_values[mask]).statistic
        relationships.append(
            {
                "property": name,
                "descriptor": descriptor,
                "n": int(mask.sum()),
                "pearson": float(pearson),
                "spearman": float(rank),
            }
        )
    relationships_frame = pd.DataFrame(relationships)

    def grouped(group_column: str) -> pd.DataFrame:
        frame = molecule_table.assign(_property=numeric)
        result = (
            frame.groupby(group_column)["_property"]
            .agg(["count", "mean", "median", "min", "max", "std"])
            .reset_index()
        )
        result["iqr"] = (
            frame.groupby(group_column)["_property"]
            .quantile(0.75)
            .sub(frame.groupby(group_column)["_property"].quantile(0.25))
            .to_numpy()
        )
        return result.rename(columns={group_column: group_column})

    dynamic_range = float(valid.max() - valid.min()) if len(valid) else None
    skewness = float(skew(valid, bias=False)) if len(valid) >= 3 else None
    if skewness is not None and not np.isfinite(skewness):
        skewness = None
    notes = []
    if dynamic_range is not None:
        notes.append(f"The selected property spans {dynamic_range:.3g} units.")
    if skewness is not None and abs(skewness) >= 1:
        notes.append(
            f"The property distribution is substantially {'right' if skewness > 0 else 'left'}-skewed."
        )
    missing_fraction = float(numeric.isna().mean())
    if missing_fraction:
        notes.append(f"{missing_fraction:.1%} of valid molecules lack a usable property value.")
    if property_profile.semantic_type == "potency_log":
        notes.append(
            "Larger values indicate stronger potency; a difference of 1 is approximately ten-fold."
        )
    elif property_profile.semantic_type == "potency_linear":
        notes.append(
            "Lower values usually indicate stronger potency. Convert to a common log-molar scale before comparing mixed units."
        )
    else:
        notes.append("High and low values are described neutrally because this is not a recognised potency-log property.")
    summary = {
        "kind": "numeric",
        "property": name,
        "count": int(valid.count()),
        "missing": int(numeric.isna().sum()),
        "missing_fraction": missing_fraction,
        "min": float(valid.min()) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
        "dynamic_range": dynamic_range,
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "std": float(valid.std(ddof=0)) if len(valid) else None,
        "skew": skewness,
        "quantiles": quantiles,
        "iqr_outlier_count": int(outlier_mask.sum()),
        "interpretation_notes": notes,
    }
    return (
        summary,
        bins,
        outliers,
        grouped("scaffold_id"),
        grouped("cluster_id"),
        relationships_frame,
    )


def _categorical_property_distribution(
    molecule_table: pd.DataFrame,
    property_profile: PropertyProfile,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    name = property_profile.column
    values = molecule_table[name].fillna("<missing>").astype(str)
    counts = values.value_counts()
    bins = molecule_table[["structure_index", "compound_id"]].copy()
    bins["property_bin"] = values
    summary = {
        "kind": "categorical",
        "property": name,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "class_fractions": {
            str(key): float(value / len(values)) for key, value in counts.items()
        },
        "interpretation_notes": [
            "Class frequencies and scaffold association should be reviewed before classification modelling."
        ],
    }
    by_scaffold = pd.crosstab(molecule_table["scaffold_id"], values).reset_index()
    by_cluster = pd.crosstab(molecule_table["cluster_id"], values).reset_index()
    return summary, bins, pd.DataFrame(), by_scaffold, by_cluster, pd.DataFrame()


def build_property_distribution(
    molecule_table: pd.DataFrame,
    property_profile: PropertyProfile | None,
    descriptor_cols: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if property_profile is None:
        return (
            {"kind": "none", "interpretation_notes": ["No property was selected."]},
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    if property_profile.semantic_type in {"classification", "generic_categorical"}:
        return _categorical_property_distribution(molecule_table, property_profile)
    return _numeric_property_distribution(molecule_table, property_profile, descriptor_cols)
