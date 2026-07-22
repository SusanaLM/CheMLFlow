"""Scientifically explicit property-cohort preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .schema import is_activity_property, is_log_activity_property


@dataclass
class PropertyCohort:
    df: pd.DataFrame
    raw_matrix: np.ndarray
    processed_matrix: np.ndarray
    exclusions: pd.DataFrame
    summary: Dict[str, object]


def select_transform(property_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if is_log_activity_property(property_name):
        return "none"
    return "none"


def transform_property(
    values: np.ndarray,
    method: str,
    random_state: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if method == "none":
        return arr[:, 0].astype(np.float32)
    if method == "log1p":
        if np.any(arr < 0):
            raise ValueError("log1p requires non-negative property values.")
        return np.log1p(arr[:, 0]).astype(np.float32)
    if method == "signed_log1p":
        return (np.sign(arr[:, 0]) * np.log1p(np.abs(arr[:, 0]))).astype(
            np.float32
        )
    if method == "quantile":
        transformer = QuantileTransformer(
            n_quantiles=min(200, len(arr)),
            output_distribution="normal",
            random_state=random_state,
        )
        return transformer.fit_transform(arr)[:, 0].astype(np.float32)
    raise ValueError(f"Unsupported property transform: {method}")


def _activity_exclusion_reasons(
    df: pd.DataFrame,
    activity_selected: bool,
) -> List[List[str]]:
    reasons: List[List[str]] = [[] for _ in range(len(df))]
    if not activity_selected:
        return reasons

    for column in ("data_validity_comment", "data_validity_description"):
        if column in df.columns:
            for idx, value in enumerate(df[column].tolist()):
                if pd.notna(value) and str(value).strip():
                    reasons[idx].append(f"{column}: {str(value).strip()}")

    for relation_column in ("standard_relation", "relation"):
        if relation_column not in df.columns:
            continue
        for idx, value in enumerate(df[relation_column].tolist()):
            if pd.notna(value) and str(value).strip() not in {"", "="}:
                reasons[idx].append(
                    f"censored activity ({relation_column}={str(value).strip()})"
                )
    return reasons


def prepare_property_cohort(
    valid_molecule_df: pd.DataFrame,
    property_cols: Sequence[str],
    requested_transforms: Dict[str, str],
    random_state: int,
) -> PropertyCohort:
    if not property_cols:
        empty = np.empty((0, 0), dtype=np.float32)
        return PropertyCohort(
            df=valid_molecule_df.iloc[0:0].copy(),
            raw_matrix=empty,
            processed_matrix=empty,
            exclusions=pd.DataFrame(
                columns=["source_row", "compound_id", "exclusion_stage", "reason"]
            ),
            summary={
                "property_cols": [],
                "n_rows": 0,
                "note": "No property was selected; property-dependent maps were not generated.",
            },
        )

    numeric = pd.DataFrame(index=valid_molecule_df.index)
    for name in property_cols:
        numeric[name] = pd.to_numeric(valid_molecule_df[name], errors="coerce")

    activity_selected = any(is_activity_property(name) for name in property_cols)
    reasons = _activity_exclusion_reasons(valid_molecule_df, activity_selected)
    for idx, name in enumerate(property_cols):
        values = numeric[name].to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        for row_idx in np.where(missing)[0].tolist():
            reasons[row_idx].append(f"missing, non-numeric, or non-finite property: {name}")

    keep_mask = np.asarray([not row_reasons for row_reasons in reasons], dtype=bool)
    exclusions = []
    for idx, row_reasons in enumerate(reasons):
        if row_reasons:
            row = valid_molecule_df.iloc[idx]
            exclusions.append(
                {
                    "source_row": int(row["_source_row"]),
                    "compound_id": row["_compound_id"],
                    "exclusion_stage": "property_cohort",
                    "reason": "; ".join(dict.fromkeys(row_reasons)),
                }
            )

    cohort_df = valid_molecule_df.loc[keep_mask].reset_index(drop=True).copy()
    raw_df = numeric.loc[keep_mask].reset_index(drop=True)
    if len(cohort_df) < 4:
        raise ValueError(
            "At least four scientifically valid rows are required for "
            "property-dependent analysis."
        )

    processed = pd.DataFrame(index=raw_df.index)
    transforms: Dict[str, str] = {}
    property_stats: Dict[str, Dict[str, float]] = {}
    for name in property_cols:
        method = select_transform(name, requested_transforms.get(name, "auto"))
        transformed = transform_property(
            raw_df[name].to_numpy(),
            method=method,
            random_state=random_state,
        )
        if not np.isfinite(transformed).all():
            raise ValueError(f"Property '{name}' is non-finite after transformation.")
        if float(np.nanstd(transformed)) <= 1e-12:
            raise ValueError(f"Property '{name}' is constant after transformation.")
        processed[name] = transformed
        transforms[name] = method
        property_stats[name] = {
            "count": int(raw_df[name].count()),
            "min": float(raw_df[name].min()),
            "median": float(raw_df[name].median()),
            "mean": float(raw_df[name].mean()),
            "max": float(raw_df[name].max()),
            "std": float(raw_df[name].std(ddof=0)),
        }
        cohort_df[f"{name}__processed"] = transformed

    standardized = StandardScaler().fit_transform(processed.to_numpy()).astype(
        np.float32
    )
    return PropertyCohort(
        df=cohort_df,
        raw_matrix=raw_df.to_numpy(dtype=np.float32),
        processed_matrix=standardized,
        exclusions=pd.DataFrame(exclusions),
        summary={
            "property_cols": list(property_cols),
            "activity_property_selected": activity_selected,
            "n_rows": int(len(cohort_df)),
            "n_excluded": int((~keep_mask).sum()),
            "transforms": transforms,
            "property_stats": property_stats,
        },
    )
