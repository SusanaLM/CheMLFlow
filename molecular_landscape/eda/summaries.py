"""EDA-oriented molecule, descriptor, scaffold, and gallery summaries."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..chemistry import descriptor_audit
from ..embedding import EmbeddingResult


def _class_column(df: pd.DataFrame) -> Optional[str]:
    return next(
        (name for name in ("class", "activity_class", "label") if name in df.columns),
        None,
    )


def build_molecule_table(
    structure_df: pd.DataFrame,
    property_df: pd.DataFrame,
    property_cols: Sequence[str],
    descriptor_cols: Sequence[str],
    svg_paths: Dict[int, str],
    structure_results: Dict[str, EmbeddingResult],
    property_results: Optional[Dict[str, EmbeddingResult]],
) -> pd.DataFrame:
    """Create a report-facing table while preserving property-cohort exclusions."""
    table = pd.DataFrame(
        {
            "structure_index": structure_df["_structure_index"].astype(int),
            "source_row": structure_df["_source_row"].astype(int),
            "compound_id": structure_df["_compound_id"],
            "canonical_smiles": structure_df["_canonical_smiles"],
            "input_smiles": structure_df["_input_smiles"],
            "scaffold_id": structure_df["scaffold_id"].astype(int),
            "scaffold_smiles": structure_df["scaffold_smiles"],
            "scaffold_size": structure_df["scaffold_size"].astype(int),
            "cluster_id": structure_df["butina_cluster_id"].astype(int),
            "fingerprint_collision": structure_df[
                "fingerprint_collision"
            ].astype(bool),
            "fingerprint_collision_group": structure_df[
                "fingerprint_collision_group"
            ].astype("Int64"),
            "fingerprint_collision_group_size": structure_df[
                "fingerprint_collision_group_size"
            ].astype(int),
        }
    )
    class_col = _class_column(structure_df)
    if class_col:
        table["activity_class"] = structure_df[class_col]
    for name in descriptor_cols:
        table[name] = structure_df[name]
    for method, result in structure_results.items():
        table[f"structure_{method}_x"] = result.coordinates[:, 0]
        table[f"structure_{method}_y"] = result.coordinates[:, 1]

    table["has_valid_property"] = False
    for name in property_cols:
        table[name] = pd.Series([None] * len(table), dtype=object)
    if property_cols and not property_df.empty:
        property_by_index = property_df.set_index("_structure_index")
        valid_indices = property_by_index.index.astype(int)
        table.loc[table["structure_index"].isin(valid_indices), "has_valid_property"] = True
        for name in property_cols:
            table[name] = table["structure_index"].map(property_by_index[name])
        if property_results:
            for method, result in property_results.items():
                coordinates = {
                    int(index): coordinate
                    for index, coordinate in zip(valid_indices, result.coordinates)
                }
                table[f"property_aware_{method}_x"] = table["structure_index"].map(
                    lambda index: coordinates.get(int(index), [np.nan, np.nan])[0]
                )
                table[f"property_aware_{method}_y"] = table["structure_index"].map(
                    lambda index: coordinates.get(int(index), [np.nan, np.nan])[1]
                )
    table["svg_path"] = table["structure_index"].map(svg_paths)
    table["cluster_size"] = table.groupby("cluster_id")["cluster_id"].transform("size")
    table["structural_outlier"] = (
        table["scaffold_size"].eq(1) & table["cluster_size"].eq(1)
    )
    return table


def build_descriptor_tables(
    structure_df: pd.DataFrame,
    descriptor_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return descriptor_audit(structure_df, descriptor_cols)


def _active_mask(values: pd.Series) -> Optional[pd.Series]:
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    labels = set(normalized[normalized.ne("")].unique())
    if not labels:
        return None
    active_labels = {"active", "1", "true", "yes", "positive"}
    inactive_labels = {"inactive", "0", "false", "no", "negative"}
    if not labels.issubset(active_labels | inactive_labels):
        return None
    return normalized.isin(active_labels)


def build_scaffold_summary(
    molecule_table: pd.DataFrame,
    property_col: Optional[str],
    structure_distance: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows = []
    for scaffold_id, group in molecule_table.groupby("scaffold_id", sort=False):
        if structure_distance is not None:
            members = group.index.to_numpy(dtype=int)
            medoid_position = int(
                members[np.argmin(structure_distance[np.ix_(members, members)].mean(axis=1))]
            )
            representative = molecule_table.iloc[medoid_position]
        else:
            representative = group.iloc[0]
        row = {
            "scaffold_id": int(scaffold_id),
            "scaffold_smiles": representative["scaffold_smiles"],
            "size": int(len(group)),
            "n_molecules": int(len(group)),
            "representative_compound_id": representative["compound_id"],
            "representative_structure_index": int(representative["structure_index"]),
            "representative_svg": representative.get("svg_path"),
            "representative_svg_path": representative.get("svg_path"),
            "cluster_overlap": int(group["cluster_id"].nunique()),
        }
        for descriptor in (
            "MolWt",
            "MolLogP",
            "TPSA",
            "NumHDonors",
            "NumHAcceptors",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ):
            if descriptor in group.columns:
                row[f"{descriptor}_median"] = float(
                    pd.to_numeric(group[descriptor], errors="coerce").median()
                )
        if property_col:
            values = pd.to_numeric(group[property_col], errors="coerce").dropna()
            row.update(
                {
                    "property_count": int(len(values)),
                    "property_min": float(values.min()) if len(values) else None,
                    "property_q1": float(values.quantile(0.25)) if len(values) else None,
                    "property_mean": float(values.mean()) if len(values) else None,
                    "property_median": float(values.median()) if len(values) else None,
                    "property_q3": float(values.quantile(0.75)) if len(values) else None,
                    "property_max": float(values.max()) if len(values) else None,
                    "property_iqr": float(
                        values.quantile(0.75) - values.quantile(0.25)
                    )
                    if len(values)
                    else None,
                }
            )
        if "activity_class" in group.columns:
            mask = _active_mask(group["activity_class"])
            row["active_fraction"] = float(mask.mean()) if mask is not None else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["size", "scaffold_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_cluster_summary(
    molecule_table: pd.DataFrame,
    butina_summary: pd.DataFrame,
    property_col: Optional[str],
) -> pd.DataFrame:
    rows = []
    summary_by_id = butina_summary.set_index("cluster_id")
    for cluster_id, group in molecule_table.groupby("cluster_id", sort=False):
        base = summary_by_id.loc[int(cluster_id)].to_dict()
        medoid_source_row = int(base["medoid_source_row"])
        representative = group[group["source_row"] == medoid_source_row].iloc[0]
        row = {
            "cluster_id": int(cluster_id),
            "size": int(len(group)),
            "representative_compound_id": representative["compound_id"],
            "representative_structure_index": int(representative["structure_index"]),
            "representative_svg": representative.get("svg_path"),
            "minimum_pairwise_similarity": base["minimum_pairwise_similarity"],
            "median_pairwise_similarity": base["median_pairwise_similarity"],
            "mean_pairwise_similarity": base["mean_pairwise_similarity"],
        }
        for descriptor in (
            "MolWt",
            "MolLogP",
            "TPSA",
            "NumHDonors",
            "NumHAcceptors",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ):
            if descriptor in group.columns:
                row[f"{descriptor}_median"] = float(
                    pd.to_numeric(group[descriptor], errors="coerce").median()
                )
        if property_col:
            values = pd.to_numeric(group[property_col], errors="coerce").dropna()
            row.update(
                {
                    "property_count": int(len(values)),
                    "property_min": float(values.min()) if len(values) else None,
                    "property_q1": float(values.quantile(0.25)) if len(values) else None,
                    "property_mean": float(values.mean()) if len(values) else None,
                    "property_median": float(values.median()) if len(values) else None,
                    "property_q3": float(values.quantile(0.75)) if len(values) else None,
                    "property_max": float(values.max()) if len(values) else None,
                    "property_iqr": float(
                        values.quantile(0.75) - values.quantile(0.25)
                    )
                    if len(values)
                    else None,
                }
            )
        if "activity_class" in group.columns:
            mask = _active_mask(group["activity_class"])
            row["active_fraction"] = float(mask.mean()) if mask is not None else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["size", "cluster_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_gallery_candidates(
    molecule_table: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    descriptor_outliers: pd.DataFrame,
    property_col: Optional[str],
    representative_molecules: int,
    random_state: int,
) -> Dict[str, List[int]]:
    limit = representative_molecules
    random_rows = molecule_table.sample(
        n=min(limit, len(molecule_table)),
        random_state=random_state,
    )
    galleries: Dict[str, List[int]] = {
        "random_representatives": random_rows["structure_index"].astype(int).tolist(),
        "top_scaffold_representatives": scaffold_summary[
            "representative_structure_index"
        ]
        .head(limit)
        .astype(int)
        .tolist(),
    }
    if property_col:
        valid = molecule_table.dropna(subset=[property_col])
        galleries["high_property"] = (
            valid.nlargest(min(limit, len(valid)), property_col)["structure_index"]
            .astype(int)
            .tolist()
        )
        galleries["low_property"] = (
            valid.nsmallest(min(limit, len(valid)), property_col)["structure_index"]
            .astype(int)
            .tolist()
        )
    outlier_sources = set(descriptor_outliers.get("source_row", pd.Series(dtype=int)))
    galleries["descriptor_outliers"] = (
        molecule_table[molecule_table["source_row"].isin(outlier_sources)]
        .head(limit)["structure_index"]
        .astype(int)
        .tolist()
    )
    galleries["structural_outliers"] = (
        molecule_table[molecule_table["structural_outlier"]]
        .head(limit)["structure_index"]
        .astype(int)
        .tolist()
    )
    return galleries


def scaffold_representatives(scaffold_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scaffold_id",
        "scaffold_smiles",
        "n_molecules",
        "representative_compound_id",
        "representative_structure_index",
        "representative_svg_path",
    ]
    return scaffold_summary[[name for name in columns if name in scaffold_summary.columns]]


def scaffold_property_enrichment(
    scaffold_summary: pd.DataFrame,
    higher_is_better: bool | None,
) -> pd.DataFrame:
    if "property_median" not in scaffold_summary.columns:
        return pd.DataFrame()
    frame = scaffold_summary.copy()
    global_weighted_median = float(
        np.average(
            frame["property_median"].fillna(frame["property_median"].median()),
            weights=frame["property_count"].clip(lower=1),
        )
    )
    frame["property_median_difference_from_weighted_global"] = (
        frame["property_median"] - global_weighted_median
    )
    if higher_is_better is None:
        frame["enrichment_direction"] = np.where(
            frame["property_median_difference_from_weighted_global"].ge(0),
            "higher property",
            "lower property",
        )
    else:
        favorable = frame["property_median_difference_from_weighted_global"].ge(0)
        if not higher_is_better:
            favorable = ~favorable
        frame["enrichment_direction"] = np.where(favorable, "favorable", "unfavorable")
    return frame.sort_values(
        "property_median_difference_from_weighted_global",
        ascending=False,
    )
