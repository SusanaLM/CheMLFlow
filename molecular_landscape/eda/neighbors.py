"""Tanimoto nearest-neighbor and similarity-defined discontinuity analysis."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


NEIGHBOR_COLUMNS = [
    "query_structure_index",
    "neighbor_structure_index",
    "query_compound_id",
    "neighbor_compound_id",
    "tanimoto_similarity",
    "query_property",
    "neighbor_property",
    "property_difference",
    "absolute_property_difference",
    "same_scaffold",
    "query_fingerprint_collision",
    "neighbor_fingerprint_collision",
    "same_fingerprint_collision_group",
    "collision_derived_match",
    "query_svg_path",
    "neighbor_svg_path",
]


def compute_neighbors_and_cliffs(
    molecule_table: pd.DataFrame,
    structure_distance: np.ndarray,
    property_col: Optional[str],
    nearest_neighbors: int,
    cliff_similarity: float,
    cliff_delta: float,
    property_is_numeric: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use an existing pairwise distance matrix to create local SAR tables."""
    if structure_distance.shape != (len(molecule_table), len(molecule_table)):
        raise ValueError("Structure-distance matrix and molecule table are misaligned.")

    rows = []
    n_neighbors = min(nearest_neighbors, max(0, len(molecule_table) - 1))
    for query_idx in range(len(molecule_table)):
        order = np.argsort(structure_distance[query_idx], kind="stable")
        neighbors = [int(idx) for idx in order if int(idx) != query_idx][:n_neighbors]
        query = molecule_table.iloc[query_idx]
        for neighbor_idx in neighbors:
            neighbor = molecule_table.iloc[neighbor_idx]
            if property_col and property_is_numeric:
                query_property = pd.to_numeric(
                    pd.Series([query[property_col]]), errors="coerce"
                ).iloc[0]
                neighbor_property = pd.to_numeric(
                    pd.Series([neighbor[property_col]]), errors="coerce"
                ).iloc[0]
            elif property_col:
                query_property = query[property_col]
                neighbor_property = neighbor[property_col]
            else:
                query_property = np.nan
                neighbor_property = np.nan
            difference = (
                float(query_property - neighbor_property)
                if property_is_numeric
                and pd.notna(query_property)
                and pd.notna(neighbor_property)
                else np.nan
            )
            rows.append(
                {
                    "query_structure_index": int(query["structure_index"]),
                    "neighbor_structure_index": int(neighbor["structure_index"]),
                    "query_compound_id": query["compound_id"],
                    "neighbor_compound_id": neighbor["compound_id"],
                    "tanimoto_similarity": float(1.0 - structure_distance[query_idx, neighbor_idx]),
                    "query_property": query_property,
                    "neighbor_property": neighbor_property,
                    "property_difference": difference,
                    "absolute_property_difference": (
                        abs(difference) if np.isfinite(difference) else np.nan
                    ),
                    "same_scaffold": bool(query["scaffold_id"] == neighbor["scaffold_id"]),
                    "query_fingerprint_collision": bool(
                        query.get("fingerprint_collision", False)
                    ),
                    "neighbor_fingerprint_collision": bool(
                        neighbor.get("fingerprint_collision", False)
                    ),
                    "same_fingerprint_collision_group": bool(
                        pd.notna(query.get("fingerprint_collision_group"))
                        and query.get("fingerprint_collision_group")
                        == neighbor.get("fingerprint_collision_group")
                    ),
                    "collision_derived_match": bool(
                        pd.notna(query.get("fingerprint_collision_group"))
                        and query.get("fingerprint_collision_group")
                        == neighbor.get("fingerprint_collision_group")
                        and query["canonical_smiles"] != neighbor["canonical_smiles"]
                    ),
                    "query_svg_path": query.get("svg_path"),
                    "neighbor_svg_path": neighbor.get("svg_path"),
                }
            )
    nearest = pd.DataFrame(rows, columns=NEIGHBOR_COLUMNS)

    cliff_rows = []
    if property_col and property_is_numeric:
        values = pd.to_numeric(molecule_table[property_col], errors="coerce").to_numpy()
        for left in range(len(molecule_table) - 1):
            if not np.isfinite(values[left]):
                continue
            for right in range(left + 1, len(molecule_table)):
                if not np.isfinite(values[right]):
                    continue
                similarity = float(1.0 - structure_distance[left, right])
                difference = float(values[left] - values[right])
                if similarity < cliff_similarity or abs(difference) < cliff_delta:
                    continue
                left_row = molecule_table.iloc[left]
                right_row = molecule_table.iloc[right]
                cliff_rows.append(
                    {
                        "query_structure_index": int(left_row["structure_index"]),
                        "neighbor_structure_index": int(right_row["structure_index"]),
                        "query_compound_id": left_row["compound_id"],
                        "neighbor_compound_id": right_row["compound_id"],
                        "tanimoto_similarity": similarity,
                        "query_property": float(values[left]),
                        "neighbor_property": float(values[right]),
                        "property_difference": difference,
                        "absolute_property_difference": abs(difference),
                        "same_scaffold": bool(
                            left_row["scaffold_id"] == right_row["scaffold_id"]
                        ),
                        "query_fingerprint_collision": bool(
                            left_row.get("fingerprint_collision", False)
                        ),
                        "neighbor_fingerprint_collision": bool(
                            right_row.get("fingerprint_collision", False)
                        ),
                        "same_fingerprint_collision_group": bool(
                            pd.notna(left_row.get("fingerprint_collision_group"))
                            and left_row.get("fingerprint_collision_group")
                            == right_row.get("fingerprint_collision_group")
                        ),
                        "collision_derived_match": bool(
                            pd.notna(left_row.get("fingerprint_collision_group"))
                            and left_row.get("fingerprint_collision_group")
                            == right_row.get("fingerprint_collision_group")
                            and left_row["canonical_smiles"]
                            != right_row["canonical_smiles"]
                        ),
                        "query_svg_path": left_row.get("svg_path"),
                        "neighbor_svg_path": right_row.get("svg_path"),
                    }
                )
    cliffs = pd.DataFrame(cliff_rows, columns=NEIGHBOR_COLUMNS)
    if not cliffs.empty:
        cliffs = cliffs.sort_values(
            ["absolute_property_difference", "tanimoto_similarity"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return nearest, cliffs


def summarize_neighborhood_consistency(
    nearest_neighbors: pd.DataFrame,
    discontinuities: pd.DataFrame,
    is_numeric: bool,
) -> dict:
    """Summarize local property smoothness without implying mechanism."""
    if nearest_neighbors.empty:
        return {
            "n_neighbor_rows": 0,
            "n_discontinuities": int(len(discontinuities)),
            "note": "No nearest-neighbor rows were generated.",
        }
    similarities = pd.to_numeric(
        nearest_neighbors["tanimoto_similarity"], errors="coerce"
    )
    summary: dict[str, Any] = {
        "n_neighbor_rows": int(len(nearest_neighbors)),
        "n_queries": int(nearest_neighbors["query_structure_index"].nunique()),
        "median_tanimoto_similarity": float(similarities.median()),
        "n_discontinuities": int(len(discontinuities)),
        "discontinuity_fraction_of_neighbor_rows": float(
            len(discontinuities) / len(nearest_neighbors)
        ),
    }
    if is_numeric:
        deltas = pd.to_numeric(
            nearest_neighbors["absolute_property_difference"], errors="coerce"
        ).dropna()
        summary.update(
            {
                "property_comparable_neighbor_rows": int(len(deltas)),
                "median_absolute_property_difference": float(deltas.median())
                if len(deltas)
                else None,
                "mean_absolute_property_difference": float(deltas.mean())
                if len(deltas)
                else None,
            }
        )
    return summary
