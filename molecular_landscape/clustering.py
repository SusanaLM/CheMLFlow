"""Chemically interpretable Butina clustering and diagnostics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


def butina_labels(
    fingerprints,
    similarity_threshold: float,
) -> Tuple[np.ndarray, Tuple[Tuple[int, ...], ...]]:
    condensed_distances: List[float] = []
    for idx in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[idx],
            fingerprints[:idx],
        )
        condensed_distances.extend(1.0 - value for value in similarities)

    clusters = Butina.ClusterData(
        condensed_distances,
        len(fingerprints),
        1.0 - similarity_threshold,
        isDistData=True,
        reordering=True,
    )
    labels = np.empty(len(fingerprints), dtype=np.int32)
    for cluster_id, members in enumerate(clusters):
        labels[list(members)] = cluster_id
    return labels, clusters


def cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    property_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for cluster_id in sorted(np.unique(labels).tolist()):
        members = np.where(labels == cluster_id)[0]
        sub_distances = distance_matrix[np.ix_(members, members)]
        mean_distances = sub_distances.mean(axis=1)
        medoid_local = int(np.argmin(mean_distances))
        medoid_idx = int(members[medoid_local])
        upper = sub_distances[np.triu_indices(len(members), k=1)]

        row: Dict[str, object] = {
            "cluster_id": int(cluster_id),
            "size": int(len(members)),
            "medoid_source_row": int(df.iloc[medoid_idx]["_source_row"]),
            "medoid_compound_id": df.iloc[medoid_idx]["_compound_id"],
            "medoid_smiles": df.iloc[medoid_idx]["_canonical_smiles"],
            "minimum_pairwise_similarity": (
                1.0 - float(upper.max()) if upper.size else 1.0
            ),
            "median_pairwise_similarity": (
                1.0 - float(np.median(upper)) if upper.size else 1.0
            ),
            "mean_pairwise_similarity": (
                1.0 - float(upper.mean()) if upper.size else 1.0
            ),
        }
        for name in property_cols:
            values = pd.to_numeric(df.iloc[members][name], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            row[f"{name}_count"] = int(values.count())
            row[f"{name}_mean"] = (
                None if values.dropna().empty else float(values.mean())
            )
            row[f"{name}_std"] = (
                None if values.dropna().empty else float(values.std(ddof=0))
            )
            row[f"{name}_min"] = (
                None if values.dropna().empty else float(values.min())
            )
            row[f"{name}_max"] = (
                None if values.dropna().empty else float(values.max())
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["size", "cluster_id"],
        ascending=[False, True],
    )


def hdbscan_clustering(
    distance_matrix: np.ndarray,
    min_cluster_size: int,
    butina_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Density-based, threshold-free clustering on the precomputed Tanimoto distance.

    HDBSCAN needs no similarity cut-off and labels structurally isolated molecules
    as noise (label -1). Agreement with the Butina partition is reported as the
    adjusted Rand index, so the two groupings can be compared rather than assumed
    interchangeable.
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score

    n_items = len(distance_matrix)
    effective_min = max(2, min(int(min_cluster_size), n_items))
    model = HDBSCAN(min_cluster_size=effective_min, metric="precomputed")
    labels = model.fit_predict(np.asarray(distance_matrix, dtype=np.float64)).astype(
        np.int32
    )
    n_noise = int((labels == -1).sum())
    summary = {
        "min_cluster_size": int(effective_min),
        "n_clusters": int(len({int(label) for label in labels} - {-1})),
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / n_items) if n_items else 0.0,
        "adjusted_rand_index_to_butina": float(
            adjusted_rand_score(butina_labels, labels)
        ),
    }
    return labels, summary


def threshold_sensitivity(
    fingerprints,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows = []
    for threshold in sorted(set(thresholds)):
        labels, clusters = butina_labels(fingerprints, threshold)
        sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.int32)
        rows.append(
            {
                "similarity_threshold": float(threshold),
                "n_clusters": int(len(clusters)),
                "n_singletons": int(np.sum(sizes == 1)),
                "singleton_fraction": float(np.mean(sizes == 1)),
                "largest_cluster_size": int(sizes.max()),
                "largest_cluster_fraction": float(sizes.max() / len(labels)),
                "median_cluster_size": float(np.median(sizes)),
            }
        )
    return pd.DataFrame(rows)
