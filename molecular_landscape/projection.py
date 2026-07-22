"""Scalable, inductive chemical-space projection.

This is an additive path that complements (does not replace) the exact-pairwise
workflow. It fits a feature-based UMAP on Morgan bit vectors using the Jaccard
metric, which is exactly Tanimoto distance on binary fingerprints, via UMAP's
approximate nearest-neighbour search. Consequences:

* No O(n^2) distance matrix is built, so it scales well past the exact-pairwise
  cap used by the main workflow.
* The fitted reducer supports ``transform`` for **out-of-sample** projection of
  new molecules onto an existing map (applicability-domain visualisation, or
  projecting a held-out / scaffold split onto the training map).

Because no full distance matrix exists at scale, preservation is reported on a
random sample of pairs and points using exact Tanimoto, keeping the diagnostics
honest without materialising the matrix. The exact-pairwise workflow remains the
authoritative, fully-diagnosed analysis (clustering, property-aware geometry,
full sensitivity) for cohorts within its limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from rdkit import DataStructs
from scipy.stats import pearsonr, spearmanr


@dataclass
class FittedProjection:
    reducer: Any
    coordinates: np.ndarray
    diagnostics: Dict[str, Any]


def fit_reference_projection(
    feature_matrix: np.ndarray,
    *,
    random_state: int,
    n_neighbors: int = 30,
    min_dist: float = 0.10,
    metric: str = "jaccard",
) -> FittedProjection:
    """Fit a feature-based UMAP supporting out-of-sample transform.

    ``metric="jaccard"`` on Morgan bit vectors is exactly Tanimoto; ``metric="cosine"``
    suits dense learned/foundation-model embeddings. Either way no NxN matrix is built
    and the fitted reducer can project new molecules.
    """
    matrix = np.asarray(feature_matrix, dtype=np.float32)
    n_items = len(matrix)
    if n_items < 4:
        raise ValueError("Reference projection requires at least four molecules.")
    try:
        import umap
    except ModuleNotFoundError as exc:  # pragma: no cover - umap is a core dependency
        raise ImportError("umap-learn is required for scalable projection.") from exc

    effective_neighbors = min(max(2, n_neighbors), n_items - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=1,
    )
    coordinates = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
    diagnostics = {
        "method": "umap",
        "metric": metric,
        "approximate_neighbors": True,
        "n_neighbors": int(effective_neighbors),
        "min_dist": float(min_dist),
        "n_reference": int(n_items),
        "n_features": int(matrix.shape[1]),
    }
    return FittedProjection(reducer, coordinates, diagnostics)


def embedding_applicability_domain(
    reference_matrix: np.ndarray,
    query_matrix: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """Mean top-k cosine similarity of each query embedding to the reference set.

    The dense-embedding analogue of the Tanimoto applicability domain: high means the
    query lies within the learned chemical space the map was fit on.
    """
    reference = np.asarray(reference_matrix, dtype=np.float64)
    query = np.asarray(query_matrix, dtype=np.float64)
    if query.size == 0:
        return np.empty((0,), dtype=np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12
    query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-12
    similarities = query @ reference.T
    top_k = max(1, min(k, reference.shape[0]))
    top = np.sort(similarities, axis=1)[:, -top_k:]
    return top.mean(axis=1).astype(np.float32)


def project_query(
    fitted: FittedProjection,
    query_fingerprint_matrix: np.ndarray,
) -> np.ndarray:
    """Project new molecules onto the fitted reference map (out-of-sample)."""
    matrix = np.asarray(query_fingerprint_matrix, dtype=np.float32)
    if matrix.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(fitted.reducer.transform(matrix), dtype=np.float32)


def applicability_domain(
    reference_fingerprints: Sequence[Any],
    query_fingerprints: Sequence[Any],
    k: int = 1,
) -> np.ndarray:
    """Mean of the top-k Tanimoto similarities of each query to the reference set.

    A high value means the query lies inside the chemical space the map was fit on;
    a low value flags an out-of-domain compound. Computed pairwise without an
    n_reference x n_reference matrix.
    """
    reference = list(reference_fingerprints)
    top_k = max(1, min(k, len(reference)))
    scores = []
    for fingerprint in query_fingerprints:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, reference)
        scores.append(
            float(np.mean(sorted(similarities, reverse=True)[:top_k]))
        )
    return np.asarray(scores, dtype=np.float32)


def sampled_preservation_diagnostics(
    reference_fingerprints: Sequence[Any],
    coordinates: np.ndarray,
    *,
    random_state: int,
    n_pairs: int = 200_000,
    n_recall_points: int = 300,
    recall_k: int = 15,
) -> Dict[str, float]:
    """Honest, scale-friendly preservation: sampled pairwise distances + sampled kNN recall."""
    reference = list(reference_fingerprints)
    n_items = len(reference)
    rng = np.random.default_rng(random_state)

    pair_budget = min(n_pairs, n_items * (n_items - 1) // 2)
    left = rng.integers(0, n_items, size=pair_budget)
    right = rng.integers(0, n_items, size=pair_budget)
    keep = left != right
    left, right = left[keep], right[keep]
    tanimoto_distance = np.fromiter(
        (
            1.0 - DataStructs.TanimotoSimilarity(reference[int(a)], reference[int(b)])
            for a, b in zip(left, right)
        ),
        dtype=np.float64,
        count=len(left),
    )
    embedded_distance = np.linalg.norm(
        coordinates[left] - coordinates[right], axis=1
    ).astype(np.float64)

    spearman = spearmanr(tanimoto_distance, embedded_distance).statistic
    pearson = pearsonr(tanimoto_distance, embedded_distance).statistic

    sample = rng.choice(n_items, size=min(n_recall_points, n_items), replace=False)
    effective_k = max(1, min(recall_k, n_items - 1))
    recalls = []
    for idx in sample:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(reference[int(idx)], reference),
            dtype=np.float32,
        )
        similarities[idx] = -1.0
        true_neighbors = set(np.argsort(similarities)[::-1][:effective_k].tolist())
        embedded = np.linalg.norm(coordinates - coordinates[idx], axis=1)
        embedded[idx] = np.inf
        embedded_neighbors = set(np.argsort(embedded)[:effective_k].tolist())
        recalls.append(len(true_neighbors & embedded_neighbors) / effective_k)

    return {
        "sampled_distance_spearman": float(spearman),
        "sampled_distance_pearson": float(pearson),
        f"sampled_knn_recall_at_{effective_k}": float(np.mean(recalls)),
        "n_sampled_pairs": int(len(left)),
        "n_recall_points": int(len(sample)),
    }


def fingerprint_matrix(fingerprints: Sequence[Any], n_bits: int) -> np.ndarray:
    """Dense float32 bit matrix from RDKit fingerprints (for UMAP features)."""
    matrix = np.zeros((len(fingerprints), n_bits), dtype=np.float32)
    for index, fingerprint in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix
