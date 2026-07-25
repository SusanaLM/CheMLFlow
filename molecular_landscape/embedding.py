"""Validated PCA, Tanimoto-aware UMAP, and Tanimoto-aware t-SNE embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class EmbeddingResult:
    coordinates: np.ndarray
    diagnostics: Dict[str, object]


def normalize_feature_block(matrix: np.ndarray) -> np.ndarray:
    block = np.asarray(matrix, dtype=np.float64)
    centered = block - block.mean(axis=0, keepdims=True)
    rms_row_norm = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if rms_row_norm <= 1e-12:
        raise ValueError("Cannot normalize a constant feature block.")
    return (centered / rms_row_norm).astype(np.float32)


def weighted_feature_matrix(
    structure_similarity: np.ndarray,
    property_matrix: Optional[np.ndarray],
    property_weight: float,
) -> np.ndarray:
    structure_block = normalize_feature_block(structure_similarity)
    if property_matrix is None or property_matrix.size == 0:
        return structure_block
    if property_weight <= 0.0:
        return structure_block

    property_block = normalize_feature_block(property_matrix)
    if property_weight >= 1.0:
        return property_block
    structure_weight = 1.0 - property_weight
    return np.hstack(
        [
            np.sqrt(structure_weight) * structure_block,
            np.sqrt(property_weight) * property_block,
        ]
    ).astype(np.float32)


def pca_embedding(
    feature_matrix: np.ndarray,
    random_state: int,
) -> EmbeddingResult:
    if len(feature_matrix) < 3:
        raise ValueError("PCA embedding requires at least three rows.")
    solver = "randomized" if min(feature_matrix.shape) > 200 else "full"
    model = PCA(
        n_components=2,
        svd_solver=solver,
        random_state=random_state,
        iterated_power=7 if solver == "randomized" else "auto",
    )
    coordinates = model.fit_transform(feature_matrix).astype(np.float32)
    diagnostics = {
        "method": "pca",
        "explained_variance_ratio": [
            float(value) for value in model.explained_variance_ratio_
        ],
        "cumulative_explained_variance": float(
            model.explained_variance_ratio_.sum()
        ),
        "n_input_features": int(feature_matrix.shape[1]),
        "svd_solver": solver,
    }
    return EmbeddingResult(coordinates, diagnostics)


def normalized_property_distance(property_matrix: np.ndarray) -> np.ndarray:
    distances = squareform(pdist(property_matrix, metric="euclidean")).astype(
        np.float32
    )
    upper = distances[np.triu_indices(len(distances), k=1)]
    positive = upper[upper > 0]
    if not positive.size:
        raise ValueError("Property-distance matrix is constant.")
    scale = float(np.percentile(positive, 95.0))
    scale = max(scale, 1e-12)
    return np.clip(distances / scale, 0.0, 1.0).astype(np.float32)


def fused_distance_matrix(
    structure_distance: np.ndarray,
    property_distance: Optional[np.ndarray],
    property_weight: float,
) -> np.ndarray:
    if property_distance is None or property_distance.size == 0:
        return np.asarray(structure_distance, dtype=np.float32)
    if structure_distance.shape != property_distance.shape:
        raise ValueError("Structure and property distance matrices are misaligned.")
    fused = (
        (1.0 - property_weight) * structure_distance
        + property_weight * property_distance
    )
    np.fill_diagonal(fused, 0.0)
    return fused.astype(np.float32)


def umap_embedding(
    distance_matrix: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> EmbeddingResult:
    if len(distance_matrix) < 4:
        raise ValueError("UMAP embedding requires at least four rows.")
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ImportError(
            "UMAP requested but umap-learn is not installed."
        ) from exc

    effective_neighbors = min(max(2, n_neighbors), len(distance_matrix) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        init="random" if len(distance_matrix) < 10 else "spectral",
        random_state=random_state,
        n_jobs=1,
    )
    coordinates = reducer.fit_transform(distance_matrix).astype(np.float32)
    diagnostics = {
        "method": "umap",
        "metric": "precomputed",
        "n_neighbors": int(effective_neighbors),
        "min_dist": float(min_dist),
    }
    return EmbeddingResult(coordinates, diagnostics)


def tsne_embedding(
    distance_matrix: np.ndarray,
    random_state: int,
    perplexity: float,
) -> EmbeddingResult:
    """t-SNE on the precomputed (Tanimoto or fused) distance matrix.

    Perplexity is clamped below the cohort size so the call remains valid for
    small inputs; precomputed distances require random initialisation.
    """
    if len(distance_matrix) < 4:
        raise ValueError("t-SNE embedding requires at least four rows.")
    n_items = len(distance_matrix)
    effective_perplexity = float(min(perplexity, max(2.0, (n_items - 1) / 3.0)))
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        metric="precomputed",
        init="random",
        random_state=random_state,
    )
    coordinates = reducer.fit_transform(
        np.asarray(distance_matrix, dtype=np.float64)
    ).astype(np.float32)
    diagnostics = {
        "method": "tsne",
        "metric": "precomputed",
        "perplexity": float(effective_perplexity),
    }
    return EmbeddingResult(coordinates, diagnostics)


def pacmap_embedding(
    similarity_matrix: np.ndarray,
    random_state: int,
    n_neighbors: int,
) -> EmbeddingResult:
    """PaCMAP on the Tanimoto-similarity representation.

    PaCMAP consumes a feature matrix rather than a precomputed distance, so it is
    run on the same normalized pairwise-similarity representation used by the
    structure PCA. It balances global and local structure better than UMAP/t-SNE.
    """
    n_items = len(similarity_matrix)
    if n_items < 4:
        raise ValueError("PaCMAP embedding requires at least four rows.")
    try:
        import pacmap
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PaCMAP requested but the 'pacmap' package is not installed."
        ) from exc

    features = normalize_feature_block(similarity_matrix)
    effective_neighbors = min(max(2, n_neighbors), n_items - 1)
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        distance="euclidean",
        apply_pca=True,
        random_state=int(random_state),
        verbose=False,
    )
    coordinates = np.asarray(
        reducer.fit_transform(np.asarray(features, dtype=np.float32), init="pca"),
        dtype=np.float32,
    )
    diagnostics = {
        "method": "pacmap",
        "representation": "tanimoto_similarity",
        "n_neighbors": int(effective_neighbors),
    }
    return EmbeddingResult(coordinates, diagnostics)


def _trimap_neighbor_counts(n_items: int, n_inliers: int | None) -> tuple[int, int, int]:
    """Scale TriMap's neighbour counts so its internal sampling stays valid.

    TriMap's precomputed-distance path samples ``n_inliers + 50`` neighbours, which
    must remain strictly below the cohort size.
    """
    inliers = n_inliers if n_inliers is not None else min(12, max(3, n_items - 52))
    inliers = max(3, min(inliers, n_items - 52))
    outliers = min(4, max(1, inliers // 3))
    randoms = min(3, max(1, inliers // 4))
    return inliers, outliers, randoms


def trimap_embedding(
    distance_matrix: np.ndarray,
    random_state: int,
    n_inliers: int | None = None,
) -> EmbeddingResult:
    """TriMap on the precomputed (Tanimoto or fused) distance matrix.

    Uses TriMap's ``use_dist_matrix`` path for exact Tanimoto consistency with
    UMAP/t-SNE. TriMap has no seed argument, so the global NumPy RNG is set and
    restored around the call to make a given seed reproducible.
    """
    n_items = len(distance_matrix)
    if n_items < 55:
        raise ValueError(
            "TriMap requires at least 55 rows because of its internal +50 "
            "neighbour sampling; use a larger cohort or a different map method."
        )
    try:
        import trimap
    except ModuleNotFoundError as exc:
        raise ImportError(
            "TriMap requested but the 'trimap' package is not installed."
        ) from exc

    inliers, outliers, randoms = _trimap_neighbor_counts(n_items, n_inliers)
    reducer = trimap.TRIMAP(
        n_dims=2,
        n_inliers=inliers,
        n_outliers=outliers,
        n_random=randoms,
        use_dist_matrix=True,
        verbose=False,
    )
    rng_state = np.random.get_state()
    np.random.seed(int(random_state))
    try:
        coordinates = np.asarray(
            reducer.fit_transform(np.asarray(distance_matrix, dtype=np.float32)),
            dtype=np.float32,
        )
    finally:
        np.random.set_state(rng_state)
    diagnostics = {
        "method": "trimap",
        "metric": "precomputed",
        "n_inliers": int(inliers),
        "n_outliers": int(outliers),
        "n_random": int(randoms),
    }
    return EmbeddingResult(coordinates, diagnostics)


def _neighbor_order_without_self(distances: np.ndarray, k: int) -> np.ndarray:
    """Return stable nearest-neighbour indices after excluding the diagonal."""
    ranking_distances = np.asarray(distances, dtype=float).copy()
    np.fill_diagonal(ranking_distances, np.inf)
    return np.argsort(ranking_distances, axis=1, kind="stable")[:, :k]


def distance_preservation_diagnostics(
    source_distances: np.ndarray,
    coordinates: np.ndarray,
    n_neighbors: int,
    random_state: int,
    max_pairs: int = 200_000,
) -> Dict[str, float]:
    coordinate_distances = squareform(pdist(coordinates, metric="euclidean"))
    upper_indices = np.triu_indices(len(source_distances), k=1)
    source_upper = source_distances[upper_indices]
    coordinate_upper = coordinate_distances[upper_indices]

    if len(source_upper) > max_pairs:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(len(source_upper), size=max_pairs, replace=False)
        source_upper = source_upper[selected]
        coordinate_upper = coordinate_upper[selected]

    spearman = spearmanr(source_upper, coordinate_upper).statistic
    pearson = pearsonr(source_upper, coordinate_upper).statistic

    n_items = len(source_distances)
    max_rank_metric_k = max(1, (2 * n_items - 2) // 3)
    k = min(max(2, n_neighbors), n_items - 1, max_rank_metric_k)
    # Exclude self explicitly. With tied zero distances (for example folded
    # fingerprint collisions), self is not guaranteed to sort first.
    source_for_ranking = np.asarray(source_distances, dtype=float).copy()
    coordinate_for_ranking = np.asarray(coordinate_distances, dtype=float).copy()
    np.fill_diagonal(source_for_ranking, np.inf)
    np.fill_diagonal(coordinate_for_ranking, np.inf)
    source_order = np.argsort(source_for_ranking, axis=1, kind="stable")
    coordinate_order = np.argsort(coordinate_for_ranking, axis=1, kind="stable")
    source_neighbors = _neighbor_order_without_self(source_distances, k)
    coordinate_neighbors = _neighbor_order_without_self(coordinate_distances, k)
    recalls = [
        len(set(source_neighbors[idx]).intersection(coordinate_neighbors[idx])) / k
        for idx in range(len(source_distances))
    ]
    source_ranks = np.empty_like(source_order)
    coordinate_ranks = np.empty_like(coordinate_order)
    ranks = np.arange(1, n_items + 1)
    source_ranks[np.arange(n_items)[:, None], source_order] = ranks
    coordinate_ranks[np.arange(n_items)[:, None], coordinate_order] = ranks
    source_ranks[np.arange(n_items), np.arange(n_items)] = 0
    coordinate_ranks[np.arange(n_items), np.arange(n_items)] = 0
    trust_penalty = 0.0
    continuity_penalty = 0.0
    for idx in range(n_items):
        source_set = set(source_neighbors[idx])
        coordinate_set = set(coordinate_neighbors[idx])
        trust_penalty += sum(
            int(source_ranks[idx, item]) - k for item in coordinate_set - source_set
        )
        continuity_penalty += sum(
            int(coordinate_ranks[idx, item]) - k
            for item in source_set - coordinate_set
        )
    normalizer = 2.0 / (n_items * k * (2 * n_items - 3 * k - 1))
    return {
        "distance_spearman": float(spearman),
        "distance_pearson": float(pearson),
        f"knn_recall_at_{k}": float(np.mean(recalls)),
        f"trustworthiness_at_{k}": float(1.0 - normalizer * trust_penalty),
        f"continuity_at_{k}": float(1.0 - normalizer * continuity_penalty),
    }


def property_coordinate_correlations(
    coordinates: np.ndarray,
    property_df,
) -> Dict[str, Dict[str, Optional[float]]]:
    correlations: Dict[str, Dict[str, Optional[float]]] = {}

    def correlation(left: np.ndarray, right: np.ndarray, method: str) -> Optional[float]:
        if float(np.ptp(left)) <= 1e-12 or float(np.ptp(right)) <= 1e-12:
            return None
        value = (
            pearsonr(left, right).statistic
            if method == "pearson"
            else spearmanr(left, right).statistic
        )
        return None if not np.isfinite(value) else float(value)

    for name in property_df.columns:
        values = np.asarray(property_df[name], dtype=float)
        correlations[name] = {
            "component_1_pearson": correlation(coordinates[:, 0], values, "pearson"),
            "component_2_pearson": correlation(coordinates[:, 1], values, "pearson"),
            "component_1_spearman": correlation(coordinates[:, 0], values, "spearman"),
            "component_2_spearman": correlation(coordinates[:, 1], values, "spearman"),
        }
    return correlations


def umap_seed_stability(
    distance_matrix: np.ndarray,
    reference_coordinates: np.ndarray,
    reference_seed: int,
    seeds: list[int],
    n_neighbors: int,
    min_dist: float,
) -> list[Dict[str, float | int]]:
    """Quantify UMAP layout sensitivity after removing rotation/scale ambiguity."""
    reference_pairwise = pdist(reference_coordinates, metric="euclidean")
    rows: list[Dict[str, float | int]] = []
    for seed in sorted(set([reference_seed, *seeds])):
        coordinates = (
            reference_coordinates
            if seed == reference_seed
            else umap_embedding(distance_matrix, seed, n_neighbors, min_dist).coordinates
        )
        _, _, disparity = procrustes(reference_coordinates, coordinates)
        pairwise_correlation = spearmanr(
            reference_pairwise,
            pdist(coordinates, metric="euclidean"),
        ).statistic
        rows.append(
            {
                "seed": int(seed),
                "procrustes_disparity_to_default": float(disparity),
                "pairwise_distance_spearman_to_default": float(pairwise_correlation),
            }
        )
    return rows


def tsne_seed_stability(
    distance_matrix: np.ndarray,
    reference_coordinates: np.ndarray,
    reference_seed: int,
    seeds: list[int],
    perplexity: float,
) -> list[Dict[str, float | int]]:
    """Quantify t-SNE layout sensitivity across seeds (t-SNE is stochastic)."""
    return _layout_seed_stability(
        lambda seed: tsne_embedding(distance_matrix, seed, perplexity).coordinates,
        reference_coordinates,
        reference_seed,
        seeds,
    )


def _layout_seed_stability(
    embed_for_seed,
    reference_coordinates: np.ndarray,
    reference_seed: int,
    seeds: list[int],
) -> list[Dict[str, float | int]]:
    """Generic seed-stability: Procrustes disparity + pairwise-distance rank corr."""
    reference_pairwise = pdist(reference_coordinates, metric="euclidean")
    rows: list[Dict[str, float | int]] = []
    for seed in sorted(set([reference_seed, *seeds])):
        coordinates = (
            reference_coordinates
            if seed == reference_seed
            else embed_for_seed(seed)
        )
        _, _, disparity = procrustes(reference_coordinates, coordinates)
        pairwise_correlation = spearmanr(
            reference_pairwise,
            pdist(coordinates, metric="euclidean"),
        ).statistic
        rows.append(
            {
                "seed": int(seed),
                "procrustes_disparity_to_default": float(disparity),
                "pairwise_distance_spearman_to_default": float(pairwise_correlation),
            }
        )
    return rows


def pacmap_seed_stability(
    similarity_matrix: np.ndarray,
    reference_coordinates: np.ndarray,
    reference_seed: int,
    seeds: list[int],
    n_neighbors: int,
) -> list[Dict[str, float | int]]:
    """Quantify PaCMAP layout sensitivity across seeds (PaCMAP is stochastic)."""
    return _layout_seed_stability(
        lambda seed: pacmap_embedding(similarity_matrix, seed, n_neighbors).coordinates,
        reference_coordinates,
        reference_seed,
        seeds,
    )


def trimap_seed_stability(
    distance_matrix: np.ndarray,
    reference_coordinates: np.ndarray,
    reference_seed: int,
    seeds: list[int],
) -> list[Dict[str, float | int]]:
    """Quantify TriMap layout sensitivity across seeds (TriMap is stochastic)."""
    return _layout_seed_stability(
        lambda seed: trimap_embedding(distance_matrix, seed).coordinates,
        reference_coordinates,
        reference_seed,
        seeds,
    )


def coranking_metrics(
    source_distances: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[Dict[str, float | int], Dict[str, np.ndarray]]:
    """Parameter-free, multi-scale neighbourhood-preservation from the co-ranking matrix.

    Returns the baseline-corrected quality R_NX(K) summarised by its area under the
    log-K curve (AUC_R_NX in [0, 1], 0 = random, 1 = perfect), the local-continuity
    meta-criterion peak (LCMC), and the local/global quality split (Lee & Verleysen).
    Strictly more informative than a single-k trustworthiness/continuity.
    """
    source = np.asarray(source_distances, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n = len(source)
    if n < 4:
        raise ValueError("Co-ranking metrics require at least four rows.")
    coordinate_distances = squareform(pdist(coords))
    # Force every point to be its own nearest neighbour (rank 0) so ties at
    # distance 0 (e.g. identical fingerprints) cannot displace self and create a
    # negative co-rank index.
    source = source.copy()
    coordinate_distances = coordinate_distances.copy()
    np.fill_diagonal(source, -1.0)
    np.fill_diagonal(coordinate_distances, -1.0)
    source_order = np.argsort(source, axis=1, kind="stable")
    coord_order = np.argsort(coordinate_distances, axis=1, kind="stable")
    rows = np.arange(n)[:, None]
    source_rank = np.empty((n, n), dtype=np.int64)
    coord_rank = np.empty((n, n), dtype=np.int64)
    source_rank[rows, source_order] = np.arange(n)
    coord_rank[rows, coord_order] = np.arange(n)

    off_diagonal = ~np.eye(n, dtype=bool)
    high = source_rank[off_diagonal] - 1  # ranks 1..n-1 -> 0..n-2
    low = coord_rank[off_diagonal] - 1
    coranking = (
        np.bincount(high * (n - 1) + low, minlength=(n - 1) ** 2)
        .reshape(n - 1, n - 1)
        .astype(np.float64)
    )
    upper_left = coranking.cumsum(axis=0).cumsum(axis=1)
    neighbourhoods = np.arange(1, n)  # K = 1..n-1
    q_nx = upper_left[neighbourhoods - 1, neighbourhoods - 1] / (neighbourhoods * n)

    k_rnx = np.arange(1, n - 1)  # K = 1..n-2
    r_nx = ((n - 1) * q_nx[: n - 2] - k_rnx) / (n - 1 - k_rnx)
    auc = float(np.sum(r_nx / k_rnx) / np.sum(1.0 / k_rnx))
    lcmc = q_nx - neighbourhoods / (n - 1)
    k_max = int(np.argmax(lcmc)) + 1

    summary: Dict[str, float | int] = {
        "coranking_auc_rnx": auc,
        "lcmc_kmax": k_max,
        "lcmc_max": float(lcmc[k_max - 1]),
        "q_local": float(np.mean(r_nx[:k_max])) if k_max >= 1 else float("nan"),
        "q_global": float(np.mean(r_nx[k_max:])) if k_max < len(r_nx) else float("nan"),
        "n_used": int(n),
    }
    curve = {
        "k": k_rnx,
        "q_nx": q_nx[: n - 2],
        "r_nx": r_nx,
        "lcmc": lcmc[: n - 2],
    }
    return summary, curve


def advanced_map_diagnostics(
    source_distances: np.ndarray,
    coordinates: np.ndarray,
    n_neighbors: int,
    random_state: int,
    max_points: int = 3000,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Co-ranking quality plus a random-layout baseline for context.

    Subsamples to ``max_points`` for tractable O(n^2) co-ranking on large cohorts.
    The random baseline shuffles the same coordinates, so the floor each metric is
    compared against is reported alongside it.
    """
    source = np.asarray(source_distances, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    n_full = len(source)
    if n_full > max_points:
        rng = np.random.default_rng(random_state)
        index = np.sort(rng.choice(n_full, size=max_points, replace=False))
        source = source[np.ix_(index, index)]
        coords = coords[index]

    summary, curve = coranking_metrics(source, coords)
    actual_preservation = distance_preservation_diagnostics(
        source, coords, n_neighbors=n_neighbors, random_state=random_state
    )

    shuffle = np.random.default_rng(random_state + 1).permutation(len(coords))
    baseline_summary, _ = coranking_metrics(source, coords[shuffle])
    baseline_preservation = distance_preservation_diagnostics(
        source, coords[shuffle], n_neighbors=n_neighbors, random_state=random_state
    )

    summary["preservation"] = actual_preservation
    summary["random_baseline"] = {
        "coranking_auc_rnx": baseline_summary["coranking_auc_rnx"],
        **baseline_preservation,
    }
    return summary, curve
