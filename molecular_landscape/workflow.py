"""End-to-end standalone molecular landscape workflow."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import procrustes

from . import __version__
from .chemistry import (
    assign_scaffold_families,
    build_fingerprints,
    descriptor_audit,
    fingerprint_collision_summary,
    identity_audit,
    molecular_descriptor_frame,
    parse_molecules,
    representation_sensitivity,
    tanimoto_distance_matrix,
)
from .clustering import (
    butina_labels,
    cluster_summary,
    hdbscan_clustering,
    threshold_sensitivity,
)
from .config import WorkflowConfig
from .embedding import (
    EmbeddingResult,
    advanced_map_diagnostics,
    distance_preservation_diagnostics,
    fused_distance_matrix,
    normalized_property_distance,
    pacmap_embedding,
    pacmap_seed_stability,
    pca_embedding,
    property_coordinate_correlations,
    trimap_embedding,
    trimap_seed_stability,
    tsne_embedding,
    tsne_seed_stability,
    umap_embedding,
    umap_seed_stability,
    weighted_feature_matrix,
)
from .io_utils import (
    atomic_output_directory,
    dependency_versions,
    public_invocation,
    redact_host_paths,
    safe_filename_token,
    sha256_file,
    write_artifact_manifest,
    write_json,
)
from .plotting import plot_cluster_map, plot_property_map, plot_shepard
from .properties import PropertyCohort, prepare_property_cohort
from .schema import SchemaSelection, detect_schema


def _merge_exclusions(frames: List[pd.DataFrame]) -> pd.DataFrame:
    populated = [frame for frame in frames if frame is not None and not frame.empty]
    if not populated:
        return pd.DataFrame(
            columns=["source_row", "compound_id", "exclusion_stage", "reason"]
        )
    return pd.concat(populated, ignore_index=True).sort_values(
        ["source_row", "exclusion_stage"]
    )


def _validate_selected_transforms(
    config: WorkflowConfig,
    schema: SchemaSelection,
) -> None:
    unused = sorted(set(config.property_transforms).difference(schema.property_cols))
    if unused:
        raise ValueError(
            "Property transforms were provided for unselected columns: "
            f"{unused}"
        )


def _map_diagnostics(
    result: EmbeddingResult,
    source_distances: np.ndarray,
    property_df: pd.DataFrame | None,
    config: WorkflowConfig,
    geometry_inputs: Dict[str, Any],
    descriptor_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    diagnostics = dict(result.diagnostics)
    diagnostics["geometry_inputs"] = geometry_inputs
    diagnostics["distance_preservation"] = distance_preservation_diagnostics(
        source_distances,
        result.coordinates,
        n_neighbors=config.embedding.validation_neighbors,
        random_state=config.embedding.random_state,
    )
    if property_df is not None and not property_df.empty:
        diagnostics["property_coordinate_correlations"] = (
            property_coordinate_correlations(result.coordinates, property_df)
        )
    if descriptor_df is not None and not descriptor_df.empty:
        diagnostics["descriptor_coordinate_correlations"] = (
            property_coordinate_correlations(result.coordinates, descriptor_df)
        )
    return diagnostics


def _embed_structure(
    similarity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    descriptor_df: pd.DataFrame,
    config: WorkflowConfig,
) -> Tuple[Dict[str, EmbeddingResult], Dict[str, Dict[str, Any]]]:
    pca_result = pca_embedding(
        weighted_feature_matrix(similarity_matrix, None, property_weight=0.0),
        random_state=config.embedding.random_state,
    )
    umap_result = umap_embedding(
        distance_matrix,
        random_state=config.embedding.random_state,
        n_neighbors=config.embedding.umap_neighbors,
        min_dist=config.embedding.umap_min_dist,
    )
    results = {"pca": pca_result, "umap": umap_result}
    if config.embedding.include_tsne:
        results["tsne"] = tsne_embedding(
            distance_matrix,
            random_state=config.embedding.random_state,
            perplexity=config.embedding.tsne_perplexity,
        )
    if config.embedding.include_pacmap:
        results["pacmap"] = pacmap_embedding(
            similarity_matrix,
            random_state=config.embedding.random_state,
            n_neighbors=config.embedding.pacmap_neighbors,
        )
    if config.embedding.include_trimap:
        results["trimap"] = trimap_embedding(
            distance_matrix,
            random_state=config.embedding.random_state,
        )
    diagnostics = {
        method: _map_diagnostics(
            result,
            source_distances=distance_matrix,
            property_df=None,
            config=config,
            descriptor_df=descriptor_df,
            geometry_inputs={
                "structure": "full pairwise chirality-aware Morgan Tanimoto",
                "property_in_geometry": False,
            },
        )
        for method, result in results.items()
    }
    return results, diagnostics


def _embed_property_weight(
    structure_distance: np.ndarray,
    property_distance: np.ndarray,
    property_df: pd.DataFrame,
    descriptor_df: pd.DataFrame,
    property_weight: float,
    config: WorkflowConfig,
) -> Tuple[Dict[str, EmbeddingResult], Dict[str, Dict[str, Any]], np.ndarray]:
    fused_distance = fused_distance_matrix(
        structure_distance,
        property_distance,
        property_weight,
    )
    pca_result = pca_embedding(
        weighted_feature_matrix(
            1.0 - fused_distance,
            None,
            property_weight=0.0,
        ),
        random_state=config.embedding.random_state,
    )
    umap_result = umap_embedding(
        fused_distance,
        random_state=config.embedding.random_state,
        n_neighbors=config.embedding.umap_neighbors,
        min_dist=config.embedding.umap_min_dist,
    )
    results = {"pca": pca_result, "umap": umap_result}
    if config.embedding.include_tsne:
        results["tsne"] = tsne_embedding(
            fused_distance,
            random_state=config.embedding.random_state,
            perplexity=config.embedding.tsne_perplexity,
        )
    if config.embedding.include_pacmap:
        results["pacmap"] = pacmap_embedding(
            1.0 - fused_distance,
            random_state=config.embedding.random_state,
            n_neighbors=config.embedding.pacmap_neighbors,
        )
    if config.embedding.include_trimap:
        results["trimap"] = trimap_embedding(
            fused_distance,
            random_state=config.embedding.random_state,
        )
    geometry_inputs = {
        "structure": "full pairwise chirality-aware Morgan Tanimoto",
        "property_in_geometry": True,
        "property_weight": float(property_weight),
        "structure_weight": float(1.0 - property_weight),
        "pca_methodology": (
            "PCA on the similarity representation derived from the same "
            "explicitly weighted fused distance used for property-aware UMAP."
        ),
        "umap_methodology": (
            "UMAP on weighted sum of Tanimoto distance and standardized "
            "property distance normalized at its 95th percentile."
        ),
    }
    diagnostics = {
        method: _map_diagnostics(
            result,
            source_distances=fused_distance,
            property_df=property_df,
            config=config,
            descriptor_df=descriptor_df,
            geometry_inputs=geometry_inputs,
        )
        for method, result in results.items()
    }
    return results, diagnostics, fused_distance


def _flatten_sensitivity_row(
    weight: float,
    method: str,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    preservation = diagnostics["distance_preservation"]
    row = {
        "property_weight": float(weight),
        "structure_weight": float(1.0 - weight),
        "method": method,
    }
    row.update(preservation)
    for property_name, correlations in diagnostics[
        "property_coordinate_correlations"
    ].items():
        row.update(
            {
                f"{property_name}_{key}": value
                for key, value in correlations.items()
            }
        )
    return row


def _plot_outputs(
    staging: Path,
    structure_df: pd.DataFrame,
    property_cohort: PropertyCohort,
    structure_results: Dict[str, EmbeddingResult],
    structure_diagnostics: Dict[str, Dict[str, Any]],
    property_results: Dict[str, EmbeddingResult] | None,
    property_diagnostics: Dict[str, Dict[str, Any]] | None,
    property_cols: List[str],
) -> Dict[str, str]:
    plot_dir = staging / "figures"
    for method, result in structure_results.items():
        plot_cluster_map(
            result.coordinates,
            structure_df["butina_cluster_id"].to_numpy(),
            title=f"Structure-only chemical space ({method.upper()})",
            method=method,
            diagnostics=structure_diagnostics[method],
            output_stem=plot_dir / f"structure_only_{method}",
        )

    if not property_cols:
        return {}

    property_indices = property_cohort.df["_structure_index"].to_numpy(dtype=int)
    filename_map = {
        property_name: safe_filename_token(property_name)
        for property_name in property_cols
    }
    for property_name in property_cols:
        property_token = filename_map[property_name]
        color_limits = (
            float(property_cohort.df[property_name].min()),
            float(property_cohort.df[property_name].max()),
        )
        for method, result in structure_results.items():
            plot_property_map(
                result.coordinates[property_indices],
                property_cohort.df[property_name].to_numpy(),
                property_name=property_name,
                title=f"Structure geometry colored by {property_name} ({method.upper()})",
                method=method,
                diagnostics=structure_diagnostics[method],
                output_stem=(
                    plot_dir
                    / f"activity_colored_structure_{property_token}_{method}"
                ),
                color_limits=color_limits,
            )

        assert property_results is not None
        assert property_diagnostics is not None
        for method, result in property_results.items():
            plot_property_map(
                result.coordinates,
                property_cohort.df[property_name].to_numpy(),
                property_name=property_name,
                title=f"Property-aware chemical space: {property_name} ({method.upper()})",
                method=method,
                diagnostics=property_diagnostics[method],
                output_stem=plot_dir / f"property_aware_{property_token}_{method}",
                color_limits=color_limits,
            )
    return filename_map


def _write_run_summary(
    path: Path,
    schema: SchemaSelection,
    counts: Dict[str, int],
    config: WorkflowConfig,
) -> None:
    property_text = ", ".join(schema.property_cols) if schema.property_cols else "none"
    text = f"""# Molecular Landscape Run Summary

## Scientific Contract

- Structure-only geometry uses chirality-aware Morgan Tanimoto similarity.
- Activity-colored maps reuse structure-only geometry; activity does not affect
  point positions.
- Property-aware maps include the selected properties in their geometry with
  property weight `{config.embedding.property_weight:.2f}`.
- Property-aware map sensitivity is evaluated at weights
  `{", ".join(f"{weight:.2f}" for weight in config.embedding.property_weight_sensitivity)}`;
  inspect `diagnostics/property_weight_sensitivity.csv` before interpreting
  supervised-map topology.
- Butina clusters use similarity threshold
  `{config.clustering.butina_similarity_threshold:.2f}`.
- Missing, non-numeric, flagged, or censored activity records are excluded
  from property-dependent analyses and retained in the exclusion audit.
- Missing identifiers and duplicate identifiers/canonical structures are
  reported, not silently removed.

## Selected Schema

- SMILES: `{schema.smiles_col}`
- Identifier: `{schema.id_col}`
- Properties: `{property_text}`
- Property detection: `{schema.property_detection}`

## Cohort Counts

- Input rows: {counts['input_rows']}
- Sampled rows: {counts['sampled_rows']}
- Valid structure rows: {counts['structure_rows']}
- Property-cohort rows: {counts['property_rows']}
- Excluded rows/events: {counts['exclusion_rows']}

## Interpretation

Property-aware geometry is explicitly supervised by the selected properties.
Visual gradients or separation in those maps are not independent evidence of
structure-activity relationships. Use the structure-only geometry colored by
activity for the least circular visual assessment of SAR. Treat a
property-aware neighborhood as robust only when it remains interpretable
across the recorded nearby-weight sensitivity analyses.
"""
    path.write_text(text, encoding="utf-8")


def _write_methods_and_captions(
    path: Path,
    schema: SchemaSelection,
    config: WorkflowConfig,
    filename_map: Dict[str, str],
) -> None:
    property_text = ", ".join(f"`{name}`" for name in schema.property_cols) or "none"
    method_names = ["pca", "umap"]
    if config.embedding.include_tsne:
        method_names.append("tsne")
    if config.embedding.include_pacmap:
        method_names.append("pacmap")
    if config.embedding.include_trimap:
        method_names.append("trimap")
    methods = ",".join(method_names)
    captions = []
    for property_name in schema.property_cols:
        token = filename_map[property_name]
        captions.append(
            f"- `activity_colored_structure_{token}_{{{methods}}}`: structure-only "
            f"geometry colored by `{property_name}`; the property does not influence "
            "point positions."
        )
        captions.append(
            f"- `property_aware_{token}_{{{methods}}}`: explicitly supervised geometry "
            f"combining structure and `{property_name}` at the configured property "
            "weight; this is not independent evidence of SAR."
        )
    text = f"""# Reproducible Methods And Figure Captions

## Methods

RDKit-valid molecules were represented using chirality-aware Morgan bit
fingerprints (radius {config.fingerprint.radius}, {config.fingerprint.n_bits}
bits). Exact pairwise Tanimoto distances defined structural dissimilarity.
Structure-only PCA was fit to the normalized pairwise-similarity
representation, while structure-only UMAP used the precomputed Tanimoto
distance matrix. Selected properties ({property_text}) were standardized after
the recorded transforms. Property-aware geometry used a weighted sum of
Tanimoto distance and normalized property distance with property weight
{config.embedding.property_weight:.2f}.

Butina clusters used a Tanimoto similarity operating point of
{config.clustering.butina_similarity_threshold:.2f}; this threshold does not
imply that every within-cluster pair exceeds the operating point. Bemis-Murcko
scaffolds were reported separately. Map validation includes global distance
correlations, k-nearest-neighbor recall, trustworthiness, continuity,
descriptor-coordinate correlations, property-coordinate correlations where
applicable, property-weight sensitivity, and UMAP seed sensitivity.

## Figure Captions

- `structure_only_{{{methods}}}`: chemical-space geometry determined only by
  molecular structure and colored by Butina cluster membership.
{chr(10).join(captions) if captions else "- No property-dependent figures were generated."}

## Required Interpretation

PCA and UMAP are complementary projections, not interchangeable evidence.
UMAP axes have no direct chemical meaning. Property-aware maps are supervised
visual summaries and must be interpreted alongside their weight- and
seed-sensitivity diagnostics. The exact software environment, input checksum,
configuration, warnings, and artifact checksums are recorded with the run.
"""
    path.write_text(text, encoding="utf-8")


def _map_method_selection(
    structure_similarity: np.ndarray,
    structure_distance: np.ndarray,
    config: WorkflowConfig,
) -> pd.DataFrame:
    """Structure-only hyperparameter sweep for the enabled map methods.

    For each enabled method, a small grid over its key neighbourhood-size knob is
    embedded and scored by the same distance/neighbourhood-preservation diagnostics
    used elsewhere, so a defensible operating point can be chosen from evidence.
    """
    seed = config.embedding.random_state
    rows: List[Dict[str, Any]] = []

    def record(method: str, hyperparameter: str, value: Any, coordinates: np.ndarray) -> None:
        preservation = distance_preservation_diagnostics(
            structure_distance,
            coordinates,
            n_neighbors=config.embedding.validation_neighbors,
            random_state=seed,
        )
        rows.append(
            {"method": method, "hyperparameter": hyperparameter, "value": value, **preservation}
        )

    record(
        "pca",
        "n_components",
        2,
        pca_embedding(
            weighted_feature_matrix(structure_similarity, None, property_weight=0.0),
            random_state=seed,
        ).coordinates,
    )
    for neighbors in (15, 30, 50):
        record(
            "umap",
            "n_neighbors",
            neighbors,
            umap_embedding(
                structure_distance, seed, neighbors, config.embedding.umap_min_dist
            ).coordinates,
        )
    if config.embedding.include_tsne:
        for perplexity in (15, 30, 50):
            record(
                "tsne",
                "perplexity",
                perplexity,
                tsne_embedding(structure_distance, seed, float(perplexity)).coordinates,
            )
    if config.embedding.include_pacmap:
        for neighbors in (5, 10, 20):
            record(
                "pacmap",
                "n_neighbors",
                neighbors,
                pacmap_embedding(structure_similarity, seed, neighbors).coordinates,
            )
    if config.embedding.include_trimap:
        for inliers in (8, 12, 20):
            record(
                "trimap",
                "n_inliers",
                inliers,
                trimap_embedding(structure_distance, seed, n_inliers=inliers).coordinates,
            )
    return pd.DataFrame(rows)


def _execute_workflow(
    config: WorkflowConfig,
    staging: Path,
) -> Dict[str, Any]:
    stage_times: Dict[str, float] = {}
    start = time.perf_counter()

    input_df = pd.read_csv(config.input_path)
    input_rows = len(input_df)
    if config.id_col is None and not any(
        candidate in input_df.columns
        for candidate in (
            "molecule_chembl_id",
            "parent_molecule_chembl_id",
            "mol_id",
            "compound_id",
            "id",
            "name",
            "__row_index",
        )
    ):
        input_df = input_df.copy()
        input_df["__row_index"] = [f"row_{index}" for index in input_df.index]
    schema = detect_schema(
        input_df,
        smiles_col=config.smiles_col,
        id_col=config.id_col,
        property_cols=config.property_cols,
    )
    _validate_selected_transforms(config, schema)
    if config.sample_size is not None and config.sample_size < len(input_df):
        input_df = input_df.sample(
            n=config.sample_size,
            random_state=config.embedding.random_state,
        )
    sampled_rows = len(input_df)
    geometry_property_cols = list(schema.property_cols)
    if config.eda.advanced and len(schema.property_cols) == 1:
        from .eda.property_registry import infer_property_profile

        selected_name = schema.property_cols[0]
        selected_profile = infer_property_profile(
            selected_name,
            input_df[selected_name],
            metadata=input_df,
            requested_type=config.eda.property_type,
            higher_is_better=config.eda.higher_is_better,
        )
        if selected_profile.semantic_type in {
            "classification",
            "generic_categorical",
        }:
            geometry_property_cols = []
    geometry_schema = SchemaSelection(
        smiles_col=schema.smiles_col,
        id_col=schema.id_col,
        property_cols=geometry_property_cols,
        property_detection=schema.property_detection,
    )
    stage_times["load_and_schema_seconds"] = time.perf_counter() - start

    stage = time.perf_counter()
    molecule_cohort = parse_molecules(input_df, schema.smiles_col, schema.id_col)
    structure_df = molecule_cohort.df.copy()
    structure_df["_structure_index"] = np.arange(len(structure_df), dtype=np.int32)
    descriptor_df = molecular_descriptor_frame(molecule_cohort.mols)
    structure_df = pd.concat([structure_df, descriptor_df], axis=1)
    descriptor_cols = descriptor_df.columns.tolist()
    identity_rows, identity_summary = identity_audit(structure_df)
    descriptor_summary, descriptor_outliers = descriptor_audit(
        structure_df,
        descriptor_cols,
    )
    structure_df, scaffold_summary = assign_scaffold_families(
        structure_df,
        molecule_cohort.mols,
    )
    fingerprints = build_fingerprints(molecule_cohort.mols, config.fingerprint)
    collision_rows, collision_summary = fingerprint_collision_summary(
        structure_df,
        fingerprints,
    )
    collision_group_by_row = (
        collision_rows.set_index("source_row")["collision_group"].to_dict()
        if not collision_rows.empty
        else {}
    )
    collision_size_by_row = (
        collision_rows.set_index("source_row")["group_size"].to_dict()
        if not collision_rows.empty
        else {}
    )
    structure_df["fingerprint_collision_group"] = structure_df["_source_row"].map(
        collision_group_by_row
    ).astype("Int64")
    structure_df["fingerprint_collision_group_size"] = (
        structure_df["_source_row"].map(collision_size_by_row).fillna(1).astype(int)
    )
    structure_df["fingerprint_collision"] = structure_df[
        "fingerprint_collision_group"
    ].notna()
    stage_times["molecular_preparation_seconds"] = time.perf_counter() - stage

    if len(structure_df) > config.embedding.max_pairwise_molecules:
        raise ValueError(
            f"Exact pairwise analysis is limited to "
            f"{config.embedding.max_pairwise_molecules} molecules; received "
            f"{len(structure_df)}. Use --sample-size or increase the explicit "
            "limit after reviewing memory and runtime requirements."
        )

    matrix_bytes = int(len(structure_df) ** 2 * 4)
    resource_estimate = {
        "n_molecules": int(len(structure_df)),
        "pairwise_matrix_bytes": matrix_bytes,
        "estimated_peak_pairwise_bytes": int(matrix_bytes * 8),
        "complexity": "O(n^2)",
        "exact_pairwise_limit": int(config.embedding.max_pairwise_molecules),
    }

    stage = time.perf_counter()
    structure_distance = tanimoto_distance_matrix(fingerprints)
    structure_similarity = (1.0 - structure_distance).astype(np.float32)
    butina_labels_default, _ = butina_labels(
        fingerprints,
        config.clustering.butina_similarity_threshold,
    )
    structure_df["butina_cluster_id"] = butina_labels_default
    butina_summary = cluster_summary(
        structure_df,
        butina_labels_default,
        structure_distance,
        property_cols=geometry_property_cols,
    )
    butina_sensitivity = threshold_sensitivity(
        fingerprints,
        [
            config.clustering.butina_similarity_threshold,
            *config.clustering.threshold_sensitivity,
        ],
    )
    hdbscan_summary_df = None
    hdbscan_meta = None
    if config.clustering.hdbscan:
        hdbscan_label_array, hdbscan_meta = hdbscan_clustering(
            structure_distance,
            config.clustering.hdbscan_min_cluster_size,
            butina_labels_default,
        )
        structure_df["hdbscan_cluster_id"] = hdbscan_label_array
        hdbscan_summary_df = cluster_summary(
            structure_df,
            hdbscan_label_array,
            structure_distance,
            property_cols=geometry_property_cols,
        )
    stage_times["distances_and_clustering_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    property_cohort = prepare_property_cohort(
        structure_df,
        property_cols=geometry_property_cols,
        requested_transforms=config.property_transforms,
        random_state=config.embedding.random_state,
    )
    exclusions = _merge_exclusions(
        [molecule_cohort.exclusions, property_cohort.exclusions]
    )
    stage_times["property_cohort_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    structure_results, structure_diagnostics = _embed_structure(
        structure_similarity,
        structure_distance,
        structure_df[descriptor_cols],
        config,
    )
    structure_seed_stability = umap_seed_stability(
        structure_distance,
        structure_results["umap"].coordinates,
        reference_seed=config.embedding.random_state,
        seeds=config.embedding.umap_seed_sensitivity,
        n_neighbors=config.embedding.umap_neighbors,
        min_dist=config.embedding.umap_min_dist,
    )
    structure_diagnostics["umap"]["seed_stability"] = structure_seed_stability
    if "tsne" in structure_results:
        structure_diagnostics["tsne"]["seed_stability"] = tsne_seed_stability(
            structure_distance,
            structure_results["tsne"].coordinates,
            reference_seed=config.embedding.random_state,
            seeds=config.embedding.umap_seed_sensitivity,
            perplexity=config.embedding.tsne_perplexity,
        )
    if "pacmap" in structure_results:
        structure_diagnostics["pacmap"]["seed_stability"] = pacmap_seed_stability(
            structure_similarity,
            structure_results["pacmap"].coordinates,
            reference_seed=config.embedding.random_state,
            seeds=config.embedding.umap_seed_sensitivity,
            n_neighbors=config.embedding.pacmap_neighbors,
        )
    if "trimap" in structure_results:
        structure_diagnostics["trimap"]["seed_stability"] = trimap_seed_stability(
            structure_distance,
            structure_results["trimap"].coordinates,
            reference_seed=config.embedding.random_state,
            seeds=config.embedding.umap_seed_sensitivity,
        )
    coranking_curves: Dict[str, Dict[str, np.ndarray]] = {}
    if config.embedding.coranking_diagnostics:
        for method, result in structure_results.items():
            summary, curve = advanced_map_diagnostics(
                structure_distance,
                result.coordinates,
                n_neighbors=config.embedding.validation_neighbors,
                random_state=config.embedding.random_state,
            )
            structure_diagnostics[method]["coranking"] = summary
            coranking_curves[method] = curve
    property_results = None
    property_diagnostics = None
    activity_colored_diagnostics = None
    property_weight_sensitivity_rows: List[Dict[str, Any]] = []
    umap_seed_sensitivity_rows: List[Dict[str, Any]] = [
        {"map_family": "structure_only", "property_weight": None, **row}
        for row in structure_seed_stability
    ]
    if geometry_property_cols:
        property_indices = property_cohort.df["_structure_index"].to_numpy(dtype=int)
        property_structure_distance = structure_distance[
            np.ix_(property_indices, property_indices)
        ]
        property_distance = normalized_property_distance(
            property_cohort.processed_matrix
        )
        activity_colored_diagnostics = {}
        property_descriptor_df = property_cohort.df[descriptor_cols]
        for method, result in structure_results.items():
            activity_colored_diagnostics[method] = _map_diagnostics(
                EmbeddingResult(
                    result.coordinates[property_indices],
                    result.diagnostics,
                ),
                source_distances=property_structure_distance,
                property_df=property_cohort.df[geometry_property_cols],
                descriptor_df=property_descriptor_df,
                config=config,
                geometry_inputs={
                    "structure": "subset of structure-only map on the property cohort",
                    "property_in_geometry": False,
                    "note": "Property changes color only; coordinates are inherited.",
                },
            )
        weight_cache = {}
        for weight in sorted(
            set(
                [
                    config.embedding.property_weight,
                    *config.embedding.property_weight_sensitivity,
                ]
            )
        ):
            results, diagnostics, fused_distance = _embed_property_weight(
                property_structure_distance,
                property_distance,
                property_cohort.df[geometry_property_cols],
                property_descriptor_df,
                property_weight=weight,
                config=config,
            )
            weight_cache[weight] = (results, diagnostics, fused_distance)
        property_results, property_diagnostics, default_fused_distance = weight_cache[
            config.embedding.property_weight
        ]
        property_seed_stability = umap_seed_stability(
            default_fused_distance,
            property_results["umap"].coordinates,
            reference_seed=config.embedding.random_state,
            seeds=config.embedding.umap_seed_sensitivity,
            n_neighbors=config.embedding.umap_neighbors,
            min_dist=config.embedding.umap_min_dist,
        )
        property_diagnostics["umap"]["seed_stability"] = property_seed_stability
        if "tsne" in property_results:
            property_diagnostics["tsne"]["seed_stability"] = tsne_seed_stability(
                default_fused_distance,
                property_results["tsne"].coordinates,
                reference_seed=config.embedding.random_state,
                seeds=config.embedding.umap_seed_sensitivity,
                perplexity=config.embedding.tsne_perplexity,
            )
        if "pacmap" in property_results:
            property_diagnostics["pacmap"]["seed_stability"] = pacmap_seed_stability(
                1.0 - default_fused_distance,
                property_results["pacmap"].coordinates,
                reference_seed=config.embedding.random_state,
                seeds=config.embedding.umap_seed_sensitivity,
                n_neighbors=config.embedding.pacmap_neighbors,
            )
        if "trimap" in property_results:
            property_diagnostics["trimap"]["seed_stability"] = trimap_seed_stability(
                default_fused_distance,
                property_results["trimap"].coordinates,
                reference_seed=config.embedding.random_state,
                seeds=config.embedding.umap_seed_sensitivity,
            )
        umap_seed_sensitivity_rows.extend(
            {
                "map_family": "property_aware",
                "property_weight": config.embedding.property_weight,
                **row,
            }
            for row in property_seed_stability
        )
        for weight, (results, diagnostics, _) in sorted(weight_cache.items()):
            for method in results:
                row = _flatten_sensitivity_row(
                    weight,
                    method,
                    diagnostics[method],
                )
                _, _, disparity = procrustes(
                    property_results[method].coordinates,
                    results[method].coordinates,
                )
                row["procrustes_disparity_to_default"] = float(disparity)
                property_weight_sensitivity_rows.append(row)
    stage_times["embeddings_and_validation_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    data_dir = staging / "data"
    cluster_dir = staging / "clusters"
    diagnostic_dir = staging / "diagnostics"
    data_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    structure_export = structure_df.copy()
    for method, result in structure_results.items():
        structure_export[f"structure_{method}_x"] = result.coordinates[:, 0]
        structure_export[f"structure_{method}_y"] = result.coordinates[:, 1]
    structure_export.to_csv(data_dir / "structure_cohort_coordinates.csv", index=False)

    if geometry_property_cols:
        property_export = property_cohort.df.copy()
        property_indices = property_export["_structure_index"].to_numpy(dtype=int)
        for method, result in structure_results.items():
            property_export[f"structure_{method}_x"] = result.coordinates[
                property_indices, 0
            ]
            property_export[f"structure_{method}_y"] = result.coordinates[
                property_indices, 1
            ]
        assert property_results is not None
        for method, result in property_results.items():
            property_export[f"property_aware_{method}_x"] = result.coordinates[:, 0]
            property_export[f"property_aware_{method}_y"] = result.coordinates[:, 1]
        property_export.to_csv(
            data_dir / "property_cohort_coordinates.csv",
            index=False,
        )

    exclusions.to_csv(data_dir / "exclusion_audit.csv", index=False)
    identity_rows.to_csv(data_dir / "identity_audit.csv", index=False)
    collision_rows.to_csv(data_dir / "fingerprint_collisions.csv", index=False)
    scaffold_summary.to_csv(cluster_dir / "scaffold_families.csv", index=False)
    butina_summary.to_csv(cluster_dir / "butina_cluster_summary.csv", index=False)
    butina_sensitivity.to_csv(
        cluster_dir / "butina_threshold_sensitivity.csv",
        index=False,
    )
    if hdbscan_summary_df is not None:
        hdbscan_summary_df.to_csv(
            cluster_dir / "hdbscan_cluster_summary.csv", index=False
        )
    if property_weight_sensitivity_rows:
        pd.DataFrame(property_weight_sensitivity_rows).to_csv(
            diagnostic_dir / "property_weight_sensitivity.csv",
            index=False,
        )
    pd.DataFrame(umap_seed_sensitivity_rows).to_csv(
        diagnostic_dir / "umap_seed_sensitivity.csv",
        index=False,
    )
    if config.embedding.map_method_selection:
        _map_method_selection(
            structure_similarity, structure_distance, config
        ).to_csv(diagnostic_dir / "map_method_selection.csv", index=False)
    if config.embedding.coranking_diagnostics:
        for method, curve in coranking_curves.items():
            pd.DataFrame(curve).to_csv(
                diagnostic_dir / f"coranking_{method}.csv", index=False
            )
            plot_shepard(
                structure_distance,
                structure_results[method].coordinates,
                staging / "figures" / f"shepard_{method}",
                random_state=config.embedding.random_state,
            )
    if config.fingerprint.representation_sensitivity:
        representation_sensitivity(
            structure_distance,
            molecule_cohort.mols,
            config.fingerprint.comparison_representations,
            n_neighbors=config.embedding.validation_neighbors,
            random_state=config.embedding.random_state,
            n_bits=config.fingerprint.n_bits,
            radius=config.fingerprint.radius,
        ).to_csv(diagnostic_dir / "representation_sensitivity.csv", index=False)
    descriptor_summary.to_csv(diagnostic_dir / "descriptor_summary.csv", index=False)
    descriptor_outliers.to_csv(
        diagnostic_dir / "descriptor_outliers.csv",
        index=False,
    )

    filename_map = _plot_outputs(
        staging,
        structure_df,
        property_cohort,
        structure_results,
        structure_diagnostics,
        property_results,
        property_diagnostics,
        geometry_property_cols,
    )
    write_json(diagnostic_dir / "property_filename_map.json", filename_map)
    stage_times["exports_and_figures_seconds"] = time.perf_counter() - stage

    report_property_df = property_cohort.df
    if schema.property_cols and not geometry_property_cols:
        report_property_df = structure_df.dropna(subset=schema.property_cols).copy()
    counts = {
        "input_rows": int(input_rows),
        "sampled_rows": int(sampled_rows),
        "structure_rows": int(len(structure_df)),
        "property_rows": int(len(report_property_df)),
        "exclusion_rows": int(len(exclusions)),
    }
    _write_run_summary(staging / "RUN_SUMMARY.md", geometry_schema, counts, config)
    _write_methods_and_captions(
        staging / "METHODS_AND_CAPTIONS.md",
        geometry_schema,
        config,
        filename_map,
    )
    write_json(
        diagnostic_dir / "map_diagnostics.json",
        {
            "structure_maps": structure_diagnostics,
            "activity_colored_structure_maps": activity_colored_diagnostics,
            "property_aware_maps": property_diagnostics,
        },
    )
    write_json(
        diagnostic_dir / "dataset_diagnostics.json",
        {
            "counts": counts,
            "schema": schema.__dict__,
            "property_cohort": property_cohort.summary,
            "fingerprint_collisions": collision_summary,
            "identity_audit": identity_summary,
            "descriptor_audit": {
                "n_descriptors": len(descriptor_cols),
                "n_outlier_rows": int(len(descriptor_outliers)),
                "robust_z_threshold": 3.5,
            },
            "n_scaffold_families": int(len(scaffold_summary)),
            "n_default_butina_clusters": int(
                structure_df["butina_cluster_id"].nunique()
            ),
            "hdbscan": hdbscan_meta,
            "resource_estimate": resource_estimate,
        },
    )

    eda_result = None
    if config.eda.enabled:
        from .eda.report_data import write_eda_artifacts

        stage = time.perf_counter()
        eda_result = write_eda_artifacts(
            staging=staging,
            schema=schema,
            config=config,
            input_rows=input_rows,
            sampled_rows=sampled_rows,
            input_df=input_df,
            structure_df=structure_df,
            property_df=report_property_df,
            molecule_exclusions=molecule_cohort.exclusions,
            mols=molecule_cohort.mols,
            descriptor_cols=descriptor_cols,
            fingerprints=fingerprints,
            structure_distance=structure_distance,
            scaffold_summary=scaffold_summary,
            butina_summary=butina_summary,
            structure_results=structure_results,
            property_results=property_results,
        )
        stage_times["eda_report_seconds"] = time.perf_counter() - stage

    return {
        "schema": schema.__dict__,
        "counts": counts,
        "property_cohort": property_cohort.summary,
        "fingerprint_collisions": collision_summary,
        "identity_audit": identity_summary,
        "stage_times_seconds": stage_times,
        "resource_estimate": resource_estimate,
        "eda": eda_result,
    }


def run_workflow(config: WorkflowConfig) -> Dict[str, Any]:
    """Run the complete workflow and atomically finalize its outputs."""
    config.validate()
    total_start = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    final_output = config.output_dir.resolve()

    with atomic_output_directory(final_output, overwrite=config.overwrite) as staging:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = _execute_workflow(config, staging)
        warning_summary = [
            {
                "category": item.category.__name__,
                "message": str(item.message),
                "filename": Path(item.filename).name,
                "line": int(item.lineno),
            }
            for item in captured
        ]
        input_sha256 = sha256_file(config.input_path)
        semantic_config = config.as_dict()
        semantic_config.pop("input_path", None)
        semantic_config.pop("output_dir", None)
        semantic_config.pop("overwrite", None)
        config_fingerprint = hashlib.sha256(
            json.dumps(semantic_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        manifest = {
            "workflow_version": __version__,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "invocation": public_invocation([sys.executable, *sys.argv]),
            "input": {
                "path": str(config.input_path.resolve()),
                "sha256": input_sha256,
            },
            "output": str(final_output),
            "config": config.as_dict(),
            "analysis_identity": {
                "curated_input_sha256": input_sha256,
                "molecular_eda_config_sha256": config_fingerprint,
                "implementation_version": __version__,
            },
            "provenance": config.provenance,
            "runtime": dependency_versions(),
            "result": result,
            "warning_count": len(warning_summary),
            "total_runtime_seconds": time.perf_counter() - total_start,
            "privacy": {
                "host_paths": "absolute paths reduced to file or directory names",
                "invocation": "command arguments omitted",
            },
        }
        manifest = redact_host_paths(manifest)
        write_json(staging / "diagnostics" / "warnings.json", warning_summary)
        write_json(staging / "run_manifest.json", manifest)
        write_artifact_manifest(staging, staging / "artifact_manifest.csv")

    return manifest
