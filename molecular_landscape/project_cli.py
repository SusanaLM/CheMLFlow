"""CLI for scalable, inductive chemical-space projection.

Fit a feature-based UMAP (Jaccard = Tanimoto) on a reference set and, optionally,
project a query set onto it with an applicability-domain score. Scales past the
exact-pairwise cap and supports out-of-sample molecules. See ``projection.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .chemistry import build_fingerprints, parse_molecules
from .config import FingerprintConfig
from .io_utils import (
    atomic_output_directory,
    dependency_versions,
    public_invocation,
    redact_host_paths,
    sha256_file,
    write_artifact_manifest,
    write_json,
)
from .projection import (
    applicability_domain,
    embedding_applicability_domain,
    fingerprint_matrix,
    fit_reference_projection,
    project_query,
    sampled_preservation_diagnostics,
)
from .schema import detect_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scalable, inductive chemical-space projection. Fits a feature-based "
            "UMAP (Tanimoto) on a reference set and optionally projects a query set "
            "with an applicability-domain score."
        )
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smiles-col", type=str)
    parser.add_argument("--id-col", type=str)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ad-k", type=int, default=1, help="Top-k Tanimoto for applicability domain.")
    parser.add_argument("--sample-pairs", type=int, default=200_000)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--no-chirality", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--embeddings",
        type=Path,
        help=(
            "Use precomputed learned/foundation-model embeddings (.npy or .csv, one "
            "row per reference input row) instead of Morgan fingerprints."
        ),
    )
    parser.add_argument(
        "--query-embeddings",
        type=Path,
        help="Embeddings for the query set (required with --embeddings when --query is given).",
    )
    parser.add_argument("--embedding-metric", default="cosine")
    return parser


def _load_embeddings(path: Path) -> np.ndarray:
    if str(path).endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float32)
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    return numeric.to_numpy(dtype=np.float32)


def _prepare(
    df: pd.DataFrame,
    smiles_col: Optional[str],
    id_col: Optional[str],
    fp_config,
    compute_fingerprints: bool = True,
):
    schema = detect_schema(df, smiles_col=smiles_col, id_col=id_col, property_cols=[])
    cohort = parse_molecules(df, schema.smiles_col, schema.id_col)
    fingerprints = matrix = None
    if compute_fingerprints:
        fingerprints = build_fingerprints(cohort.mols, fp_config)
        matrix = fingerprint_matrix(fingerprints, fp_config.n_bits)
    return schema, cohort, fingerprints, matrix


def _aligned_embeddings(path: Path, df: pd.DataFrame, cohort) -> np.ndarray:
    """Load embeddings (aligned to input rows) and select the valid-molecule subset."""
    embeddings = _load_embeddings(path)
    if len(embeddings) != len(df):
        raise ValueError(
            f"Embeddings in {path} have {len(embeddings)} rows but the input CSV has "
            f"{len(df)}; they must be aligned one row per input molecule."
        )
    return embeddings[cohort.df["_source_row"].to_numpy(dtype=int)]


def _plot(staging: Path, ref_xy, query_xy, ad_scores, ad_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import PLOT_STYLE, _save_all_formats

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
        ax.scatter(
            ref_xy[:, 0], ref_xy[:, 1], s=8, c="#c7c9cc", alpha=0.5,
            linewidths=0, rasterized=True, label="Reference",
        )
        if query_xy is not None and len(query_xy):
            scatter = ax.scatter(
                query_xy[:, 0], query_xy[:, 1], c=ad_scores, cmap="viridis",
                s=26, alpha=0.9, linewidths=0.3, edgecolors="white",
                vmin=0.0, vmax=1.0, rasterized=True, label="Query",
            )
            colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            colorbar.set_label(ad_label)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
               title="Scalable chemical-space projection")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        _save_all_formats(fig, staging / "figures" / "projection")
        plt.close(fig)


def run_projection(args: argparse.Namespace) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    total_start = time.perf_counter()
    fp_config = FingerprintConfig(
        radius=args.fp_radius,
        n_bits=args.fp_bits,
        include_chirality=not args.no_chirality,
    )
    fp_config.validate()

    use_embeddings = args.embeddings is not None
    metric = args.embedding_metric if use_embeddings else "jaccard"
    ad_label = (
        f"Applicability (max cosine to reference)"
        if use_embeddings
        else "Applicability (max Tanimoto to reference)"
    )
    ad_column = (
        "applicability_max_cosine" if use_embeddings else "applicability_max_tanimoto"
    )

    reference_df = pd.read_csv(args.reference)
    ref_schema, ref_cohort, ref_fps, ref_matrix = _prepare(
        reference_df, args.smiles_col, args.id_col, fp_config,
        compute_fingerprints=not use_embeddings,
    )
    if use_embeddings:
        ref_matrix = _aligned_embeddings(args.embeddings, reference_df, ref_cohort)
    fitted = fit_reference_projection(
        ref_matrix,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=metric,
    )
    diagnostics = dict(fitted.diagnostics)
    if not use_embeddings:
        diagnostics["preservation"] = sampled_preservation_diagnostics(
            ref_fps, fitted.coordinates, random_state=args.random_state, n_pairs=args.sample_pairs
        )

    query_xy = None
    ad_scores = None
    query_count = 0
    if args.query is not None:
        query_df = pd.read_csv(args.query)
        _, query_cohort, query_fps, query_matrix = _prepare(
            query_df, args.smiles_col, args.id_col, fp_config,
            compute_fingerprints=not use_embeddings,
        )
        if use_embeddings:
            if args.query_embeddings is None:
                raise ValueError("--query-embeddings is required with --embeddings when --query is given.")
            query_matrix = _aligned_embeddings(args.query_embeddings, query_df, query_cohort)
            ad_scores = embedding_applicability_domain(ref_matrix, query_matrix, k=args.ad_k)
        else:
            ad_scores = applicability_domain(ref_fps, query_fps, k=args.ad_k)
        query_xy = project_query(fitted, query_matrix)
        query_count = len(query_cohort.df)

    final_output = args.output_dir.resolve()
    with atomic_output_directory(final_output, overwrite=args.overwrite) as staging:
        (staging / "data").mkdir(parents=True, exist_ok=True)
        reference_out = ref_cohort.df[["_source_row", "_compound_id", "_canonical_smiles"]].copy()
        reference_out.columns = ["source_row", "compound_id", "canonical_smiles"]
        reference_out["umap_x"] = fitted.coordinates[:, 0]
        reference_out["umap_y"] = fitted.coordinates[:, 1]
        reference_out.to_csv(staging / "data" / "reference_coordinates.csv", index=False)

        if query_xy is not None:
            query_out = query_cohort.df[["_source_row", "_compound_id", "_canonical_smiles"]].copy()
            query_out.columns = ["source_row", "compound_id", "canonical_smiles"]
            query_out["umap_x"] = query_xy[:, 0]
            query_out["umap_y"] = query_xy[:, 1]
            query_out[ad_column] = ad_scores
            if not use_embeddings:
                query_out["in_domain"] = ad_scores >= 0.30  # conventional Tanimoto AD heuristic
            query_out.to_csv(staging / "data" / "query_coordinates.csv", index=False)

        if not args.no_figure:
            _plot(staging, fitted.coordinates, query_xy, ad_scores, ad_label)

        write_json(staging / "diagnostics.json", diagnostics)
        manifest = {
            "tool": "molecular-landscape scalable projection",
            "status": "complete",
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "invocation": public_invocation([sys.executable, *sys.argv]),
            "reference": {
                "path": str(args.reference.resolve()),
                "sha256": sha256_file(args.reference),
                "n_molecules": int(len(ref_cohort.df)),
            },
            "query": (
                {
                    "path": str(args.query.resolve()),
                    "sha256": sha256_file(args.query),
                    "n_molecules": int(query_count),
                }
                if args.query is not None
                else None
            ),
            "config": {
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "random_state": args.random_state,
                "metric": metric,
                "embeddings": bool(use_embeddings),
                "ad_k": args.ad_k,
                "fingerprint": {"radius": fp_config.radius, "n_bits": fp_config.n_bits},
            },
            "diagnostics": diagnostics,
            "runtime": dependency_versions(),
            "total_runtime_seconds": time.perf_counter() - total_start,
            "privacy": {
                "host_paths": "absolute paths reduced to file or directory names",
                "invocation": "command arguments omitted",
            },
        }
        manifest = redact_host_paths(manifest)
        write_json(staging / "projection_manifest.json", manifest)
        write_artifact_manifest(staging, staging / "artifact_manifest.csv")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_projection(args)
    print(f"Projection written: {manifest['reference']['n_molecules']} reference molecules")
    if manifest["query"]:
        print(f"Projected query molecules: {manifest['query']['n_molecules']}")
    preservation = manifest["diagnostics"].get("preservation")
    if preservation:
        print(
            "Sampled preservation: "
            f"distance Spearman {preservation['sampled_distance_spearman']:.3f}, "
            + ", ".join(
                f"{k} {v:.3f}" for k, v in preservation.items() if k.startswith("sampled_knn")
            )
        )
    return 0
