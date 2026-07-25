"""Command-line interface for the standalone molecular landscape workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from . import __version__
from .config import (
    ClusteringConfig,
    EmbeddingConfig,
    FingerprintConfig,
    WorkflowConfig,
)
from .eda.config import EDAConfig


def _property_transform(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Property transforms must use COLUMN=TRANSFORM syntax."
        )
    column, transform = value.split("=", 1)
    if not column or not transform:
        raise argparse.ArgumentTypeError(
            "Property transforms must use COLUMN=TRANSFORM syntax."
        )
    return column, transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validated structure-only, activity-colored, and explicitly "
            "property-aware molecular landscape analyses."
        )
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smiles-col")
    parser.add_argument("--id-col")
    parser.add_argument(
        "--property-cols",
        nargs="*",
        default=None,
        help=(
            "Properties to analyze. Omit for conservative auto-detection; pass "
            "the option with no values to explicitly disable property maps."
        ),
    )
    parser.add_argument(
        "--property-transform",
        action="append",
        type=_property_transform,
        default=[],
        metavar="COLUMN=TRANSFORM",
        help="Per-property transform: auto, none, log1p, signed_log1p, or quantile.",
    )
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--eda-report",
        action="store_true",
        help="Generate an offline interactive molecular EDA report bundle.",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Generate and open the EDA report in the default browser.",
    )
    parser.add_argument(
        "--advanced-eda",
        action="store_true",
        help="Enable advanced property, chemistry-health, and model-readiness artifacts.",
    )
    parser.add_argument(
        "--drug-discovery-panel",
        action="store_true",
        help="Enable optional small-molecule drug-discovery heuristics.",
    )
    parser.add_argument(
        "--model-readiness",
        action="store_true",
        help="Explicitly enable model-readiness diagnostics.",
    )
    parser.add_argument(
        "--property-type",
        choices=[
            "auto",
            "potency_log",
            "potency_linear",
            "physchem",
            "admet",
            "qm_energy",
            "qm_gap",
            "classification",
            "generic_numeric",
            "generic_categorical",
        ],
        default="auto",
    )
    parser.add_argument(
        "--higher-is-better",
        choices=["auto", "true", "false"],
        default="auto",
        help="Override property direction semantics for advanced EDA.",
    )
    parser.add_argument(
        "--export-selected-template",
        action="store_true",
        help=(
            "Also emit selection/export schema artifacts "
            "(eda/selection_columns.json, eda/export_schema.json) for basic runs; "
            "advanced runs always write them. The interactive map's selected-ID "
            "CSV download is available in every report regardless of this flag."
        ),
    )
    parser.add_argument("--activity-cliff-similarity", type=float, default=0.70)
    parser.add_argument("--activity-cliff-delta", type=float, default=1.0)
    parser.add_argument("--representative-molecules", type=int, default=48)
    parser.add_argument("--max-svg-molecules", type=int, default=5000)
    parser.add_argument("--eda-nearest-neighbors", type=int, default=10)
    parser.add_argument("--eda-top-scaffolds", type=int, default=20)
    parser.add_argument(
        "--eda-map-method",
        choices=["pca", "umap", "tsne", "pacmap", "trimap"],
        default="umap",
    )

    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--no-chirality", action="store_true")
    parser.add_argument("--use-feature-invariants", action="store_true")
    parser.add_argument(
        "--representation-sensitivity",
        action="store_true",
        help=(
            "Compare the default Morgan Tanimoto geometry against alternative "
            "fingerprint families (diagnostics/representation_sensitivity.csv)."
        ),
    )
    parser.add_argument(
        "--comparison-representations",
        nargs="+",
        choices=["morgan", "fcfp", "rdkit", "atompair", "torsion", "maccs"],
        default=["fcfp", "rdkit", "atompair", "torsion", "maccs"],
    )

    parser.add_argument("--property-weight", type=float, default=0.20)
    parser.add_argument(
        "--property-weight-sensitivity",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--umap-seed-sensitivity",
        nargs="+",
        type=int,
        default=[7, 42, 99],
        help="Seeds used to quantify UMAP layout stability.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.10)
    parser.add_argument("--validation-neighbors", type=int, default=15)
    parser.add_argument("--max-pairwise-molecules", type=int, default=5000)
    parser.add_argument(
        "--include-tsne",
        action="store_true",
        help="Also compute an optional Tanimoto-aware t-SNE map (off by default; O(n^2)).",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--include-pacmap",
        action="store_true",
        help="Also compute an optional PaCMAP map (better global/local balance; off by default).",
    )
    parser.add_argument(
        "--include-trimap",
        action="store_true",
        help="Also compute an optional TriMap map (strong global structure; needs >=55 molecules).",
    )
    parser.add_argument("--pacmap-neighbors", type=int, default=10)
    parser.add_argument(
        "--map-method-selection",
        action="store_true",
        help=(
            "Write a structure-only hyperparameter-selection sweep for the enabled "
            "map methods to diagnostics/map_method_selection.csv."
        ),
    )
    parser.add_argument(
        "--coranking-diagnostics",
        action="store_true",
        help=(
            "Compute multi-scale co-ranking (R_NX/LCMC) quality, random-layout "
            "baselines, and Shepard-diagram figures for the structure-only maps."
        ),
    )

    parser.add_argument("--butina-similarity-threshold", type=float, default=0.65)
    parser.add_argument(
        "--butina-threshold-sensitivity",
        nargs="+",
        type=float,
        default=[0.55, 0.65, 0.75],
    )
    parser.add_argument(
        "--hdbscan",
        action="store_true",
        help="Also report threshold-free HDBSCAN clustering and its agreement with Butina.",
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5)
    return parser


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    transforms: Dict[str, str] = dict(args.property_transform)
    advanced = bool(
        args.advanced_eda or args.drug_discovery_panel or args.model_readiness
    )
    eda_enabled = bool(args.eda_report or args.open_report or advanced)
    higher_is_better: str | bool = args.higher_is_better
    if higher_is_better == "true":
        higher_is_better = True
    elif higher_is_better == "false":
        higher_is_better = False
    return WorkflowConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        property_cols=args.property_cols,
        property_transforms=transforms,
        sample_size=args.sample_size,
        overwrite=args.overwrite,
        fingerprint=FingerprintConfig(
            radius=args.fp_radius,
            n_bits=args.fp_bits,
            include_chirality=not args.no_chirality,
            use_features=args.use_feature_invariants,
            representation_sensitivity=bool(args.representation_sensitivity),
            comparison_representations=list(args.comparison_representations),
        ),
        embedding=EmbeddingConfig(
            property_weight=args.property_weight,
            property_weight_sensitivity=list(args.property_weight_sensitivity),
            random_state=args.random_state,
            umap_seed_sensitivity=list(args.umap_seed_sensitivity),
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            validation_neighbors=args.validation_neighbors,
            max_pairwise_molecules=args.max_pairwise_molecules,
            # Selecting a workbench map method implies computing it.
            include_tsne=bool(args.include_tsne or args.eda_map_method == "tsne"),
            tsne_perplexity=args.tsne_perplexity,
            include_pacmap=bool(args.include_pacmap or args.eda_map_method == "pacmap"),
            include_trimap=bool(args.include_trimap or args.eda_map_method == "trimap"),
            pacmap_neighbors=args.pacmap_neighbors,
            map_method_selection=bool(args.map_method_selection),
            coranking_diagnostics=bool(args.coranking_diagnostics),
        ),
        clustering=ClusteringConfig(
            butina_similarity_threshold=args.butina_similarity_threshold,
            threshold_sensitivity=list(args.butina_threshold_sensitivity),
            hdbscan=bool(args.hdbscan),
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        ),
        eda=EDAConfig(
            enabled=eda_enabled,
            advanced=advanced,
            open_report=bool(args.open_report),
            property_type=args.property_type,
            higher_is_better=higher_is_better,
            include_drug_discovery_panel=bool(args.drug_discovery_panel),
            include_model_readiness=bool(args.model_readiness or advanced),
            representative_molecules=args.representative_molecules,
            max_svg_molecules=args.max_svg_molecules,
            nearest_neighbors=args.eda_nearest_neighbors,
            top_scaffolds=args.eda_top_scaffolds,
            map_method=args.eda_map_method,
            activity_cliff_similarity=args.activity_cliff_similarity,
            activity_cliff_delta=args.activity_cliff_delta,
            export_selected_template=bool(args.export_selected_template),
        ),
    )


def _configure_runtime_caches(output_dir: Path) -> None:
    cache_root = output_dir.parent / f".{output_dir.name}.runtime-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_runtime_caches(args.output_dir)
    config = config_from_args(args)

    from .workflow import run_workflow

    manifest = run_workflow(config)
    result = manifest["result"]
    print(f"Completed molecular landscape workflow: {manifest['output']}")
    print(
        "Rows: "
        f"input={result['counts']['input_rows']}, "
        f"structure={result['counts']['structure_rows']}, "
        f"property={result['counts']['property_rows']}"
    )
    print(f"Runtime: {manifest['total_runtime_seconds']:.2f} seconds")
    if config.eda.open_report:
        import webbrowser

        report_path = config.output_dir.resolve() / "eda_report.html"
        if not webbrowser.open(report_path.as_uri()):
            print(f"Report generated but could not be opened automatically: {report_path}")
    return 0
