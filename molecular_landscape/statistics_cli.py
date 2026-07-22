"""CLI for the rigorous, exploratory dataset-statistics report (additive, opt-in).

Reads a finished run's tidy export (eda/molecule_table.csv) and writes a statistics
report: target description + normality, univariate descriptor<->property association
(FDR-controlled), Kruskal-Wallis across activity classes, an active-vs-inactive
contrast, chi-square categorical associations, class balance, and descriptor
collinearity. See molecular_landscape/statistics.py for the methodology and caveats.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .io_utils import (
    atomic_output_directory,
    dependency_versions,
    public_invocation,
    redact_host_paths,
    sha256_file,
    write_artifact_manifest,
    write_json,
)
from . import statistics as ds

_DEFAULT_DESCRIPTORS = [
    "MolWt", "LogP", "TPSA", "HBA", "HBD", "RotBonds", "RingCount",
    "HeavyAtoms", "FractionCSP3", "QED", "Lipinski_Violation_Count",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rigorous exploratory dataset statistics (FDR-controlled associations, "
            "effect sizes, class balance) from a run's eda/molecule_table.csv."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/curated_advanced_eda"))
    parser.add_argument("--table", type=str, default="eda/molecule_table.csv")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <run-dir>/statistics.")
    parser.add_argument("--property-col", type=str, default="pchembl_value")
    parser.add_argument("--group-col", type=str, default="activity_class")
    parser.add_argument("--active-class", type=str, default="active")
    parser.add_argument("--inactive-class", type=str, default="inactive")
    parser.add_argument("--descriptors", nargs="+", default=_DEFAULT_DESCRIPTORS)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--collinearity-threshold", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_statistics(args: argparse.Namespace) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    table_path = (args.run_dir / args.table).resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"Molecule table not found: {table_path}")
    df = pd.read_csv(table_path, low_memory=False)
    descriptors = [d for d in args.descriptors if d in df.columns]
    alpha = args.fdr_alpha

    association = ds.descriptor_property_association(df, descriptors, args.property_col)
    kruskal = (
        ds.kruskal_across_groups(df, descriptors, args.group_col)
        if args.group_col in df.columns else pd.DataFrame()
    )
    contrast = (
        ds.binary_contrast(df, descriptors, args.group_col, args.active_class, args.inactive_class)
        if args.group_col in df.columns else pd.DataFrame()
    )
    collinearity = ds.descriptor_collinearity(df, descriptors, args.collinearity_threshold)

    # Chi-square: is activity class independent of structural categories?
    categorical: Dict[str, Any] = {}
    if args.group_col in df.columns and "scaffold_size" in df.columns:
        singleton = df["scaffold_size"].eq(1).map({True: "singleton", False: "multi-member"})
        categorical["activity_class_vs_scaffold_singleton"] = ds.categorical_association(
            df[args.group_col], singleton
        )
    if args.group_col in df.columns and "Lipinski_Passes" in df.columns:
        categorical["activity_class_vs_lipinski_pass"] = ds.categorical_association(
            df[args.group_col], df["Lipinski_Passes"].astype(str)
        )

    summary = {
        "property": args.property_col,
        "property_description": ds.describe_property(df[args.property_col])
        if args.property_col in df.columns else None,
        "class_balance": ds.class_balance(df[args.group_col])
        if args.group_col in df.columns else None,
        "categorical_associations": categorical,
        "n_descriptors_tested": len(descriptors),
        "fdr_alpha": alpha,
        "significant_after_fdr": {
            "descriptor_property_spearman": int(
                (association["spearman_q_bh"] < alpha).sum()) if not association.empty else 0,
            "kruskal_across_classes": int((kruskal["q_bh"] < alpha).sum()) if not kruskal.empty else 0,
            "active_vs_inactive_mannwhitney": int(
                (contrast["mannwhitney_q_bh"] < alpha).sum()) if not contrast.empty else 0,
        },
        "interpretation_notes": [
            "All associations are univariate and exploratory, not causal or predictive.",
            "p-values are accompanied by effect sizes and controlled with Benjamini-Hochberg FDR.",
            "Non-parametric tests (Spearman, Kruskal-Wallis, Mann-Whitney) are primary; "
            "parametric counterparts are reported for reference only.",
            "Pooled bioactivity can mix assay contexts; treat associations as hypotheses.",
        ],
    }

    final_output = (args.output_dir or (args.run_dir / "statistics")).resolve()
    with atomic_output_directory(final_output, overwrite=args.overwrite) as staging:
        association.to_csv(staging / "descriptor_property_association.csv", index=False)
        kruskal.to_csv(staging / "group_comparison_kruskal.csv", index=False)
        contrast.to_csv(staging / "active_vs_inactive_contrast.csv", index=False)
        collinearity.to_csv(staging / "descriptor_collinearity.csv", index=False)
        write_json(staging / "statistics_summary.json", summary)
        manifest = {
            "tool": "molecular-landscape dataset statistics",
            "status": "complete",
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "invocation": public_invocation([sys.executable, *sys.argv]),
            "input": {"path": str(table_path), "sha256": sha256_file(table_path), "n_rows": int(len(df))},
            "config": {
                "property_col": args.property_col, "group_col": args.group_col,
                "descriptors": descriptors, "fdr_alpha": alpha,
                "collinearity_threshold": args.collinearity_threshold,
            },
            "runtime": dependency_versions(),
            "total_runtime_seconds": time.perf_counter() - start,
            "privacy": {
                "host_paths": "absolute paths reduced to file or directory names",
                "invocation": "command arguments omitted",
            },
        }
        manifest = redact_host_paths(manifest)
        write_json(staging / "statistics_manifest.json", manifest)
        write_artifact_manifest(staging, staging / "artifact_manifest.csv")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_statistics(args)
    sig = summary["significant_after_fdr"]
    print(f"Dataset statistics written for property '{summary['property']}'.")
    print(
        f"Descriptors tested: {summary['n_descriptors_tested']}; "
        f"FDR-significant (q<{summary['fdr_alpha']}): "
        f"property-corr={sig['descriptor_property_spearman']}, "
        f"class-difference={sig['kruskal_across_classes']}, "
        f"active-vs-inactive={sig['active_vs_inactive_mannwhitney']}."
    )
    return 0
