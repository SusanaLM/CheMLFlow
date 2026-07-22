"""Assembly and export of all optional EDA report artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from ..config import WorkflowConfig
from ..embedding import EmbeddingResult
from ..io_utils import write_json
from ..schema import SchemaSelection, is_log_activity_property
from .descriptors import (
    DESCRIPTOR_NAMES,
    calculate_molecular_descriptors,
    summarize_descriptors,
)
from .depictions import generate_molecule_svgs
from .distributions import build_property_distribution
from .druglikeness import calculate_druglikeness
from .model_readiness import build_model_readiness
from .neighbors import (
    compute_neighbors_and_cliffs,
    summarize_neighborhood_consistency,
)
from .profile import build_dataset_health, build_dataset_profile
from .property_registry import PropertyProfile, infer_property_profile
from .report_html import write_html_report
from .summaries import (
    build_descriptor_tables,
    build_gallery_candidates,
    build_cluster_summary,
    build_molecule_table,
    build_scaffold_summary,
    scaffold_property_enrichment,
    scaffold_representatives,
)


def write_eda_artifacts(
    *,
    staging: Path,
    schema: SchemaSelection,
    config: WorkflowConfig,
    input_rows: int,
    sampled_rows: int,
    input_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    property_df: pd.DataFrame,
    molecule_exclusions: pd.DataFrame,
    mols: list[Chem.Mol],
    descriptor_cols: list[str],
    fingerprints: Any,
    structure_distance: np.ndarray,
    scaffold_summary: pd.DataFrame,
    butina_summary: pd.DataFrame,
    structure_results: Dict[str, EmbeddingResult],
    property_results: Optional[Dict[str, EmbeddingResult]],
) -> dict:
    """Write EDA CSV, JSON, SVG, and HTML artifacts from authoritative workflow data."""
    del fingerprints, scaffold_summary
    eda_dir = staging / "eda"
    svg_dir = eda_dir / "molecule_svgs"
    eda_dir.mkdir(parents=True, exist_ok=True)
    svg_paths, depiction_warnings = generate_molecule_svgs(
        structure_df,
        svg_dir,
        config.eda.max_svg_molecules,
    )
    primary_property = schema.property_cols[0] if schema.property_cols else None
    property_profile: PropertyProfile | None = None
    if primary_property:
        property_profile = infer_property_profile(
            primary_property,
            structure_df[primary_property],
            metadata=structure_df,
            requested_type=config.eda.property_type,
            higher_is_better=config.eda.higher_is_better,
        )

    molecule_table = build_molecule_table(
        structure_df,
        property_df,
        schema.property_cols,
        descriptor_cols,
        svg_paths,
        structure_results,
        property_results,
    )
    basic_descriptor_summary, basic_descriptor_outliers = build_descriptor_tables(
        structure_df,
        descriptor_cols,
    )

    advanced_descriptor_frame = pd.DataFrame()
    advanced_descriptor_summary: dict[str, Any] = {}
    advanced_descriptor_outliers = pd.DataFrame()
    advanced_warnings: list[str] = []
    if config.eda.advanced:
        advanced_descriptor_frame, descriptor_warnings = calculate_molecular_descriptors(
            structure_df,
            mols,
        )
        advanced_descriptor_summary, advanced_descriptor_outliers = (
            summarize_descriptors(advanced_descriptor_frame)
        )
        advanced_warnings.extend(descriptor_warnings)
        extra_descriptors = advanced_descriptor_frame.drop(
            columns=["source_row", "compound_id"],
        )
        molecule_table = molecule_table.merge(
            extra_descriptors,
            on="structure_index",
            how="left",
            suffixes=("", "_advanced"),
            validate="one_to_one",
        )
    descriptor_outliers = (
        advanced_descriptor_outliers
        if config.eda.advanced
        else basic_descriptor_outliers.rename(
            columns={"source_row": "source_row"}
        ).assign(
            structure_index=lambda frame: frame["source_row"].map(
                dict(zip(structure_df["_source_row"], structure_df["_structure_index"]))
            )
        )
    )
    property_is_numeric = bool(
        property_profile
        and property_profile.semantic_type
        not in {"classification", "generic_categorical"}
    )

    (
        property_distribution,
        property_bins,
        property_outliers,
        property_by_scaffold,
        property_by_cluster,
        property_descriptor_relationships,
    ) = build_property_distribution(
        molecule_table,
        property_profile,
        (
            DESCRIPTOR_NAMES if config.eda.advanced else descriptor_cols
        )
        if config.eda.include_property_descriptor_plots
        else [],
    )
    if not property_bins.empty:
        molecule_table = molecule_table.merge(
            property_bins[["structure_index", "property_bin"]],
            on="structure_index",
            how="left",
            validate="one_to_one",
        )

    drug_panel_enabled = bool(
        config.eda.include_drug_discovery_panel
        or (
            config.eda.advanced
            and property_profile is not None
            and property_profile.is_potency
        )
    )
    druglikeness = pd.DataFrame()
    structural_alerts = pd.DataFrame()
    druglikeness_summary: dict[str, Any] | None = None
    if drug_panel_enabled:
        druglikeness, structural_alerts, druglikeness_summary, drug_warnings = (
            calculate_druglikeness(
                molecule_table,
                mols,
                qed_low_threshold=config.eda.qed_low_threshold,
                lipinski_violation_warning_threshold=(
                    config.eda.lipinski_violation_warning_threshold
                ),
            )
        )
        advanced_warnings.extend(drug_warnings)
        molecule_table = molecule_table.merge(
            druglikeness.drop(columns=["compound_id"]),
            on="structure_index",
            how="left",
            validate="one_to_one",
        )

    scaffold_summary_eda = build_scaffold_summary(
        molecule_table,
        primary_property if property_is_numeric else None,
        structure_distance,
    )
    cluster_summary_eda = build_cluster_summary(
        molecule_table,
        butina_summary,
        primary_property if property_is_numeric else None,
    )
    if config.eda.include_nearest_neighbors:
        nearest_neighbors, discontinuities = compute_neighbors_and_cliffs(
            molecule_table,
            structure_distance,
            primary_property,
            config.eda.nearest_neighbors,
            config.eda.activity_cliff_similarity,
            config.eda.activity_cliff_delta,
            property_is_numeric=property_is_numeric,
        )
    else:
        nearest_neighbors, discontinuities = pd.DataFrame(), pd.DataFrame()
    is_potency = bool(property_profile and property_profile.is_potency)
    collision_mask = discontinuities.get(
        "collision_derived_match",
        pd.Series(False, index=discontinuities.index),
    ).fillna(False).astype(bool)
    interpretable_discontinuities = discontinuities.loc[~collision_mask].copy()
    activity_cliffs = (
        interpretable_discontinuities.copy()
        if is_potency and config.eda.include_activity_cliffs
        else discontinuities.iloc[0:0].copy()
    )
    neighborhood_consistency = summarize_neighborhood_consistency(
        nearest_neighbors,
        interpretable_discontinuities,
        is_numeric=property_is_numeric,
    )
    galleries = build_gallery_candidates(
        molecule_table,
        scaffold_summary_eda,
        descriptor_outliers,
        primary_property if property_is_numeric else None,
        config.eda.representative_molecules,
        config.embedding.random_state,
    )
    if drug_panel_enabled and not druglikeness.empty:
        limit = config.eda.representative_molecules
        galleries["high_qed"] = (
            druglikeness.nlargest(min(limit, len(druglikeness)), "QED")[
                "structure_index"
            ]
            .astype(int)
            .tolist()
        )
        galleries["low_qed"] = (
            druglikeness.nsmallest(min(limit, len(druglikeness)), "QED")[
                "structure_index"
            ]
            .astype(int)
            .tolist()
        )
        galleries["many_lipinski_violations"] = (
            druglikeness.nlargest(
                min(limit, len(druglikeness)),
                "Lipinski_Violation_Count",
            )["structure_index"]
            .astype(int)
            .tolist()
        )
    profile = build_dataset_profile(
        schema=schema,
        input_rows=input_rows,
        structure_df=structure_df,
        property_df=property_df,
        scaffold_summary=scaffold_summary_eda,
        descriptor_outliers=basic_descriptor_outliers,
        invalid_molecules=max(0, sampled_rows - len(structure_df)),
    )
    profile["warnings"].extend([*depiction_warnings, *advanced_warnings])
    profile["counts"]["sampled_rows"] = int(sampled_rows)
    profile["counts"]["clusters"] = int(len(cluster_summary_eda))
    profile["counts"]["singleton_clusters"] = int(
        (cluster_summary_eda["size"] == 1).sum()
    )
    profile["counts"]["largest_cluster_size"] = int(cluster_summary_eda["size"].max())
    singleton_cluster_fraction = float(
        (cluster_summary_eda["size"] == 1).mean()
    )
    if singleton_cluster_fraction >= 0.5:
        profile["warnings"].append(
            f"Butina cluster fragmentation is high at the configured threshold: "
            f"{singleton_cluster_fraction:.1%} of clusters are singletons."
        )
    property_summary = {
        "selected_properties": list(schema.property_cols),
        "primary_property": primary_property,
        "primary_property_is_log_activity": bool(
            primary_property and is_log_activity_property(primary_property)
        ),
        "statistics": profile["property_summary"],
        "activity_cliff_definition": {
            "minimum_tanimoto_similarity": config.eda.activity_cliff_similarity,
            "minimum_absolute_property_difference": config.eda.activity_cliff_delta,
            "n_pairs": int(len(activity_cliffs)),
            "name": "similarity-defined activity discontinuity",
            "collision_derived_pairs_excluded": int(collision_mask.sum()),
        },
        "cohort_scope": "post-curation molecular cohort",
    }

    dataset_health = None
    dataset_warnings: list[dict[str, str]] = []
    model_readiness = None
    scaffold_enrichment = pd.DataFrame()
    if config.eda.advanced:
        dataset_health, dataset_warnings = build_dataset_health(
            input_df=input_df,
            schema=schema,
            structure_df=structure_df,
            molecule_exclusions=molecule_exclusions,
            descriptor_df=advanced_descriptor_frame,
            descriptor_outliers=advanced_descriptor_outliers,
            scaffold_summary=scaffold_summary_eda,
            singleton_warning_fraction=config.eda.singleton_scaffold_warning_fraction,
        )
        profile["warnings"].extend(
            warning["message"] for warning in dataset_warnings
        )
        scaffold_enrichment = scaffold_property_enrichment(
            scaffold_summary_eda,
            property_profile.higher_is_better if property_profile else None,
        )
        if config.eda.include_model_readiness:
            model_readiness = build_model_readiness(
                molecule_table,
                property_profile,
                nearest_neighbors,
                interpretable_discontinuities,
                advanced_descriptor_outliers,
            )

    molecule_table.to_csv(eda_dir / "molecule_table.csv", index=False)
    basic_descriptor_summary.to_csv(eda_dir / "descriptor_summary_eda.csv", index=False)
    basic_descriptor_outliers.to_csv(eda_dir / "descriptor_outliers_eda.csv", index=False)
    scaffold_summary_eda.to_csv(eda_dir / "scaffold_summary_eda.csv", index=False)
    cluster_summary_eda.to_csv(eda_dir / "cluster_summary_eda.csv", index=False)
    nearest_neighbors.to_csv(eda_dir / "nearest_neighbors.csv", index=False)
    activity_cliffs.to_csv(eda_dir / "activity_cliffs.csv", index=False)
    write_json(eda_dir / "dataset_profile.json", profile)
    write_json(eda_dir / "property_summary.json", property_summary)
    write_json(eda_dir / "gallery_candidates.json", galleries)

    advanced_artifacts: list[str] = []
    if config.eda.advanced:
        assert property_profile is not None or not schema.property_cols
        write_json(
            eda_dir / "property_profile.json",
            property_profile.as_dict() if property_profile else {"property": None},
        )
        write_json(eda_dir / "dataset_health.json", dataset_health)
        write_json(eda_dir / "dataset_warnings.json", dataset_warnings)
        advanced_descriptor_frame.to_csv(
            eda_dir / "molecular_descriptors.csv", index=False
        )
        write_json(eda_dir / "descriptor_summary.json", advanced_descriptor_summary)
        advanced_descriptor_outliers.to_csv(
            eda_dir / "descriptor_outliers.csv", index=False
        )
        write_json(eda_dir / "property_distribution.json", property_distribution)
        property_bins.to_csv(eda_dir / "property_bins.csv", index=False)
        property_outliers.to_csv(eda_dir / "property_outliers.csv", index=False)
        property_by_scaffold.to_csv(eda_dir / "property_by_scaffold.csv", index=False)
        property_by_cluster.to_csv(eda_dir / "property_by_cluster.csv", index=False)
        property_descriptor_relationships.to_csv(
            eda_dir / "property_descriptor_relationships.csv", index=False
        )
        scaffold_representatives(scaffold_summary_eda).to_csv(
            eda_dir / "scaffold_representatives.csv", index=False
        )
        scaffold_enrichment.to_csv(
            eda_dir / "scaffold_property_enrichment.csv", index=False
        )
        discontinuities.to_csv(
            eda_dir / "local_property_discontinuities.csv", index=False
        )
        write_json(
            eda_dir / "neighborhood_consistency.json", neighborhood_consistency
        )
        if model_readiness is not None:
            write_json(eda_dir / "model_readiness.json", model_readiness)
        advanced_artifacts = [
            "eda/property_profile.json",
            "eda/dataset_health.json",
            "eda/dataset_warnings.json",
            "eda/molecular_descriptors.csv",
            "eda/descriptor_summary.json",
            "eda/descriptor_outliers.csv",
            "eda/property_distribution.json",
            "eda/property_bins.csv",
            "eda/property_outliers.csv",
            "eda/property_by_scaffold.csv",
            "eda/property_by_cluster.csv",
            "eda/property_descriptor_relationships.csv",
            "eda/scaffold_representatives.csv",
            "eda/scaffold_property_enrichment.csv",
            "eda/local_property_discontinuities.csv",
            "eda/neighborhood_consistency.json",
        ]
        if model_readiness is not None:
            advanced_artifacts.append("eda/model_readiness.json")
    if drug_panel_enabled:
        druglikeness.to_csv(eda_dir / "druglikeness.csv", index=False)
        structural_alerts.to_csv(eda_dir / "structural_alerts.csv", index=False)
        write_json(eda_dir / "druglikeness_summary.json", druglikeness_summary)
        advanced_artifacts.extend(
            [
                "eda/druglikeness.csv",
                "eda/druglikeness_summary.json",
                "eda/structural_alerts.csv",
            ]
        )

    # Selection/export schema documents the interactive map's selected-ID CSV
    # download (which is always present in the report). Advanced runs always emit
    # it; --export-selected-template additionally emits it for basic runs.
    if config.eda.advanced or config.eda.export_selected_template:
        write_json(
            eda_dir / "selection_columns.json",
            {
                "default_columns": [
                    column
                    for column in (
                        "compound_id",
                        "canonical_smiles",
                        primary_property,
                        "scaffold_id",
                        "cluster_id",
                        "fingerprint_collision",
                        "svg_path",
                    )
                    if column is not None and column in molecule_table.columns
                ]
            },
        )
        write_json(
            eda_dir / "export_schema.json",
            {
                "format": "CSV",
                "row_identifier": "structure_index",
                "selection_source": "interactive structure-only map",
                "columns": molecule_table.columns.tolist(),
            },
        )
        advanced_artifacts.extend(
            ["eda/selection_columns.json", "eda/export_schema.json"]
        )

    write_html_report(
        staging / "eda_report.html",
        profile=profile,
        molecule_table=molecule_table,
        descriptor_summary=basic_descriptor_summary,
        scaffold_summary=scaffold_summary_eda,
        cluster_summary=cluster_summary_eda,
        nearest_neighbors=nearest_neighbors,
        activity_cliffs=activity_cliffs,
        galleries=galleries,
        primary_property=primary_property,
        property_profile=property_profile.as_dict() if property_profile else None,
        dataset_health=dataset_health,
        dataset_warnings=dataset_warnings,
        property_distribution=property_distribution,
        druglikeness_summary=druglikeness_summary,
        structural_alerts=structural_alerts,
        model_readiness=model_readiness,
        advanced=config.eda.advanced,
        drug_panel_enabled=drug_panel_enabled,
        use_scattergl=config.eda.use_scattergl,
        map_method=config.eda.map_method,
        top_scaffolds=config.eda.top_scaffolds,
        max_points_for_svg_hover=config.eda.max_points_for_svg_hover,
        selection_columns=[
            column
            for column in (
                "compound_id",
                "canonical_smiles",
                primary_property,
                "scaffold_id",
                "cluster_id",
                "fingerprint_collision",
                "svg_path",
            )
            if column is not None and column in molecule_table.columns
        ],
    )
    return {
        "enabled": True,
        "advanced": config.eda.advanced,
        "report": "eda_report.html",
        "eda_directory": "eda",
        "primary_property": primary_property,
        "map_geometry": f"structure-only {config.eda.map_method}",
        "n_molecule_svgs": int(len(svg_paths)),
        "n_depiction_warnings": int(len(depiction_warnings)),
        "n_nearest_neighbor_rows": int(len(nearest_neighbors)),
        "property_profile": "eda/property_profile.json" if config.eda.advanced else None,
        "dataset_health": "eda/dataset_health.json" if config.eda.advanced else None,
        "descriptors": "eda/molecular_descriptors.csv" if config.eda.advanced else None,
        "druglikeness": "eda/druglikeness.csv" if drug_panel_enabled else None,
        "nearest_neighbors": "eda/nearest_neighbors.csv",
        "activity_cliffs": "eda/activity_cliffs.csv",
        "local_property_discontinuities": (
            "eda/local_property_discontinuities.csv" if config.eda.advanced else None
        ),
        "model_readiness": (
            "eda/model_readiness.json" if model_readiness is not None else None
        ),
        "n_activity_cliffs": int(len(activity_cliffs)),
        "n_local_property_discontinuities": int(len(discontinuities)),
        "artifact_paths": [
            "eda_report.html",
            "eda/dataset_profile.json",
            "eda/molecule_table.csv",
            "eda/property_summary.json",
            "eda/descriptor_summary_eda.csv",
            "eda/descriptor_outliers_eda.csv",
            "eda/scaffold_summary_eda.csv",
            "eda/cluster_summary_eda.csv",
            "eda/nearest_neighbors.csv",
            "eda/activity_cliffs.csv",
            "eda/gallery_candidates.json",
            "eda/molecule_svgs/",
            *advanced_artifacts,
        ],
    }
