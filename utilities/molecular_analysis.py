"""Native CheMLFlow adapters for optional, dataset-scoped molecular analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from molecular_landscape.config import (
    ClusteringConfig,
    EmbeddingConfig,
    FingerprintConfig,
    WorkflowConfig,
)
from molecular_landscape.eda.config import EDAConfig
from molecular_landscape.schema import is_activity_property, is_log_activity_property
from publication_figures import run_publication_figures


_SMILES_ALIASES = ("canonical_smiles", "smiles", "SMILES", "mol_smiles")
_ID_ALIASES = (
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "mol_id",
    "compound_id",
    "id",
    "name",
    "__row_index",
)
_MAP_METHODS = {"pca", "umap", "tsne", "pacmap", "trimap"}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _molecular_config(context: dict[str, Any]) -> dict[str, Any]:
    analyze = _mapping(context.get("analyze_config"), "analyze")
    return _mapping(analyze.get("molecular_eda"), "analyze.molecular_eda")


def _figures_config(context: dict[str, Any]) -> dict[str, Any]:
    analyze = _mapping(context.get("analyze_config"), "analyze")
    return _mapping(
        analyze.get("publication_figures"), "analyze.publication_figures"
    )


def _resolve_input_path(context: dict[str, Any], config: dict[str, Any]) -> Path:
    paths = context.get("paths") or {}
    candidates = [
        config.get("input_path"),
        context.get("curated_path"),
        paths.get("curated"),
        paths.get("raw"),
    ]
    for candidate in candidates:
        if candidate and Path(str(candidate)).is_file():
            return Path(str(candidate)).resolve()
    raise FileNotFoundError(
        "analyze.molecular_eda could not locate an input CSV; set "
        "analyze.molecular_eda.input_path or run it after get_data/curate/label."
    )


def _string_list(value: Any, path: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        raise ValueError(f"{path} must be a string or list of strings.")
    normalized = [item.strip() for item in values if item.strip()]
    if not normalized:
        raise ValueError(f"{path} must not be empty.")
    return normalized


def _resolve_properties(
    config: dict[str, Any], context: dict[str, Any], columns: list[str]
) -> list[str] | None:
    requested = config.get("property_columns", config.get("property_column"))
    if requested is not None:
        properties = _string_list(requested, "analyze.molecular_eda.property_columns")
    else:
        target = str(context.get("target_column") or "").strip()
        properties = [target] if target and target in columns else []
    missing = [column for column in properties if column not in columns]
    if missing:
        raise ValueError(f"Molecular EDA property columns were not found: {missing}")
    return properties or None


def _validate_property_semantics(
    frame: pd.DataFrame,
    properties: list[str] | None,
    property_type: str,
    config: dict[str, Any],
) -> None:
    if not properties:
        return
    primary = properties[0]
    if property_type == "auto" and is_activity_property(
            primary) and not is_log_activity_property(primary):
        raise ValueError(
            f"Raw activity property '{primary}' requires an explicit property_type "
            "(normally potency_linear) and a homogeneous units column. CheMLFlow "
            "does not guess IC50 units or convert them to pIC50 in this node."
        )
    if property_type == "potency_log" and is_activity_property(
            primary) and not is_log_activity_property(primary):
        raise ValueError(
            f"Property '{primary}' looks like a linear activity value but "
            "property_type is potency_log. Run label.ic50 or select the actual "
            "log-potency column instead of relabelling raw measurements."
        )
    if property_type == "potency_linear" and is_log_activity_property(primary):
        raise ValueError(
            f"Property '{primary}' looks logarithmic but property_type is potency_linear."
        )
    if property_type != "potency_linear":
        return
    units_column = str(config.get("units_column") or "").strip()
    if not units_column:
        units_column = next(
            (
                column
                for column in ("standard_units", "qudt_units", "units")
                if column in frame.columns and frame[column].notna().any()
            ),
            "",
        )
    if not units_column or units_column not in frame.columns:
        raise ValueError(
            "potency_linear molecular EDA requires a populated units_column; "
            "prefer standardized units or run label.ic50 and analyze pIC50."
        )
    units = frame[units_column].dropna().astype(str).str.strip()
    units = sorted(value for value in units.unique() if value)
    if len(units) != 1:
        raise ValueError(
            f"potency_linear property '{primary}' must use one homogeneous unit; "
            f"{units_column} contains {units or ['<missing>']}."
        )


def build_molecular_workflow_config(context: dict[str, Any]) -> WorkflowConfig:
    """Translate a native CheMLFlow context into a validated workflow config."""
    config = _molecular_config(context)
    input_path = _resolve_input_path(context, config)
    header = pd.read_csv(input_path, nrows=0)
    columns = list(header.columns)
    smiles_col = str(config.get("smiles_column") or "").strip() or next(
        (column for column in _SMILES_ALIASES if column in columns), ""
    )
    if not smiles_col:
        raise ValueError("Molecular EDA input has no recognized SMILES column.")
    id_col = str(config.get("id_column") or "").strip() or next(
        (column for column in _ID_ALIASES if column in columns), ""
    )
    properties = _resolve_properties(config, context, columns)
    property_type = str(config.get("property_type", "auto")).strip().lower()
    if properties:
        needs_values = property_type == "potency_linear" or (
            property_type == "auto"
            and is_activity_property(properties[0])
            and not is_log_activity_property(properties[0])
        )
        semantic_frame = pd.read_csv(input_path) if needs_values else header
        _validate_property_semantics(
            semantic_frame, properties, property_type, config
        )

    map_methods = _string_list(
        config.get("map_methods", ["pca", "umap"]),
        "analyze.molecular_eda.map_methods",
    )
    unknown_maps = sorted(set(map_methods) - _MAP_METHODS)
    if unknown_maps:
        raise ValueError(f"Unsupported molecular EDA map methods: {unknown_maps}")
    primary_map = str(config.get("primary_map", "umap")).strip().lower()
    if primary_map not in map_methods:
        raise ValueError("analyze.molecular_eda.primary_map must be in map_methods.")

    embedding_cfg = _mapping(config.get("embedding"), "analyze.molecular_eda.embedding")
    clustering_cfg = _mapping(config.get("clustering"), "analyze.molecular_eda.clustering")
    fingerprint_cfg = _mapping(config.get("fingerprint"), "analyze.molecular_eda.fingerprint")
    report_cfg = _mapping(config.get("report"), "analyze.molecular_eda.report")
    run_dir = Path(str(context.get("run_dir") or os.getcwd())).resolve()
    output_dir = Path(str(config.get("output_dir") or run_dir / "molecular_eda")).resolve()
    random_state = int(embedding_cfg.get("random_state", context.get("global_random_state", 42)))

    return WorkflowConfig(
        input_path=input_path,
        output_dir=output_dir,
        smiles_col=smiles_col,
        id_col=id_col or None,
        property_cols=properties,
        property_transforms=dict(config.get("property_transforms") or {}),
        sample_size=config.get("sample_size"),
        overwrite=bool(config.get("overwrite", False)),
        provenance={
            "scope": "dataset_analysis",
            "cohort": "post-curation molecular cohort",
            "chemlflow_config_hash": context.get("config_hash"),
            "chemlflow_config_fingerprint": context.get("config_fingerprint"),
            "pipeline_nodes": list(context.get("pipeline_nodes") or []),
            "curation": context.get("curate_config") or {},
        },
        fingerprint=FingerprintConfig(
            radius=int(fingerprint_cfg.get("radius", 2)),
            n_bits=int(fingerprint_cfg.get("n_bits", 2048)),
            include_chirality=bool(fingerprint_cfg.get("include_chirality", True)),
            use_features=bool(fingerprint_cfg.get("use_features", False)),
            representation_sensitivity=bool(
                fingerprint_cfg.get(
                    "representation_sensitivity", False)),
            comparison_representations=list(
                fingerprint_cfg.get(
                    "comparison_representations", [
                        "fcfp", "rdkit", "atompair", "torsion", "maccs"])),
        ),
        embedding=EmbeddingConfig(
            property_weight=float(embedding_cfg.get("property_weight", 0.20)),
            property_weight_sensitivity=list(
                embedding_cfg.get(
                    "property_weight_sensitivity", [
                        0.10, 0.20, 0.30])),
            random_state=random_state,
            umap_seed_sensitivity=list(embedding_cfg.get("umap_seed_sensitivity", [7, 42, 99])),
            umap_neighbors=int(embedding_cfg.get("umap_neighbors", 30)),
            umap_min_dist=float(embedding_cfg.get("umap_min_dist", 0.10)),
            validation_neighbors=int(embedding_cfg.get("validation_neighbors", 15)),
            max_pairwise_molecules=int(embedding_cfg.get("max_pairwise_molecules", 5000)),
            include_tsne="tsne" in map_methods,
            tsne_perplexity=float(embedding_cfg.get("tsne_perplexity", 30.0)),
            include_pacmap="pacmap" in map_methods,
            include_trimap="trimap" in map_methods,
            pacmap_neighbors=int(embedding_cfg.get("pacmap_neighbors", 10)),
            map_method_selection=bool(embedding_cfg.get("map_method_selection", False)),
            coranking_diagnostics=bool(embedding_cfg.get("coranking_diagnostics", False)),
        ),
        clustering=ClusteringConfig(
            butina_similarity_threshold=float(
                clustering_cfg.get(
                    "butina_similarity_threshold", 0.65)),
            threshold_sensitivity=list(
                clustering_cfg.get(
                    "threshold_sensitivity", [
                        0.55, 0.65, 0.75])),
            hdbscan=bool(clustering_cfg.get("hdbscan", False)),
            hdbscan_min_cluster_size=int(clustering_cfg.get("hdbscan_min_cluster_size", 5)),
        ),
        eda=EDAConfig(
            enabled=True,
            advanced=bool(report_cfg.get("advanced", True)),
            open_report=False,
            property_type=property_type,
            higher_is_better=report_cfg.get("higher_is_better", "auto"),
            include_drug_discovery_panel=bool(report_cfg.get("drug_discovery_panel", True)),
            include_model_readiness=bool(report_cfg.get("model_readiness", True)),
            include_nearest_neighbors=bool(report_cfg.get("nearest_neighbors", True)),
            include_activity_cliffs=bool(report_cfg.get("activity_discontinuities", True)),
            include_property_descriptor_plots=bool(
                report_cfg.get("property_descriptor_plots", True)),
            top_scaffolds=int(report_cfg.get("top_scaffolds", 20)),
            singleton_scaffold_warning_fraction=float(
                report_cfg.get("singleton_scaffold_warning_fraction", 0.30)),
            representative_molecules=int(report_cfg.get("representative_molecules", 48)),
            max_svg_molecules=int(report_cfg.get("max_svg_molecules", 5000)),
            nearest_neighbors=int(report_cfg.get("nearest_neighbors_count", 10)),
            activity_cliff_similarity=float(report_cfg.get("activity_similarity_threshold", 0.70)),
            activity_cliff_delta=float(report_cfg.get("activity_difference_threshold", 1.0)),
            map_method=primary_map,
            qed_low_threshold=float(report_cfg.get("qed_low_threshold", 0.35)),
            lipinski_violation_warning_threshold=int(
                report_cfg.get("lipinski_violation_warning_threshold", 2)),
            max_points_for_svg_hover=int(report_cfg.get("max_detailed_hover_points", 5000)),
            use_scattergl=bool(report_cfg.get("use_scattergl", True)),
            export_selected_template=bool(report_cfg.get("export_selection_schema", True)),
        ),
    )


def run_molecular_eda_node(context: dict[str, Any]) -> dict[str, Any]:
    from contracts import (
        ANALYZE_MOLECULAR_EDA_INPUT_CONTRACT,
        ANALYZE_MOLECULAR_EDA_OUTPUT_CONTRACT,
        bind_output_path,
        validate_contract,
    )
    from molecular_landscape.io_utils import (
        configure_runtime_caches,
        verify_artifact_manifest,
    )

    config = build_molecular_workflow_config(context)
    configure_runtime_caches(config.output_dir)
    from molecular_landscape.workflow import run_workflow
    validate_contract(
        bind_output_path(ANALYZE_MOLECULAR_EDA_INPUT_CONTRACT, str(config.input_path)),
        warn_only=False,
    )
    manifest = run_workflow(config)
    verify_artifact_manifest(
        config.output_dir, config.output_dir / "artifact_manifest.csv"
    )
    validate_contract(
        bind_output_path(ANALYZE_MOLECULAR_EDA_OUTPUT_CONTRACT, str(config.output_dir)),
        warn_only=False,
    )
    context["molecular_eda_dir"] = str(config.output_dir)
    context["molecular_eda_report"] = str(config.output_dir / "eda_report.html")
    context["molecular_eda_manifest"] = str(config.output_dir / "run_manifest.json")
    return manifest


def run_publication_figures_node(context: dict[str, Any]) -> dict[str, Any]:
    from contracts import (
        ANALYZE_PUBLICATION_FIGURES_OUTPUT_CONTRACT,
        bind_output_path,
        validate_contract,
    )

    config = _figures_config(context)
    figures = config.get("figures")
    if figures is None:
        raise ValueError(
            "analyze.publication_figures.figures must explicitly list the selected figures."
        )
    selected = _string_list(figures, "analyze.publication_figures.figures")
    source_dir = config.get("source_dir") or context.get("molecular_eda_dir")
    if not source_dir:
        raise ValueError(
            "analyze.publication_figures needs a prior analyze.molecular_eda node "
            "or an explicit source_dir pointing to a completed molecular EDA bundle."
        )
    run_dir = Path(str(context.get("run_dir") or os.getcwd())).resolve()
    output_dir = Path(
        str(config.get("output_dir") or run_dir / "publication_figures")
    ).resolve()
    manifest = run_publication_figures(
        exports_dir=str(source_dir),
        output_dir=str(output_dir),
        figures=selected,
        formats=_string_list(
            config.get(
                "formats", [
                    "pdf", "svg", "png"]), "analyze.publication_figures.formats"),
        property_column=config.get("property_column"),
        overrides=_mapping(config.get("overrides"), "analyze.publication_figures.overrides"),
        on_missing=str(config.get("on_missing", "error")),
        overwrite=bool(config.get("overwrite", False)),
    )
    from molecular_landscape.io_utils import verify_artifact_manifest

    verify_artifact_manifest(output_dir, output_dir / "artifact_manifest.csv")
    validate_contract(
        bind_output_path(ANALYZE_PUBLICATION_FIGURES_OUTPUT_CONTRACT, str(output_dir)),
        warn_only=False,
    )
    context["publication_figures_dir"] = str(output_dir)
    context["publication_figures_manifest"] = str(
        output_dir / "publication_figures_manifest.json"
    )
    return manifest
