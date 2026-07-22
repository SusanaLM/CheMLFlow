from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main
from molecular_landscape.eda.neighbors import compute_neighbors_and_cliffs
from molecular_landscape.embedding import _neighbor_order_without_self
from molecular_landscape.io_utils import public_invocation, redact_host_paths
from utilities.config_validation import ConfigValidationError, validate_config_strict
from utilities.molecular_analysis import (
    build_molecular_workflow_config,
)
from utilities.doe import PROFILE_SPECS, _build_pipeline_nodes


SMILES = [
    "CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1",
    "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CC(=O)N",
    "C1CCCCC1", "c1ccncc1",
]


def _context(input_path: Path, run_dir: Path, **overrides) -> dict:
    molecular = {
        "input_path": str(input_path),
        "property_column": "pIC50",
        "property_type": "potency_log",
        "map_methods": ["pca", "umap"],
        "primary_map": "umap",
        "embedding": {
            "property_weight_sensitivity": [0.2],
            "umap_seed_sensitivity": [42],
            "umap_neighbors": 5,
            "validation_neighbors": 3,
            "max_pairwise_molecules": 100,
        },
        "clustering": {"threshold_sensitivity": [0.65]},
        "report": {
            "advanced": False,
            "drug_discovery_panel": False,
            "representative_molecules": 6,
            "max_svg_molecules": 20,
            "nearest_neighbors_count": 3,
        },
    }
    molecular.update(overrides)
    return {
        "run_dir": str(run_dir),
        "target_column": "pIC50",
        "global_random_state": 42,
        "pipeline_nodes": ["analyze.molecular_eda"],
        "analyze_config": {"molecular_eda": molecular},
        "paths": {},
        "curate_config": {"dedupe_strategy": "drop_conflicts"},
        "config_hash": "config-hash",
        "config_fingerprint": "config-fingerprint",
    }


def test_new_nodes_are_separate_from_existing_generic_eda() -> None:
    assert main.NODE_REGISTRY["analyze.eda"] is main.run_node_analyze_eda
    assert main.NODE_REGISTRY["analyze.molecular_eda"] is main.run_node_analyze_molecular_eda
    assert (
        main.NODE_REGISTRY["analyze.publication_figures"]
        is main.run_node_analyze_publication_figures
    )


def test_doe_pipeline_builder_never_adds_dataset_analysis_nodes() -> None:
    nodes = _build_pipeline_nodes(
        PROFILE_SPECS["reg_local_csv_ic50"],
        feature_input="featurize.morgan",
        preprocess_enabled=False,
        select_enabled=False,
        explain_enabled=False,
        label_normalize_enabled=False,
        analyze_stats_enabled=False,
        analyze_eda_enabled=True,
    )
    assert "analyze.eda" in nodes
    assert "analyze.molecular_eda" not in nodes
    assert "analyze.publication_figures" not in nodes


def test_config_requires_explicit_figure_selection_and_source() -> None:
    config = {
        "global": {
            "pipeline_type": "chembl",
            "base_dir": "data/example",
            "thresholds": {"active": 1, "inactive": 2},
        },
        "pipeline": {"nodes": ["analyze.publication_figures"]},
        "analyze": {"publication_figures": {}},
    }
    with pytest.raises(ConfigValidationError) as error:
        validate_config_strict(config, ["analyze.publication_figures"])
    message = str(error.value)
    assert "CFG_PUBLICATION_FIGURES_SELECTION_REQUIRED" in message
    assert "CFG_PUBLICATION_FIGURES_SOURCE_REQUIRED" in message


def test_pipeline_requires_figures_after_molecular_eda() -> None:
    with pytest.raises(ValueError, match="must follow"):
        main.validate_pipeline_nodes(
            ["analyze.publication_figures", "analyze.molecular_eda"]
        )


def test_raw_linear_activity_requires_homogeneous_units(tmp_path: Path) -> None:
    input_path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "canonical_smiles": SMILES,
            "compound_id": [f"M{index}" for index in range(len(SMILES))],
            "IC50": np.arange(1, len(SMILES) + 1),
            "standard_units": ["nM", "uM"] * 6,
        }
    ).to_csv(input_path, index=False)
    context = _context(
        input_path,
        tmp_path / "run",
        property_column="IC50",
        property_type="potency_linear",
    )
    with pytest.raises(ValueError, match="one homogeneous unit"):
        build_molecular_workflow_config(context)


def test_neighbor_tables_flag_folded_fingerprint_collisions() -> None:
    table = pd.DataFrame(
        {
            "structure_index": [0, 1, 2],
            "compound_id": ["A", "B", "C"],
            "canonical_smiles": ["CCO", "CCN", "c1ccccc1"],
            "pIC50": [5.0, 8.0, 6.0],
            "scaffold_id": [0, 0, 1],
            "fingerprint_collision": [True, True, False],
            "fingerprint_collision_group": pd.Series([1, 1, pd.NA], dtype="Int64"),
            "svg_path": [None, None, None],
        }
    )
    distance = np.array(
        [[0.0, 0.0, 0.8], [0.0, 0.0, 0.7], [0.8, 0.7, 0.0]], dtype=float
    )
    neighbors, discontinuities = compute_neighbors_and_cliffs(
        table, distance, "pIC50", 1, 0.7, 1.0
    )
    collision_rows = neighbors[neighbors["collision_derived_match"]]
    assert len(collision_rows) == 2
    assert discontinuities.iloc[0]["collision_derived_match"]


def test_tied_distances_never_return_self_as_neighbor() -> None:
    distances = np.array(
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.4], [0.5, 0.4, 0.0]], dtype=float
    )
    neighbors = _neighbor_order_without_self(distances, 1)
    assert neighbors[:, 0].tolist() == [1, 0, 1]


def test_public_manifest_helpers_remove_posix_and_windows_host_paths() -> None:
    payload = {
        "input": Path("/host-root/account/project/molecules.csv"),
        "windows": r"Q:\host-root\account\project\molecules.csv",
        "relative": "data/molecules.csv",
    }
    assert redact_host_paths(payload) == {
        "input": "molecules.csv",
        "windows": "molecules.csv",
        "relative": "data/molecules.csv",
    }
    invocation = public_invocation(
        [
            "/runtime/bin/python",
            "/workspace/project/cli.py",
            "--input=/workspace/project/molecules.csv",
            "--overwrite",
        ]
    )
    assert invocation == "python cli.py arguments-redacted"


def test_native_dataset_analysis_pipeline_is_end_to_end_and_uses_stable_row_ids(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "canonical_smiles": SMILES,
            "pIC50": [5.0 + index * 0.2 for index in range(len(SMILES))],
            "standard_relation": ["="] * len(SMILES),
        }
    ).to_csv(input_path, index=False)
    run_dir = tmp_path / "run"
    config = {
        "global": {
            "pipeline_type": "chembl",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data"),
            "run_dir": str(run_dir),
            "target_column": "pIC50",
            "random_state": 42,
            "thresholds": {"active": 1000, "inactive": 10000},
        },
        "pipeline": {
            "nodes": [
                "analyze.molecular_eda",
                "analyze.publication_figures",
            ]
        },
        "analyze": {
            "molecular_eda": _context(input_path, run_dir)["analyze_config"][
                "molecular_eda"
            ],
            "publication_figures": {
                "figures": ["chemical_space"],
                "formats": ["png"],
            },
        },
    }
    config_path = tmp_path / "molecular-analysis.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main.run_configured_pipeline_nodes(config, str(config_path))
    output = run_dir / "molecular_eda"
    persisted = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    molecule_table = pd.read_csv(output / "eda" / "molecule_table.csv")
    assert persisted["status"] == "complete"
    assert persisted["analysis_identity"]["implementation_version"] == "0.5.0"
    assert persisted["provenance"]["scope"] == "dataset_analysis"
    assert str(tmp_path) not in json.dumps(persisted)
    assert persisted["input"]["path"] == "molecules.csv"
    assert persisted["output"] == "molecular_eda"
    assert persisted["config"]["input_path"] == "molecules.csv"
    assert persisted["config"]["output_dir"] == "molecular_eda"
    assert persisted["privacy"]["host_paths"].startswith("absolute paths")
    assert persisted["result"]["schema"]["id_col"] == "__row_index"
    assert molecule_table["compound_id"].iloc[0] == "row_0"
    assert (output / "artifact_manifest.csv").is_file()
    assert (run_dir / "publication_figures" / "chemical_space.png").is_file()
    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "success"
    assert main.NODE_REGISTRY["analyze.eda"] is main.run_node_analyze_eda
