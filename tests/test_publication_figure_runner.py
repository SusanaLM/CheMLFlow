from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from molecular_landscape.io_utils import sha256_file, write_artifact_manifest
from publication_figures import plot_chemical_space_annotated, plot_space_panels
from publication_figures.runner import run_publication_figures
from utilities.molecular_analysis import run_publication_figures_node


def _source_bundle(root: Path) -> Path:
    source = root / "molecular_eda"
    eda = source / "eda"
    eda.mkdir(parents=True)
    pd.DataFrame(
        {
            "compound_id": ["A", "B", "C", "D"],
            "canonical_smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "pIC50": [5.0, 6.0, 7.0, 8.0],
            "structure_umap_x": [0.0, 1.0, 0.5, 1.5],
            "structure_umap_y": [0.0, 0.2, 1.0, 1.2],
            "has_valid_property": [True, False, "true", "false"],
            "activity_class": ["inactive", "intermediate", "active", "active"],
            "QED": [0.3, 0.4, 0.7, 0.8],
            "Lipinski_Violation_Count": [0, 1, 0, 2],
        }
    ).to_csv(eda / "molecule_table.csv", index=False)
    pd.DataFrame(
        {
            "tanimoto_similarity": [0.75],
            "absolute_property_difference": [1.5],
            "same_scaffold": [True],
            "collision_derived_match": [False],
        }
    ).to_csv(eda / "activity_cliffs.csv", index=False)
    manifest = {
        "status": "complete",
        "workflow_version": "0.5.0",
        "completed_at": "2026-07-22T00:00:00+00:00",
        "input": {"path": "curated.csv", "sha256": "a" * 64},
        "analysis_identity": {"molecular_eda_config_sha256": "b" * 64},
        "result": {"schema": {"property_cols": ["pIC50"]}},
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_artifact_manifest(source, source / "artifact_manifest.csv")
    return source


def test_auxiliary_publication_modules_import_as_package_modules() -> None:
    assert callable(plot_chemical_space_annotated.main)
    assert callable(plot_space_panels.main)


def test_runner_renders_only_selected_figures_with_persisted_provenance(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "figures"
    result = run_publication_figures(
        exports_dir=str(source),
        output_dir=str(output),
        figures=["qed"],
        formats=["png", "svg"],
    )
    assert result["n_figures"] == 1
    assert (output / "qed.png").is_file()
    assert "molecular EDA post-curation cohort" in (
        output / "qed.svg"
    ).read_text(encoding="utf-8")
    assert not (output / "chemical_space.png").exists()
    persisted = json.loads(
        (output / "publication_figures_manifest.json").read_text(encoding="utf-8")
    )
    assert str(tmp_path) not in json.dumps(persisted)
    assert persisted["output_dir"] == "figures"
    assert persisted["source"]["molecular_eda_dir"] == "molecular_eda"
    assert persisted["source"]["run_manifest"] == "run_manifest.json"
    assert persisted["rendered_configs"]["qed"]["input"]["run_dir"] == "molecular_eda"
    artifact = persisted["produced"][0]["files"][0]
    assert artifact["sha256"] == sha256_file(output / "qed.png")
    assert persisted["source"]["run_manifest_sha256"] == sha256_file(
        source / "run_manifest.json"
    )
    assert persisted["produced"][0]["accounting"]["rows_plotted"] == 2
    listed = pd.read_csv(output / "artifact_manifest.csv")["relative_path"].tolist()
    assert "qed.png" in listed
    assert "qed.svg" in listed
    assert "publication_figures_manifest.json" in listed


def test_runner_errors_on_missing_required_source_by_default(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    (source / "eda" / "activity_cliffs.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Manifest-listed artifact is missing"):
        run_publication_figures(
            exports_dir=str(source),
            output_dir=str(tmp_path / "figures"),
            figures=["activity_discontinuities"],
            formats=["png"],
        )


def test_native_figure_node_requires_explicit_selection(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    context = {
        "run_dir": str(tmp_path / "run"),
        "molecular_eda_dir": str(source),
        "analyze_config": {"publication_figures": {}},
    }
    with pytest.raises(ValueError, match="must explicitly list"):
        run_publication_figures_node(context)
