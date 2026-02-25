from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from utilities.doe import DOEGenerationError, generate_doe


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pgp_dataset_path() -> str:
    return str(REPO_ROOT / "tests" / "fixtures" / "data" / "pgp_small.csv")


def _flash_dataset_path() -> str:
    return str(REPO_ROOT / "tests" / "fixtures" / "data" / "flash_small.csv")


def _base_clf_doe(tmp_path: Path) -> dict:
    return {
        "version": 1,
        "dataset": {
            "profile": "clf_local_csv",
            "name": "pgp_small_doe",
            "task_type": "classification",
            "target_column": "label",
            "source": {"type": "local_csv", "path": _pgp_dataset_path()},
            "smiles_column": "SMILES",
            "label_source_column": "Activity",
            "label_map": {
                "positive": [1, "1", "active"],
                "negative": [0, "0", "inactive"],
            },
            "curate": {
                "properties": "Activity",
                "smiles_column": "SMILES",
                "dedupe_strategy": "drop_conflicts",
            },
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.morgan"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [True],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data"),
            "global.runs.enabled": False,
            "global.random_state": 42,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
            "split.require_disjoint": True,
            "split.require_full_test_coverage": True,
            "train.tuning.method": "fixed",
            "train.reporting.plot_split_performance": True,
        },
        "output": {"dir": str(tmp_path / "generated")},
    }


def test_generate_doe_skips_invalid_model_task_combos(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["catboost_classifier", "random_forest"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["total_cases"] == 2
    assert summary["valid_cases"] == 1
    assert summary["skipped_cases"] == 1
    assert summary["issue_counts"].get("DOE_MODEL_TASK_MISMATCH", 0) == 1
    assert len(result["valid_cases"]) == 1
    assert Path(result["valid_cases"][0]["config_path"]).exists()


def test_generate_doe_requires_validation_split_for_chemprop(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["none"]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.0

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["skipped_cases"] == 1
    assert summary["issue_counts"].get("DOE_VALIDATION_SPLIT_REQUIRED", 0) == 1


def test_generate_doe_propagates_chemprop_legacy_split_flag(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["none"]
    spec["search_space"]["pipeline.explain"] = [False]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["train.model.allow_legacy_split_positions"] = True

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["train"]["model"]["type"] == "chemprop"
    assert config["train"]["model"]["allow_legacy_split_positions"] is True


def test_generate_doe_skips_chemprop_with_preprocess_and_no_features(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["none"]
    spec["search_space"]["pipeline.preprocess"] = [True]
    spec["search_space"]["train.model.type"] = ["chemprop"]
    spec["defaults"]["split.val_size"] = 0.1

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_REQUIRED_FOR_PREPROCESS", 0) == 1
    assert summary["issue_counts"].get("DOE_CHEMPROP_PREPROCESS_UNSUPPORTED", 0) == 1


def test_generate_doe_enforces_split_mode_strategy_compatibility(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["tdc_scaffold"]
    spec["search_space"]["split.cv.n_splits"] = [5]
    spec["search_space"]["split.cv.repeats"] = [1]
    spec["search_space"]["split.cv.fold_index"] = [0]
    spec["search_space"]["split.cv.repeat_index"] = [0]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SPLIT_STRATEGY_MODE_INVALID", 0) == 1


def test_generate_doe_validates_cv_fold_index_bounds(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["split.mode"] = ["cv"]
    spec["search_space"]["split.strategy"] = ["random"]
    spec["search_space"]["split.cv.n_splits"] = [3]
    spec["search_space"]["split.cv.repeats"] = [1]
    spec["search_space"]["split.cv.fold_index"] = [3]
    spec["search_space"]["split.cv.repeat_index"] = [0]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SPLIT_PARAM_INVALID", 0) >= 1


def test_generate_doe_supports_auto_task_with_confirmation(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated_auto"
    spec = {
        "version": 1,
        "dataset": {
            "name": "pgp_auto",
            "task_type": "auto",
            "auto_confirmed": True,
            "target_column": "Activity",
            "source": {"type": "local_csv", "path": _pgp_dataset_path()},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "pipeline.preprocess": [False],
            "pipeline.select": [False],
            "pipeline.explain": [False],
            "pipeline.label_normalize": [False],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "data_auto"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
        },
        "output": {"dir": str(output_dir)},
    }

    result = generate_doe(spec)
    summary = result["summary"]
    assert summary["task_type"] == "classification"
    assert summary["profile"] == "clf_local_csv"
    assert summary["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["task_type"] == "classification"
    assert config["train"]["model"]["type"] == "catboost_classifier"

    manifest_lines = Path(result["manifest_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1
    payload = json.loads(manifest_lines[0])
    assert payload["status"] == "valid"


def test_generate_doe_isolates_case_artifacts_by_default(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["train.model.type"] = ["catboost_classifier", "dl_simple"]
    spec["defaults"]["global.base_dir"] = str(tmp_path / "data_root")
    spec["defaults"]["global.run_dir"] = str(tmp_path / "runs_root")

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 2

    config_paths = [Path(case["config_path"]) for case in result["valid_cases"]]
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]

    base_dirs = [cfg["global"]["base_dir"] for cfg in configs]
    run_dirs = [cfg["global"]["run_dir"] for cfg in configs]
    run_ids = [cfg["global"]["runs"]["id"] for cfg in configs]
    namespace = result["summary"]["doe_spec_hash"][:8]

    assert len(set(base_dirs)) == 2
    assert len(set(run_dirs)) == 2
    assert len(set(run_ids)) == 2
    assert any(path.endswith("case_0001") for path in base_dirs)
    assert any(path.endswith("case_0002") for path in base_dirs)
    assert all(Path(path).parts[-2] == namespace for path in base_dirs)
    assert all(Path(path).parts[-2] == namespace for path in run_dirs)


def test_generate_doe_isolation_namespaces_paths_by_spec_hash(tmp_path: Path) -> None:
    spec_a = _base_clf_doe(tmp_path)
    spec_a["search_space"]["train.model.type"] = ["catboost_classifier"]
    spec_a["defaults"]["global.base_dir"] = str(tmp_path / "shared_data")
    spec_a["defaults"]["global.run_dir"] = str(tmp_path / "shared_runs")
    spec_a["output"]["dir"] = str(tmp_path / "generated_a")

    spec_b = _base_clf_doe(tmp_path)
    spec_b["search_space"]["train.model.type"] = ["dl_simple"]
    spec_b["defaults"]["global.base_dir"] = str(tmp_path / "shared_data")
    spec_b["defaults"]["global.run_dir"] = str(tmp_path / "shared_runs")
    spec_b["output"]["dir"] = str(tmp_path / "generated_b")

    result_a = generate_doe(spec_a)
    result_b = generate_doe(spec_b)

    cfg_a = yaml.safe_load(Path(result_a["valid_cases"][0]["config_path"]).read_text(encoding="utf-8"))
    cfg_b = yaml.safe_load(Path(result_b["valid_cases"][0]["config_path"]).read_text(encoding="utf-8"))

    ns_a = result_a["summary"]["doe_spec_hash"][:8]
    ns_b = result_b["summary"]["doe_spec_hash"][:8]

    assert ns_a != ns_b
    assert Path(cfg_a["global"]["base_dir"]).parts[-2] == ns_a
    assert Path(cfg_b["global"]["base_dir"]).parts[-2] == ns_b
    assert cfg_a["global"]["base_dir"] != cfg_b["global"]["base_dir"]
    assert cfg_a["global"]["run_dir"] != cfg_b["global"]["run_dir"]


def test_generate_doe_validates_dataset_columns(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["dataset"]["smiles_column"] = "DOES_NOT_EXIST"
    spec["dataset"]["label_source_column"] = "MISSING_LABEL_SOURCE"
    spec["search_space"]["pipeline.label_normalize"] = [True]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_DATASET_COLUMN_MISSING", 0) >= 1


def test_generate_doe_defaults_feature_input_for_non_chemprop_classification(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"].pop("pipeline.feature_input")

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "featurize.morgan" in config["pipeline"]["nodes"]
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_REQUIRED", 0) == 0


def test_generate_doe_requires_regression_target_column_for_local_csv(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "reg_missing_target"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_reg_missing_target")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_TARGET_COLUMN_MISSING", 0) == 1


def test_generate_doe_reg_chembl_defaults_target_column_to_pic50(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_chembl_ic50",
            "task_type": "regression",
            "source": {"type": "chembl", "target_name": "Urease"},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "chembl_default_target"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_chembl_default_target")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["global"]["target_column"] == "pIC50"


def test_generate_doe_validates_invalid_dedupe_strategy(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.dedupe_strategy"] = "keepfirst_typo"

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CURATE_DEDUPE_INVALID", 0) == 1


def test_generate_doe_propagates_curate_drop_controls(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.drop_missing_smiles"] = False
    spec["defaults"]["curate.drop_invalid_smiles"] = False
    spec["defaults"]["curate.drop_missing_target"] = False
    spec["defaults"]["curate.required_non_null_columns"] = "SMILES,Activity"

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    curate_cfg = config["curate"]
    assert curate_cfg["drop_missing_smiles"] is False
    assert curate_cfg["drop_invalid_smiles"] is False
    assert curate_cfg["drop_missing_target"] is False
    assert curate_cfg["required_non_null_columns"] == ["SMILES", "Activity"]


def test_generate_doe_rejects_missing_required_non_null_columns(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.required_non_null_columns"] = ["does_not_exist"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CURATE_REQUIRED_COLUMNS_MISSING", 0) == 1


def test_generate_doe_allows_required_non_null_canonical_smiles_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["defaults"]["curate.required_non_null_columns"] = ["canonical_smiles", "Activity"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_CURATE_REQUIRED_COLUMNS_MISSING", 0) == 0


def test_generate_doe_writes_spec_snapshot_and_hash(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    result = generate_doe(spec)
    summary = result["summary"]

    assert summary.get("doe_spec_hash")
    snapshot_path = Path(summary["doe_spec_snapshot_path"])
    assert snapshot_path.exists()
    assert snapshot_path.name == "doe_spec.input.yaml"


def test_generate_doe_auto_task_detects_float_binary_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "float_binary.csv"
    csv_path.write_text(
        "SMILES,label\nCC,0.0\nCCC,1.0\nCCCC,0.0\n",
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "dataset": {
            "name": "float_binary_auto",
            "task_type": "auto",
            "auto_confirmed": True,
            "target_column": "label",
            "source": {"type": "local_csv", "path": str(csv_path)},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "train.model.type": ["catboost_classifier"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "float_binary_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "split.stratify": True,
        },
        "output": {"dir": str(tmp_path / "generated_float_binary")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["task_type"] == "classification"
    assert summary["valid_cases"] == 1


def test_generate_doe_auto_resolves_smiles_column_for_local_csv(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "auto_smiles_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_auto_smiles")},
    }

    result = generate_doe(spec)
    assert result["summary"]["valid_cases"] == 1

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["curate"]["smiles_column"] == "SMILES"


def test_generate_doe_rejects_unresolvable_smiles_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "no_smiles.csv"
    csv_path.write_text("structure,target\nCC,0.1\nCCC,0.2\n", encoding="utf-8")
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "target",
            "source": {"type": "local_csv", "path": str(csv_path)},
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "no_smiles_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
        },
        "output": {"dir": str(tmp_path / "generated_no_smiles")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_SMILES_COLUMN_MISSING", 0) == 1


def test_generate_doe_rejects_regression_when_curate_drops_target(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "missing_target_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "curate.properties": ["FP Calc."],
        },
        "output": {"dir": str(tmp_path / "generated_missing_target")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 0
    assert summary["issue_counts"].get("DOE_CURATE_TARGET_DROPPED", 0) == 1


def test_generate_doe_allows_regression_target_when_keep_all_columns(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.rdkit"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "keep_all_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "curate.properties": ["FP Calc."],
            "curate.keep_all_columns": True,
        },
        "output": {"dir": str(tmp_path / "generated_keep_all")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_CURATE_TARGET_DROPPED", 0) == 0


def test_generate_doe_featurize_none_defaults_keep_all_columns(tmp_path: Path) -> None:
    spec = {
        "version": 1,
        "dataset": {
            "profile": "reg_local_csv",
            "task_type": "regression",
            "target_column": "FP Exp.",
            "source": {"type": "local_csv", "path": _flash_dataset_path()},
            "smiles_column": "SMILES",
        },
        "search_space": {
            "split.mode": ["holdout"],
            "split.strategy": ["random"],
            "pipeline.feature_input": ["featurize.none"],
            "train.model.type": ["random_forest"],
        },
        "defaults": {
            "global.base_dir": str(tmp_path / "default_keep_all_data"),
            "global.runs.enabled": False,
            "split.test_size": 0.2,
            "split.val_size": 0.1,
            "curate.properties": ["FP Calc."],
        },
        "output": {"dir": str(tmp_path / "generated_default_keep_all")},
    }

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_CURATE_TARGET_DROPPED", 0) == 0

    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert (config.get("curate") or {}).get("keep_all_columns") is True


def test_generate_doe_accepts_rdkit_labeled_feature_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["featurize.rdkit_labeled"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    assert summary["issue_counts"].get("DOE_FEATURE_INPUT_NOT_SUPPORTED", 0) == 0
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "featurize.rdkit_labeled" in config["pipeline"]["nodes"]


def test_generate_doe_normalizes_legacy_curated_feature_alias(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"]["pipeline.feature_input"] = ["use.curated_features"]

    result = generate_doe(spec)
    summary = result["summary"]

    assert summary["valid_cases"] == 1
    config_path = Path(result["valid_cases"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "featurize.none" in config["pipeline"]["nodes"]
    assert "use.curated_features" not in config["pipeline"]["nodes"]


def test_generate_doe_requires_max_cases_for_large_grid(tmp_path: Path) -> None:
    spec = _base_clf_doe(tmp_path)
    spec["search_space"] = {
        "axis.a": list(range(101)),
        "axis.b": list(range(101)),
    }

    with pytest.raises(DOEGenerationError, match="constraints.max_cases"):
        generate_doe(spec)
