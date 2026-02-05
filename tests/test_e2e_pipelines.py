import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "data"


def _run_pipeline(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["CHEMLFLOW_CONFIG"] = str(config_path)
    python = os.environ.get("CHEMLFLOW_PYTHON", sys.executable)
    result = subprocess.run(
        [python, "main.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Pipeline failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return Path(config["global"]["run_dir"])


def _assert_metrics(run_dir: Path, model_type: str, keys: List[str]) -> None:
    metrics_path = run_dir / f"{model_type}_metrics.json"
    assert metrics_path.exists(), f"Missing metrics file: {metrics_path}"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in keys:
        assert key in metrics, f"Missing metric '{key}' in {metrics_path}"


def test_e2e_qm9_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_qm9"
    config = {
        "global": {
            "pipeline_type": "qm9",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data_qm9"),
            "target_column": "gap",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {"nodes": ["get_data", "curate", "featurize.rdkit", "train"]},
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "qm9_small.csv")},
        },
        "curate": {"properties": "gap"},
        "model": {
            "type": "decision_tree",
            "cv_folds": 2,
            "search_iters": 5,
            "use_hpo": False,
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "decision_tree", ["r2", "mae"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_ara_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_ara"
    config = {
        "global": {
            "pipeline_type": "ara",
            "task_type": "classification",
            "base_dir": str(tmp_path / "data_ara"),
            "target_column": "label",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "label.normalize", "split", "featurize.morgan", "train"]
        },
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "ara_small.csv")},
        },
        "curate": {
            "properties": "Activity",
            "smiles_column": "Smiles",
            "dedupe_strategy": "drop_conflicts",
            "label_column": "Activity",
            "prefer_largest_fragment": True,
        },
        "label": {
            "source_column": "Activity",
            "target_column": "label",
            "positive": ["active"],
            "negative": ["inactive"],
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": True,
        },
        "featurize": {"radius": 2, "n_bits": 128},
        "model": {
            "type": "catboost_classifier",
            "params": {
                "depth": 4,
                "learning_rate": 0.1,
                "iterations": 10,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": False,
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "catboost_classifier", ["auc", "auprc", "accuracy", "f1"])
    assert (out_dir / "run_config.yaml").exists()


def test_e2e_pgp_fast(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_pgp"
    config = {
        "global": {
            "pipeline_type": "adme",
            "task_type": "classification",
            "base_dir": str(tmp_path / "data_pgp"),
            "target_column": "label",
            "thresholds": {"active": 1000, "inactive": 10000},
            "run_dir": str(run_dir),
        },
        "pipeline": {
            "nodes": ["get_data", "curate", "label.normalize", "split", "featurize.morgan", "train"]
        },
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(FIXTURES / "pgp_small.csv")},
        },
        "curate": {
            "properties": "Activity",
            "smiles_column": "SMILES",
            "dedupe_strategy": "drop_conflicts",
            "label_column": "Activity",
            "prefer_largest_fragment": True,
        },
        "label": {
            "source_column": "Activity",
            "target_column": "label",
            "positive": ["1", 1],
            "negative": ["0", 0],
        },
        "split": {
            "strategy": "random",
            "test_size": 0.2,
            "val_size": 0.0,
            "random_state": 42,
            "stratify": True,
        },
        "featurize": {"radius": 2, "n_bits": 128},
        "model": {
            "type": "catboost_classifier",
            "params": {
                "depth": 4,
                "learning_rate": 0.1,
                "iterations": 10,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": False,
            },
        },
    }

    out_dir = _run_pipeline(tmp_path, config)
    _assert_metrics(out_dir, "catboost_classifier", ["auc", "auprc", "accuracy", "f1"])
    assert (out_dir / "run_config.yaml").exists()
