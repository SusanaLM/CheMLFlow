import json
from pathlib import Path

import numpy as np
import pandas as pd

from MLModels import train_models


def test_train_model_writes_split_metrics_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(36, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(2.5 * X["f0"] - 0.8 * X["f1"] + rng.normal(scale=0.2, size=len(X)))

    X_train = X.iloc[:24]
    y_train = y.iloc[:24]
    X_val = X.iloc[24:30]
    y_val = y.iloc[24:30]
    X_test = X.iloc[30:]
    y_test = y.iloc[30:]

    _, train_result = train_models.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="decision_tree",
        output_dir=str(tmp_path),
        cv_folds=2,
        search_iters=1,
        task_type="regression",
        model_config={"plot_split_performance": True, "n_jobs": 1},
        X_val=X_val,
        y_val=y_val,
    )

    metrics = json.loads(Path(train_result.metrics_path).read_text(encoding="utf-8"))
    split_metrics_path = metrics.get("split_metrics_path")
    split_plot_path = metrics.get("split_metrics_plot_path")

    assert split_metrics_path is not None
    assert split_plot_path is not None
    assert Path(split_metrics_path).exists()
    assert Path(split_plot_path).exists()
    parity_train_path = metrics.get("parity_plot_train_path")
    parity_val_path = metrics.get("parity_plot_val_path")
    parity_test_path = metrics.get("parity_plot_test_path")
    parity_all_splits_path = metrics.get("parity_plot_all_splits_path")
    assert parity_train_path is not None
    assert parity_val_path is not None
    assert parity_test_path is not None
    assert parity_all_splits_path is not None
    assert Path(parity_train_path).exists()
    assert Path(parity_val_path).exists()
    assert Path(parity_test_path).exists()
    assert Path(parity_all_splits_path).exists()

    split_metrics = json.loads(Path(split_metrics_path).read_text(encoding="utf-8"))
    assert set(split_metrics.keys()) == {"train", "val", "test"}
    assert {"r2", "mae"}.issubset(split_metrics["train"].keys())


def test_classification_split_plot_artifacts(tmp_path):
    split_outputs = {
        "train": {
            "y_true": np.array([0, 0, 1, 1, 1, 0]),
            "y_proba": np.array([0.05, 0.2, 0.7, 0.8, 0.9, 0.3]),
            "y_pred": np.array([0, 0, 1, 1, 1, 0]),
        },
        "val": {
            "y_true": np.array([0, 1, 0, 1]),
            "y_proba": np.array([0.15, 0.75, 0.35, 0.65]),
            "y_pred": np.array([0, 1, 0, 1]),
        },
        "test": {
            "y_true": np.array([0, 1, 0, 1, 1]),
            "y_proba": np.array([0.12, 0.61, 0.44, 0.79, 0.83]),
            "y_pred": np.array([0, 1, 0, 1, 1]),
        },
    }

    paths = train_models._save_classification_split_plots(
        output_dir=str(tmp_path),
        model_type="catboost_classifier",
        split_outputs=split_outputs,
    )

    expected_keys = {
        "pr_curve_train_path",
        "pr_curve_val_path",
        "pr_curve_test_path",
        "pr_curve_all_splits_path",
        "confusion_matrix_train_path",
        "confusion_matrix_val_path",
        "confusion_matrix_test_path",
        "confusion_matrix_all_splits_path",
    }
    assert expected_keys.issubset(paths.keys())
    for key in expected_keys:
        assert Path(paths[key]).exists()
