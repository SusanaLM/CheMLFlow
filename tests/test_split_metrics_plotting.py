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
