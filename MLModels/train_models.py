import json
import os
import logging
import re
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd

from MLModels.training import config as training_config
from MLModels.training import chemprop_models as training_chemprop_models
from MLModels.training import explainability as training_explainability
from MLModels.training import model_factory as training_model_factory
from MLModels.training import metrics as training_metrics
from MLModels.training import persistence as training_persistence
from MLModels.training import plots as training_plots
from MLModels.training import torch_models as training_torch_models
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
)

_ROW_INDEX_COL = "__row_index"


@dataclass
class TrainResult:
    model_path: str
    params_path: str
    metrics_path: str

@dataclass
class DLSearchConfig:
    model_class: Callable
    search_space: Dict[str, Any]
    default_params: Dict[str, Any]

def _ensure_dir(path: str) -> None:
    training_persistence.ensure_dir(path)


_XGBOOST_INVALID_FEATURE_CHARS = re.compile(r"[\[\]<>]")


def _sanitize_xgboost_feature_columns(columns: pd.Index) -> tuple[list[str], list[dict[str, Any]], int]:
    sanitized: list[str] = []
    records: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    changed_count = 0

    for idx, col in enumerate(columns):
        original = str(col)
        base = _XGBOOST_INVALID_FEATURE_CHARS.sub("_", original) or "feature"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used[candidate] = 1
        if candidate != original:
            changed_count += 1
        sanitized.append(candidate)
        records.append(
            {
                "index": idx,
                "original": original,
                "sanitized": candidate,
                "changed": candidate != original,
            }
        )
    return sanitized, records, changed_count


def _assign_feature_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame | None:
    if df is None:
        return None
    if df.shape[1] != len(columns):
        raise ValueError(
            "Feature column mismatch while applying sanitized feature names: "
            f"expected {len(columns)} columns, got {df.shape[1]}."
        )
    out = df.copy()
    out.columns = columns
    return out


def _sanitize_xgboost_feature_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    sanitized_columns, records, changed_count = _sanitize_xgboost_feature_columns(X_train.columns)
    payload = {
        "sanitized_columns": sanitized_columns,
        "columns": records,
        "changed_count": changed_count,
        "invalid_char_pattern": r"[\[\]<>]",
    }
    return (
        _assign_feature_columns(X_train, sanitized_columns),
        _assign_feature_columns(X_test, sanitized_columns),
        _assign_feature_columns(X_val, sanitized_columns),
        payload,
    )


def _maybe_sanitize_xgboost_feature_frames(
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str | None]:
    if model_type not in {"xgboost", "ensemble"}:
        return X_train, X_test, X_val, None

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        return X_train, X_test, X_val, None

    X_train_s, X_test_s, X_val_s, payload = _sanitize_xgboost_feature_frames(X_train, X_test, X_val)
    map_path = os.path.join(output_dir, f"{model_type}_feature_name_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    if payload["changed_count"] > 0:
        logging.info(
            "Sanitized %d feature names for %s to satisfy XGBoost constraints. map=%s",
            payload["changed_count"],
            model_type,
            map_path,
        )
    return X_train_s, X_test_s, X_val_s, map_path


def _maybe_apply_xgboost_feature_map_for_explain(
    model_type: str,
    output_dir: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_type not in {"xgboost", "ensemble"}:
        return X_train, X_test

    map_path = os.path.join(output_dir, f"{model_type}_feature_name_map.json")
    columns: list[str] | None = None
    if os.path.exists(map_path):
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            mapped = payload.get("sanitized_columns")
            if isinstance(mapped, list) and mapped:
                columns = [str(col) for col in mapped]
        except Exception as exc:
            logging.warning("Failed to load feature-name map at %s: %s", map_path, exc)

    if columns is None:
        columns, _, _ = _sanitize_xgboost_feature_columns(X_train.columns)

    X_train_s = _assign_feature_columns(X_train, columns)
    X_test_s = _assign_feature_columns(X_test, columns)
    assert X_train_s is not None and X_test_s is not None
    return X_train_s, X_test_s

def _resolve_n_jobs(model_config: Dict[str, Any] | None = None) -> int:
    return training_config.resolve_n_jobs(model_config)


def _validate_classification_score_values(
    y_score: Any,
    *,
    context: str,
) -> np.ndarray:
    return training_metrics.validate_classification_score_values(y_score, context=context)


def _safe_auc(y_true, y_score) -> float | None:
    return training_metrics.safe_auc(y_true, y_score)


def _safe_auprc(y_true, y_score) -> float | None:
    return training_metrics.safe_auprc(y_true, y_score)


def _save_roc_curve(output_dir: str, model_type: str, y_true, y_score) -> str | None:
    return training_plots.save_roc_curve(output_dir, model_type, y_true, y_score)


def _sigmoid(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return 1.0 / (1.0 + np.exp(-arr))


def _predict_classification_outputs(
    estimator: object,
    model_type: str,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_type.startswith("dl_"):
        logits = _validate_classification_score_values(
            _predict_dl(estimator, X.values),
            context=f"{model_type} classification raw scores",
        )
        y_proba = _sigmoid(logits)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_proba, y_pred, logits

    if hasattr(estimator, "predict_proba"):
        proba_raw = np.asarray(estimator.predict_proba(X))
        if proba_raw.ndim == 2:
            if proba_raw.shape[1] < 2:
                classes = np.asarray(getattr(estimator, "classes_", []))
                if classes.size == 1 and int(classes[0]) == 1:
                    y_proba = np.ones(proba_raw.shape[0], dtype=float)
                else:
                    y_proba = np.zeros(proba_raw.shape[0], dtype=float)
            else:
                y_proba = proba_raw[:, 1].reshape(-1).astype(float)
        else:
            y_proba = proba_raw.reshape(-1).astype(float)
        y_proba = _validate_classification_score_values(
            y_proba,
            context=f"{model_type} classification probabilities",
        )
        y_pred = _ensure_binary_labels(pd.Series(estimator.predict(X))).to_numpy(dtype=int)
        return y_proba, y_pred, y_proba

    if hasattr(estimator, "decision_function"):
        scores_raw = np.asarray(estimator.decision_function(X))
        if scores_raw.ndim == 2 and scores_raw.shape[1] >= 2:
            scores = scores_raw[:, 1].reshape(-1).astype(float)
        else:
            scores = scores_raw.reshape(-1).astype(float)
        scores = _validate_classification_score_values(
            scores,
            context=f"{model_type} classification raw scores",
        )
        y_proba = _sigmoid(scores)
        y_pred = _ensure_binary_labels(pd.Series(estimator.predict(X))).to_numpy(dtype=int)
        return y_proba, y_pred, scores

    y_pred = _ensure_binary_labels(pd.Series(estimator.predict(X))).to_numpy(dtype=int)
    y_proba = y_pred.astype(float)
    return y_proba, y_pred, y_pred.astype(float)


def _save_pr_curve(
    output_dir: str,
    model_type: str,
    split_name: str,
    y_true,
    y_score,
) -> str | None:
    return training_plots.save_pr_curve(output_dir, model_type, split_name, y_true, y_score)


def _save_confusion_matrix_plot(
    output_dir: str,
    model_type: str,
    split_name: str,
    y_true,
    y_pred,
) -> str | None:
    return training_plots.save_confusion_matrix_plot(output_dir, model_type, split_name, y_true, y_pred)


def _save_classification_split_plots(
    output_dir: str,
    model_type: str,
    split_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    return training_plots.save_classification_split_plots(output_dir, model_type, split_outputs)


def _as_bool(value: Any) -> bool:
    return training_config.as_bool(value)


def _parse_runtime_training_options(model_config: Dict[str, Any] | None):
    return training_config.parse_runtime_training_options(model_config)


def _validate_regression_metric_inputs(
    y_true: Any,
    y_pred: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    return training_metrics.validate_regression_metric_inputs(y_true, y_pred, context=context)


def _validate_classification_metric_inputs(
    y_true: Any,
    y_score: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    return training_metrics.validate_classification_metric_inputs(y_true, y_score, context=context)


def _classification_metrics_from_outputs(
    y_true: Any,
    y_score: Any,
    y_pred: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | None]]:
    return training_metrics.classification_metrics_from_outputs(
        y_true,
        y_score,
        y_pred,
        context=context,
        ensure_binary_labels=_ensure_binary_labels,
    )


def _safe_r2(y_true, y_pred) -> float | None:
    return training_metrics.safe_r2(y_true, y_pred)


def _safe_mae(y_true, y_pred) -> float | None:
    return training_metrics.safe_mae(y_true, y_pred)


def _save_split_metrics_artifacts(
    output_dir: str,
    model_type: str,
    split_metrics: Dict[str, Dict[str, float | None]],
) -> tuple[str | None, str | None]:
    return training_plots.save_split_metrics_artifacts(output_dir, model_type, split_metrics)


def _save_regression_parity_plots(
    output_dir: str,
    model_type: str,
    split_predictions: Dict[str, tuple[Any, Any]],
) -> Dict[str, str]:
    return training_plots.save_regression_parity_plots(output_dir, model_type, split_predictions)


def _ensure_binary_labels(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("Expected a single label column for classification.")
        series = series.iloc[:, 0]
    values = series.dropna().unique()
    if len(values) == 0:
        raise ValueError("Empty label series for classification.")
    if len(values) > 2:
        raise ValueError(f"Expected binary labels; got {len(values)} classes.")

    if set(values).issubset({0, 1}):
        return series.astype(int)

    def _coerce_numeric(val):
        if isinstance(val, (int, float, np.generic)) and val in {0, 1}:
            return int(val)
        if isinstance(val, str):
            token = val.strip().lower()
            if token in {"0", "1"}:
                return int(token)
            try:
                parsed = float(token)
            except ValueError:
                return None
            if parsed in {0.0, 1.0}:
                return int(parsed)
        return None

    def _normalize(v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    normalized = {_normalize(v) for v in values}
    if normalized == {"active", "inactive"}:
        mapping = {"active": 1, "inactive": 0}
    elif normalized == {"inactive", "active"}:
        mapping = {"active": 1, "inactive": 0}
    else:
        coerced = series.map(_coerce_numeric)
        if coerced.notna().all():
            return coerced.astype(int)
        raise ValueError("Classification labels must be 0/1 or active/inactive; add label.normalize.")

    return series.map(lambda v: mapping.get(_normalize(v)))


def _require_chemprop() -> None:
    """Raise an actionable error if chemprop is not installed."""
    try:
        import chemprop  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "chemprop is required for model.type=chemprop. "
            "Install it (and torch/lightning) e.g. `pip install chemprop`."
        ) from exc


def _resolve_chemprop_foundation_config(model_config: Dict[str, Any]) -> tuple[str, str | None, bool]:
    return training_config.resolve_chemprop_foundation_config(model_config)


def _resolve_chemprop_split_positions(
    curated_df: pd.DataFrame,
    split_indices: Dict[str, Any],
    row_index_col: str = _ROW_INDEX_COL,
    allow_legacy_positions: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Resolve split membership to positional indices for Chemprop.

    Preferred mode:
    - split indices are row IDs that match ``row_index_col`` values.

    Optional backward-compatible fallback:
    - if ``allow_legacy_positions`` is true and row-ID mapping fails but all
      indices are valid row positions, treat them as legacy positional indices.
    """

    def _to_int_list(split_name: str, values: Any) -> list[int]:
        out: list[int] = []
        for value in values or []:
            try:
                out.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"split.{split_name} contains a non-integer index value: {value!r}."
                ) from exc
        return out

    train_raw = _to_int_list("train", split_indices.get("train", []))
    val_raw = _to_int_list("val", split_indices.get("val", []))
    test_raw = _to_int_list("test", split_indices.get("test", []))
    if not test_raw and val_raw:
        test_raw, val_raw = val_raw, []

    n_rows = int(len(curated_df))
    if row_index_col in curated_df.columns:
        row_id_series = pd.to_numeric(curated_df[row_index_col], errors="coerce")
        if row_id_series.isna().any():
            raise ValueError(
                f"Curated data contains non-numeric {row_index_col!r} values; cannot map split row IDs."
            )
        row_ids = [int(v) for v in row_id_series.tolist()]
    else:
        row_ids = list(range(n_rows))

    if len(set(row_ids)) != len(row_ids):
        raise ValueError(
            f"Curated data contains duplicate {row_index_col!r} values; split row-ID mapping is ambiguous."
        )

    id_to_pos = {rid: pos for pos, rid in enumerate(row_ids)}

    def _map_row_ids(raw_ids: list[int]) -> tuple[list[int], list[int]]:
        mapped: list[int] = []
        missing: list[int] = []
        for rid in raw_ids:
            pos = id_to_pos.get(rid)
            if pos is None:
                missing.append(rid)
            else:
                mapped.append(int(pos))
        return mapped, missing

    train_pos, missing_train = _map_row_ids(train_raw)
    val_pos, missing_val = _map_row_ids(val_raw)
    test_pos, missing_test = _map_row_ids(test_raw)

    missing_total = len(missing_train) + len(missing_val) + len(missing_test)
    if missing_total > 0:
        all_raw = train_raw + val_raw + test_raw
        looks_like_legacy_positions = all(0 <= idx < n_rows for idx in all_raw)
        if looks_like_legacy_positions and allow_legacy_positions:
            logging.warning(
                "Chemprop split indices did not match %s values; treating them as legacy positional indices.",
                row_index_col,
            )
            train_pos, val_pos, test_pos = train_raw, val_raw, test_raw
        else:
            legacy_hint = ""
            if looks_like_legacy_positions:
                legacy_hint = (
                    " Indices look like legacy positional indices; regenerate splits with row IDs "
                    f"or set train.model.allow_legacy_split_positions=true to opt in."
                )
            raise ValueError(
                "Chemprop split indices do not match curated row IDs. "
                f"row_index_column={row_index_col!r} "
                f"missing(train/val/test)=({missing_train[:10]}, {missing_val[:10]}, {missing_test[:10]})."
                f"{legacy_hint}"
            )

    return train_pos, val_pos, test_pos, row_ids


def _resolve_chemprop_predictor_ctor(nn_module, task_type: str):
    if task_type == "classification":
        predictor_ctor = getattr(nn_module, "BinaryClassificationFFN", None)
        predictor_name = "BinaryClassificationFFN"
    else:
        predictor_ctor = getattr(nn_module, "RegressionFFN", None)
        predictor_name = "RegressionFFN"
    if predictor_ctor is None:
        raise ValueError(
            f"chemprop {task_type} requires nn.{predictor_name}, "
            "but it is unavailable in the installed chemprop version."
        )
    return predictor_ctor


def train_chemprop_model(
    curated_df: pd.DataFrame,
    target_column: str,
    split_indices: Dict[str, Any],
    output_dir: str,
    random_state: int = 42,
    task_type: str = "classification",
    model_config: Dict[str, Any] | None = None,
) -> Tuple[object, TrainResult]:
    return training_chemprop_models.train_chemprop_model(
        curated_df=curated_df,
        target_column=target_column,
        split_indices=split_indices,
        output_dir=output_dir,
        random_state=random_state,
        task_type=task_type,
        model_config=model_config,
        row_index_col=_ROW_INDEX_COL,
        ensure_dir=_ensure_dir,
        require_chemprop=_require_chemprop,
        resolve_chemprop_foundation_config=_resolve_chemprop_foundation_config,
        as_bool=_as_bool,
        resolve_chemprop_split_positions=_resolve_chemprop_split_positions,
        ensure_binary_labels=_ensure_binary_labels,
        resolve_chemprop_predictor_ctor=_resolve_chemprop_predictor_ctor,
        validate_classification_score_values=_validate_classification_score_values,
        sigmoid=_sigmoid,
        classification_metrics_from_outputs=_classification_metrics_from_outputs,
        validate_regression_metric_inputs=_validate_regression_metric_inputs,
        safe_r2=_safe_r2,
        safe_mae=_safe_mae,
        save_split_metrics_artifacts=_save_split_metrics_artifacts,
        save_classification_split_plots=_save_classification_split_plots,
        save_regression_parity_plots=_save_regression_parity_plots,
        save_params=training_persistence.save_params,
        save_metrics_json=lambda metrics, path: training_persistence.save_metrics_json(metrics, path, indent=2),
        train_result_cls=TrainResult,
    )

def _run_optuna(
    config: DLSearchConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_evals: int,
    random_state: int,
    patience: int,
    task_type: str = "regression",
) -> Tuple[object, dict]:
    return training_torch_models.run_optuna(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_evals=max_evals,
        random_state=random_state,
        patience=patience,
        task_type=task_type,
        seed_fn=_seed_dl_runtime,
        train_fn=_train_dl,
        predict_fn=_predict_dl,
    )

def _initialize_model(
    model_type: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    input_dim: int = None,
    n_jobs: int = -1,
    tuning_method: str = "fixed",
    model_params: Dict[str, Any] | None = None,
    task_type: str = "regression",
):
    return training_model_factory.initialize_model(
        model_type=model_type,
        random_state=random_state,
        cv_folds=cv_folds,
        search_iters=search_iters,
        input_dim=input_dim,
        n_jobs=n_jobs,
        tuning_method=tuning_method,
        model_params=model_params,
        task_type=task_type,
        dl_search_config_cls=DLSearchConfig,
    )

# DL training helper functions
def _get_device():
    return training_torch_models.get_device()


def _seed_dl_runtime(seed: int) -> None:
    training_torch_models.seed_dl_runtime(seed)


def _is_dl_model(model_type: str) -> bool:
    return training_model_factory.is_dl_model(model_type)
def _train_dl(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    random_state: int = 42,
    task_type: str = "regression",
    ) -> Dict[str, Any]:
    return training_torch_models.train_dl(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        random_state=random_state,
        task_type=task_type,
    )

def _predict_dl(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    return training_torch_models.predict_dl(model=model, X=X, batch_size=batch_size)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    random_state: int = 42,
    cv_folds: int = 5,
    search_iters: int = 100,
    use_hpo: bool = False,        
    hpo_trials: int = 30,          
    patience: int = 20,
    task_type: str = "regression",
    model_config: Dict[str, Any] | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Tuple[object, TrainResult]:
    _ensure_dir(output_dir)
    is_dl = _is_dl_model(model_type)
    runtime_options = _parse_runtime_training_options(model_config)
    model_config = runtime_options.model_config
    plot_split_performance = runtime_options.plot_split_performance
    debug_logging = runtime_options.debug_logging
    n_jobs = runtime_options.n_jobs
    tuning_method = runtime_options.tuning_method
    model_params = runtime_options.model_params

    logging.info(
        "Training start: model=%s task=%s tuning=%s X_train=%s X_test=%s",
        model_type,
        task_type,
        tuning_method,
        X_train.shape,
        X_test.shape,
    )
    X_train, X_test, X_val, feature_name_map_path = _maybe_sanitize_xgboost_feature_frames(
        model_type=model_type,
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        output_dir=output_dir,
    )

    if task_type == "classification":
        y_train = _ensure_binary_labels(y_train)
        y_test = _ensure_binary_labels(y_test)
        if y_val is not None:
            y_val = _ensure_binary_labels(y_val)

    if task_type == "regression" and model_type == "catboost_classifier":
        raise ValueError("Model type 'catboost_classifier' only supports classification tasks.")

    if task_type == "classification" and model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": random_state,
            "verbose": False,
        }
        params.update(model_config.get("params", {}))
        if not debug_logging:
            if any(key in params for key in ("verbose", "verbose_eval", "logging_level", "silent")):
                logging.info("Global debug logging is off; forcing quiet CatBoost training output.")
            params.pop("verbose_eval", None)
            params.pop("logging_level", None)
            params.pop("silent", None)
            params["verbose"] = False
        estimator = CatBoostClassifier(**params)
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = (X_val, y_val)
        fit_kwargs = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["use_best_model"] = True
        estimator.fit(X_train, y_train, **fit_kwargs)
        y_pred_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.cbm")
        estimator.save_model(model_path)
        best_params = estimator.get_params()
        y_test_arr, y_pred_proba, y_pred, metrics = _classification_metrics_from_outputs(
            y_test,
            y_pred_proba,
            y_pred,
            context=f"{model_type} test scoring",
        )
        if plot_split_performance:
            train_proba = estimator.predict_proba(X_train)[:, 1]
            train_pred = estimator.predict(X_train)
            y_train_arr, train_proba, train_pred, train_metrics = _classification_metrics_from_outputs(
                y_train,
                train_proba,
                train_pred,
                context=f"{model_type} train scoring",
            )
            split_metrics: Dict[str, Dict[str, float | None]] = {
                "train": train_metrics,
                "test": metrics.copy(),
            }
            split_outputs: Dict[str, Dict[str, Any]] = {
                "train": {
                    "y_true": y_train_arr,
                    "y_proba": train_proba,
                    "y_pred": train_pred,
                },
                "test": {
                    "y_true": y_test_arr,
                    "y_proba": y_pred_proba,
                    "y_pred": y_pred,
                },
            }
            if X_val is not None and y_val is not None and len(y_val) > 0:
                y_val_proba = estimator.predict_proba(X_val)[:, 1]
                y_val_pred = estimator.predict(X_val)
                y_val_arr, y_val_proba, y_val_pred, val_metrics = _classification_metrics_from_outputs(
                    y_val,
                    y_val_proba,
                    y_val_pred,
                    context=f"{model_type} val scoring",
                )
                split_metrics["val"] = val_metrics
                split_outputs["val"] = {
                    "y_true": y_val_arr,
                    "y_proba": y_val_proba,
                    "y_pred": y_val_pred,
                }
            split_metrics_path, split_plot_path = _save_split_metrics_artifacts(
                output_dir,
                model_type,
                split_metrics,
            )
            if split_metrics_path:
                metrics["split_metrics_path"] = split_metrics_path
            if split_plot_path:
                metrics["split_metrics_plot_path"] = split_plot_path
            metrics.update(_save_classification_split_plots(output_dir, model_type, split_outputs))

        roc_path = _save_roc_curve(output_dir, model_type, y_test, y_pred_proba)
        if roc_path:
            metrics["roc_curve_path"] = roc_path
        params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
        training_persistence.save_params(best_params, params_path)
        training_persistence.save_metrics_series(metrics, metrics_path)
        logging.info("Training complete (classification): metrics=%s", metrics)
        logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)
        return estimator, TrainResult(model_path, params_path, metrics_path)

    classification_model_types = {
        "catboost_classifier",
        "random_forest",
        "decision_tree",
        "xgboost",
        "svm",
        "ensemble",
    }

    if task_type == "classification" and not (model_type in classification_model_types or is_dl):
        raise ValueError(f"Unsupported classification model type: {model_type}")

    model = _initialize_model(
        model_type,
        random_state,
        cv_folds,
        search_iters,
        input_dim=X_train.shape[1] if is_dl else None,
        n_jobs=n_jobs,
        tuning_method=tuning_method,
        model_params=model_params,
        task_type=task_type,
    )
    if isinstance(model, DLSearchConfig):
        # ── DL Training ──
        if X_val is None or y_val is None or len(y_val) == 0:
            raise ValueError(
                "DL models require a validation split for early stopping/HPO. "
                "Ensure the pipeline includes the split node and set split.val_size > 0."
            )
        if use_hpo:
            logging.info(f"Running optuna for {model_type} ({hpo_trials} evals)")
            estimator, best_params = _run_optuna(
                model,
                X_train.values,
                y_train.values,
                X_val.values,
                y_val.values,
                hpo_trials,
                random_state,
                patience,
                task_type=task_type,
            )
        else:
            effective_params = {**model.default_params, **model_params}
            logging.info(f"Training DL model: {model_type} (fixed params)")
            _seed_dl_runtime(int(random_state))
            nn_model = model.model_class(effective_params)
            result = _train_dl(
                nn_model,
                X_train.values,
                y_train.values,
                X_val.values,
                y_val.values,
                epochs=effective_params["epochs"],
                batch_size=effective_params["batch_size"],
                learning_rate=effective_params["learning_rate"],
                patience=patience,
                random_state=random_state,
                task_type=task_type,
            )
            estimator = result["model"]
            # Persist the full effective DL config so downstream reload/explain
            # rebuilds the same architecture instead of falling back to defaults.
            best_params = {**effective_params, **result["best_params"]}
        
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        y_pred = _predict_dl(estimator, X_test.values)
    else:

        logging.info(f"Training ML model: {model_type}")
        model.fit(X_train, y_train)

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")
        training_persistence.save_model_pickle(estimator, model_path)
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}

    if task_type == "classification":
        y_pred_proba, y_pred_label, y_pred_score = _predict_classification_outputs(
            estimator=estimator,
            model_type=model_type,
            X=X_test,
        )
        y_true, y_pred_proba, y_pred_label, metrics = _classification_metrics_from_outputs(
            y_test,
            y_pred_proba,
            y_pred_label,
            context=f"{model_type} test scoring",
        )
        roc_path = _save_roc_curve(output_dir, model_type, y_true, y_pred_proba)
        if roc_path:
            metrics["roc_curve_path"] = roc_path

        pred_path = os.path.join(output_dir, f"{model_type}_predictions.csv")
        pd.DataFrame({
            "y_true": y_true,
            "y_score": np.asarray(y_pred_score).reshape(-1),
            "y_proba": y_pred_proba,
            "y_pred": y_pred_label,
        }).to_csv(pred_path, index=False)
    else:
        y_true, y_pred = _validate_regression_metric_inputs(
            y_test,
            y_pred,
            context=f"{model_type} test scoring",
        )
        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    if feature_name_map_path:
        metrics["feature_name_map_path"] = feature_name_map_path

    if plot_split_performance:
        split_metrics: Dict[str, Dict[str, float | None]] = {}

        if task_type == "classification":
            split_outputs: Dict[str, Dict[str, Any]] = {}
            # --- TRAIN predictions ---
            tr_proba, tr_pred, _ = _predict_classification_outputs(
                estimator=estimator,
                model_type=model_type,
                X=X_train,
            )
            y_tr, tr_proba, tr_pred, train_metrics = _classification_metrics_from_outputs(
                y_train,
                tr_proba,
                tr_pred,
                context=f"{model_type} train scoring",
            )
            split_metrics["train"] = train_metrics
            split_outputs["train"] = {
                "y_true": y_tr,
                "y_proba": tr_proba,
                "y_pred": tr_pred,
            }

            # --- TEST predictions ---
            te_proba, te_pred, _ = _predict_classification_outputs(
                estimator=estimator,
                model_type=model_type,
                X=X_test,
            )
            y_te, te_proba, te_pred, test_metrics = _classification_metrics_from_outputs(
                y_test,
                te_proba,
                te_pred,
                context=f"{model_type} test scoring",
            )
            split_metrics["test"] = test_metrics
            split_outputs["test"] = {
                "y_true": y_te,
                "y_proba": te_proba,
                "y_pred": te_pred,
            }

            # --- VAL predictions  ---
            if X_val is not None and y_val is not None and len(y_val) > 0:
                va_proba, va_pred, _ = _predict_classification_outputs(
                    estimator=estimator,
                    model_type=model_type,
                    X=X_val,
                )
                y_va, va_proba, va_pred, val_metrics = _classification_metrics_from_outputs(
                    y_val,
                    va_proba,
                    va_pred,
                    context=f"{model_type} val scoring",
                )
                split_metrics["val"] = val_metrics
                split_outputs["val"] = {
                    "y_true": y_va,
                    "y_proba": va_proba,
                    "y_pred": va_pred,
                }

        else:
            # --- regression ---
            if is_dl:
                y_train_pred = _predict_dl(estimator, X_train.values)
                y_test_pred = _predict_dl(estimator, X_test.values)
                y_val_pred = _predict_dl(estimator, X_val.values) if X_val is not None else None
            else:
                y_train_pred = estimator.predict(X_train)
                y_test_pred = estimator.predict(X_test)
                y_val_pred = estimator.predict(X_val) if X_val is not None else None

            split_metrics = {
                "train": {"r2": _safe_r2(y_train, y_train_pred), "mae": _safe_mae(y_train, y_train_pred)},
                "test": {"r2": _safe_r2(y_test, y_test_pred), "mae": _safe_mae(y_test, y_test_pred)},
            }
            if X_val is not None and y_val is not None and len(y_val) > 0 and y_val_pred is not None:
                split_metrics["val"] = {"r2": _safe_r2(y_val, y_val_pred), "mae": _safe_mae(y_val, y_val_pred)}

        split_metrics_path, split_plot_path = _save_split_metrics_artifacts(output_dir, model_type, split_metrics)
        if split_metrics_path:
            metrics["split_metrics_path"] = split_metrics_path
        if split_plot_path:
            metrics["split_metrics_plot_path"] = split_plot_path

        if task_type == "classification":
            metrics.update(_save_classification_split_plots(output_dir, model_type, split_outputs))
        else:
            parity_paths = _save_regression_parity_plots(
                output_dir,
                model_type,
                {
                    "train": (y_train, y_train_pred),
                    "test": (y_test, y_test_pred),
                    "val": (y_val, y_val_pred),
                },
            )
            for split_name, path in parity_paths.items():
                metrics[f"parity_plot_{split_name}_path"] = path

    params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    if is_dl:
        training_persistence.save_torch_state_dict(estimator, model_path)
    training_persistence.save_params(best_params, params_path)
    training_persistence.save_metrics_series(metrics, metrics_path)
    logging.info("Training complete (%s): metrics=%s", task_type, metrics)
    logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)

    return estimator, TrainResult(model_path, params_path, metrics_path)

def load_model(model_path: str, model_type: str, input_dim: int = None) -> object:
    is_dl = _is_dl_model(model_type)
    
    if is_dl:
        import torch
        if input_dim is None:
            raise ValueError("input_dim required to load DL models")
        
        params_path = model_path.replace("_best_model.pth", "_best_params.pkl")
        if os.path.exists(params_path):
            saved_params = training_persistence.load_pickle(params_path)
        else:
            saved_params = {}

        config = _initialize_model(model_type, random_state=42, cv_folds=5, 
                                   search_iters=100, input_dim=input_dim)
        
        model_params = {**config.default_params, **saved_params}
        
        model = config.model_class(model_params)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    if model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    return training_persistence.load_pickle(model_path)

def run_explainability(
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    background_samples: int = 100,
    task_type: str = "regression",
) -> None:
    training_explainability.run_explainability(
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        output_dir=output_dir,
        background_samples=background_samples,
        task_type=task_type,
        ensure_dir=_ensure_dir,
        ensure_binary_labels=_ensure_binary_labels,
        resolve_n_jobs=_resolve_n_jobs,
        maybe_apply_xgboost_feature_map_for_explain=_maybe_apply_xgboost_feature_map_for_explain,
        get_device=_get_device,
        predict_dl=_predict_dl,
        validate_classification_score_values=_validate_classification_score_values,
        sigmoid=_sigmoid,
        safe_auc=_safe_auc,
        validate_regression_metric_inputs=_validate_regression_metric_inputs,
    )
