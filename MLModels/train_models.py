import json
import os
import logging
import shutil
import inspect
import random
import re
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MLModels.training import config as training_config
from MLModels.training import metrics as training_metrics
from MLModels.training import persistence as training_persistence
from MLModels.training import plots as training_plots
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
)
from sklearn.inspection import permutation_importance

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
    """Train a Chemprop D-MPNN model in-process (no CLI/subprocess).

    This path is intentionally SMILES-native. It supports both classification and
    regression tasks and expects the curated dataset to include a `canonical_smiles`
    column (or an override via model_config.smiles_column).

    Artifacts written (consistent with other trainers):
    - <output_dir>/chemprop_best_model.ckpt
    - <output_dir>/chemprop_best_params.pkl
    - <output_dir>/chemprop_metrics.json
    - <output_dir>/chemprop_predictions.csv
    """
    _ensure_dir(output_dir)
    _require_chemprop()
    model_config = model_config or {}
    foundation_mode, foundation_checkpoint, freeze_encoder = _resolve_chemprop_foundation_config(model_config)

    smiles_column = model_config.get("smiles_column", "canonical_smiles")
    if smiles_column not in curated_df.columns:
        raise ValueError(
            f"chemprop requires smiles_column={smiles_column!r} in curated_df. "
            "Ensure curate emits canonical_smiles or override model.smiles_column."
        )
    if target_column not in curated_df.columns:
        raise ValueError(f"Target column {target_column!r} not found in curated_df.")

    df = curated_df.reset_index(drop=True).copy()
    allow_legacy_split_positions = _as_bool(model_config.get("allow_legacy_split_positions", False))
    tr_idx, va_idx, te_idx, row_ids = _resolve_chemprop_split_positions(
        df,
        split_indices,
        row_index_col=_ROW_INDEX_COL,
        allow_legacy_positions=allow_legacy_split_positions,
    )
    if not tr_idx or not te_idx:
        raise ValueError("chemprop training requires non-empty train and test splits.")
    if not va_idx:
        raise ValueError(
            "chemprop training requires an explicit validation split from the split node. "
            "Set split.val_size > 0."
        )
    if task_type == "classification":
        y_all = _ensure_binary_labels(df[target_column]).astype(int)
    else:
        y_all = pd.to_numeric(df[target_column], errors="coerce")
        if y_all.isna().any():
            raise ValueError(
                "chemprop regression requires numeric target values; "
                f"found non-numeric entries in target_column={target_column!r}."
            )
    # Keep stable row identifiers for prediction joins/debugging.
    df["_row_id"] = row_ids
    df["_label"] = y_all.reset_index(drop=True)

    # Hyperparameters.
    params = dict(model_config.get("params", {}))
    batch_size = int(params.get("batch_size", 64))
    max_epochs = int(params.get("max_epochs", 30))
    max_lr = float(params.get("max_lr", params.get("lr", 1e-3)))
    init_lr = float(params.get("init_lr", max_lr / 10.0))
    final_lr = float(params.get("final_lr", max_lr / 10.0))
    ff_hidden = int(params.get("ffn_hidden_dim", 300))
    mp_hidden = int(params.get("mp_hidden_dim", 300))
    depth = int(params.get("mp_depth", 3))
    plot_split_performance = _as_bool(model_config.get("plot_split_performance", False))

    # Under pytest we aggressively shrink runtime unless explicitly overridden.
    if os.environ.get("PYTEST_CURRENT_TEST") and "max_epochs" not in params:
        max_epochs = 2

    logging.info(
        "Training start (chemprop): task=%s N=%s train=%s val=%s test=%s",
        task_type,
        len(df),
        len(tr_idx),
        len(va_idx),
        len(te_idx),
    )

    # Chemprop python API (v2). Keep imports local so base installs don't require chemprop deps.
    import numpy as _np
    import torch
    from lightning import pytorch as pl
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint

    from chemprop import data, featurizers, models, nn

    pl.seed_everything(int(random_state), workers=True)

    def _to_datapoints(rows: pd.DataFrame) -> list:
        points = []
        for smi, y in zip(rows[smiles_column].astype(str).tolist(), rows["_label"].tolist()):
            points.append(data.MoleculeDatapoint.from_smi(smi, y=_np.array([float(y)], dtype=_np.float32)))
        return points

    all_points = _to_datapoints(df)
    # Avoid chemprop split API differences by slicing datapoints directly with pipeline indices.
    train_points = [all_points[i] for i in tr_idx]
    val_points = [all_points[i] for i in va_idx]
    test_points = [all_points[i] for i in te_idx]

    mol_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_data = data.MoleculeDataset(train_points, featurizer=mol_featurizer)
    val_data = data.MoleculeDataset(val_points, featurizer=mol_featurizer)
    test_data = data.MoleculeDataset(test_points, featurizer=mol_featurizer)
    if hasattr(data, "build_dataloader"):
        train_loader = data.build_dataloader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        train_eval_loader = data.build_dataloader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_loader = data.build_dataloader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = data.build_dataloader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_loader = data.MolGraphDataLoader(
            train_data, mol_featurizer, batch_size=batch_size, shuffle=True, num_workers=0
        )
        train_eval_loader = data.MolGraphDataLoader(
            train_data, mol_featurizer, batch_size=batch_size, shuffle=False, num_workers=0
        )
        val_loader = data.MolGraphDataLoader(
            val_data, mol_featurizer, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = data.MolGraphDataLoader(
            test_data, mol_featurizer, batch_size=batch_size, shuffle=False, num_workers=0
        )

    import inspect

    foundation_hparams: Dict[str, Any] | None = None
    if foundation_mode == "chemeleon":
        assert foundation_checkpoint is not None  # validated by _resolve_chemprop_foundation_config
        payload = torch.load(foundation_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid CheMeleon checkpoint payload in {foundation_checkpoint}: expected dict."
            )
        if "hyper_parameters" not in payload or "state_dict" not in payload:
            raise ValueError(
                f"Invalid CheMeleon checkpoint payload in {foundation_checkpoint}: expected keys 'hyper_parameters' and 'state_dict'."
            )
        hyper_parameters = payload["hyper_parameters"]
        state_dict = payload["state_dict"]
        if not isinstance(hyper_parameters, dict):
            raise ValueError(
                f"Invalid CheMeleon checkpoint payload in {foundation_checkpoint}: 'hyper_parameters' must be a dict."
            )
        foundation_hparams = hyper_parameters
        mp = nn.BondMessagePassing(**hyper_parameters)
        mp.load_state_dict(state_dict)
        if freeze_encoder:
            for param in mp.parameters():
                param.requires_grad = False
            # Keep frozen encoder layers in eval mode to avoid stochastic behavior.
            mp.eval()
        logging.info(
            "Chemprop foundation init: mode=%s checkpoint=%s freeze_encoder=%s",
            foundation_mode,
            foundation_checkpoint,
            freeze_encoder,
        )
    else:
        mp = nn.BondMessagePassing(d_h=mp_hidden, depth=depth)

    mp_output_dim = int(getattr(mp, "output_dim", mp_hidden))
    configured_or_ckpt_depth = (
        foundation_hparams.get("depth")
        if isinstance(foundation_hparams, dict) and "depth" in foundation_hparams
        else depth
    )
    mp_depth_effective = int(getattr(mp, "depth", configured_or_ckpt_depth))
    agg = nn.MeanAggregation()

    predictor_ctor = _resolve_chemprop_predictor_ctor(nn, task_type)

    # Chemprop FFN constructors changed across versions:
    # - older: *(..., d_mp=..., d_h=...)
    # - newer: *(..., input_dim=..., hidden_dim=...)
    ffn_sig = inspect.signature(predictor_ctor.__init__)
    ffn_kwargs: Dict[str, Any] = {"n_tasks": 1}
    if "d_mp" in ffn_sig.parameters:
        ffn_kwargs["d_mp"] = mp_output_dim
    if "input_dim" in ffn_sig.parameters:
        ffn_kwargs["input_dim"] = mp_output_dim
    if "d_h" in ffn_sig.parameters:
        ffn_kwargs["d_h"] = ff_hidden
    if "hidden_dim" in ffn_sig.parameters:
        ffn_kwargs["hidden_dim"] = ff_hidden
    ffn = predictor_ctor(**ffn_kwargs)
    mpnn = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=False,
        metrics=None,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    checkpointing = ModelCheckpoint(
        dirpath=output_dir,
        filename="chemprop-best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [checkpointing]
    if foundation_mode == "chemeleon" and freeze_encoder:
        from lightning.pytorch.callbacks import Callback

        class _FrozenEncoderEvalCallback(Callback):
            """Keep frozen encoder modules in eval mode across Lightning train/eval toggles."""

            def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
                for attr_name in ("message_passing", "mp", "encoder"):
                    module = getattr(pl_module, attr_name, None)
                    if isinstance(module, torch.nn.Module):
                        module.eval()

        callbacks.append(_FrozenEncoderEvalCallback())
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        deterministic=True,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
    )
    trainer.fit(mpnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Predict outputs from the best checkpoint.
    def _extract_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)) and obj:
            return _extract_tensor(obj[0])
        if isinstance(obj, dict) and obj:
            # Best-effort: common key, else first value.
            if "preds" in obj:
                return _extract_tensor(obj["preds"])
            return _extract_tensor(next(iter(obj.values())))
        raise TypeError(f"Unsupported prediction batch type: {type(obj)!r}")

    def _predict_batches(loader):
        try:
            return trainer.predict(dataloaders=loader, ckpt_path="best", weights_only=False)
        except TypeError:
            # Backward compatibility with older Lightning signatures.
            return trainer.predict(dataloaders=loader, ckpt_path="best")

    def _predict_values(loader, *, context: str) -> np.ndarray:
        pred_batches = _predict_batches(loader)
        if isinstance(pred_batches, list):
            preds_t = torch.cat(
                [_extract_tensor(p).detach().cpu().reshape(-1, 1) for p in pred_batches],
                dim=0,
            )
            preds = preds_t.numpy().reshape(-1)
        else:
            preds = _extract_tensor(pred_batches).detach().cpu().numpy().reshape(-1)
        preds = preds.astype(float)
        if task_type == "classification":
            preds = _validate_classification_score_values(
                preds,
                context=f"{context}: raw classification scores",
            )
            if preds.size > 0 and (np.nanmin(preds) < 0.0 or np.nanmax(preds) > 1.0):
                # Some configs may emit logits; convert to probabilities for metrics.
                preds = _sigmoid(preds)
            preds = _validate_classification_score_values(
                preds,
                context=f"{context}: classification probabilities",
            )
        return preds

    preds = _predict_values(test_loader, context="chemprop test scoring")

    metrics = {
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "init_lr": float(init_lr),
        "max_lr": float(max_lr),
        "final_lr": float(final_lr),
        "mp_hidden_dim": int(mp_output_dim),
        "mp_depth": int(mp_depth_effective),
        "ffn_hidden_dim": int(ff_hidden),
        "foundation": foundation_mode,
        "foundation_checkpoint": foundation_checkpoint,
        "freeze_encoder": bool(freeze_encoder),
    }
    if task_type == "classification":
        y_true, preds, y_pred, classification_metrics = _classification_metrics_from_outputs(
            df.loc[te_idx, "_label"].to_numpy(dtype=int),
            preds,
            (preds >= 0.5).astype(int),
            context="chemprop test scoring",
        )
        metrics.update(classification_metrics)
    else:
        y_true, y_pred = _validate_regression_metric_inputs(
            df.loc[te_idx, "_label"].to_numpy(dtype=float),
            preds.astype(float),
            context="chemprop test scoring",
        )
        metrics.update(
            {
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        )
    if plot_split_performance:
        train_preds = _predict_values(train_eval_loader, context="chemprop train scoring")
        val_preds = _predict_values(val_loader, context="chemprop val scoring")
        if task_type == "classification":
            train_y, train_preds, train_pred_labels, train_metrics = _classification_metrics_from_outputs(
                df.loc[tr_idx, "_label"].to_numpy(dtype=int),
                train_preds,
                (train_preds >= 0.5).astype(int),
                context="chemprop train scoring",
            )
            val_y, val_preds, val_pred_labels, val_metrics = _classification_metrics_from_outputs(
                df.loc[va_idx, "_label"].to_numpy(dtype=int),
                val_preds,
                (val_preds >= 0.5).astype(int),
                context="chemprop val scoring",
            )
            split_metrics: Dict[str, Dict[str, float | None]] = {
                "train": train_metrics,
                "val": val_metrics,
                "test": {
                    "auc": metrics["auc"],
                    "auprc": metrics["auprc"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                },
            }
        else:
            train_y = df.loc[tr_idx, "_label"].to_numpy(dtype=float)
            val_y = df.loc[va_idx, "_label"].to_numpy(dtype=float)
            split_metrics = {
                "train": {"r2": _safe_r2(train_y, train_preds), "mae": _safe_mae(train_y, train_preds)},
                "val": {"r2": _safe_r2(val_y, val_preds), "mae": _safe_mae(val_y, val_preds)},
                "test": {"r2": metrics["r2"], "mae": metrics["mae"]},
            }
        split_metrics_path, split_plot_path = _save_split_metrics_artifacts(
            output_dir,
            "chemprop",
            split_metrics,
        )
        if split_metrics_path:
            metrics["split_metrics_path"] = split_metrics_path
        if split_plot_path:
            metrics["split_metrics_plot_path"] = split_plot_path
        if task_type == "classification":
            metrics.update(
                _save_classification_split_plots(
                    output_dir,
                    "chemprop",
                    {
                        "train": {
                            "y_true": train_y,
                            "y_proba": train_preds,
                            "y_pred": train_pred_labels,
                        },
                        "val": {
                            "y_true": val_y,
                            "y_proba": val_preds,
                            "y_pred": val_pred_labels,
                        },
                        "test": {
                            "y_true": y_true,
                            "y_proba": preds,
                            "y_pred": y_pred,
                        },
                    },
                )
            )
        else:
            parity_paths = _save_regression_parity_plots(
                output_dir,
                "chemprop",
                {
                    "train": (train_y, train_preds),
                    "val": (val_y, val_preds),
                    "test": (y_true, y_pred),
                },
            )
            for split_name, path in parity_paths.items():
                metrics[f"parity_plot_{split_name}_path"] = path

    # Save artifacts.
    model_path = os.path.join(output_dir, "chemprop_best_model.ckpt")
    best_model_path = checkpointing.best_model_path
    if best_model_path and os.path.isfile(best_model_path):
        shutil.copyfile(best_model_path, model_path)
    else:
        trainer.save_checkpoint(model_path)
    params_path = os.path.join(output_dir, "chemprop_best_params.pkl")
    metrics_path = os.path.join(output_dir, "chemprop_metrics.json")
    best_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "init_lr": init_lr,
        "max_lr": max_lr,
        "final_lr": final_lr,
        "mp_hidden_dim": mp_output_dim,
        "mp_depth": mp_depth_effective,
        "ffn_hidden_dim": ff_hidden,
        "foundation": foundation_mode,
        "foundation_checkpoint": foundation_checkpoint,
        "freeze_encoder": bool(freeze_encoder),
    }
    training_persistence.save_params(best_params, params_path)
    training_persistence.save_metrics_json(metrics, metrics_path, indent=2)

    pred_path = os.path.join(output_dir, "chemprop_predictions.csv")
    out_pred = df.loc[te_idx, ["_row_id", smiles_column, target_column]].copy()
    out_pred.rename(columns={target_column: "y_true", smiles_column: "smiles"}, inplace=True)
    out_pred["y_pred"] = y_pred
    if task_type == "classification":
        out_pred["y_proba"] = preds
    out_pred.to_csv(pred_path, index=False)

    report_keys = ["auc", "auprc", "accuracy", "f1"] if task_type == "classification" else ["r2", "mae"]
    logging.info(
        "Training complete (chemprop): task=%s metrics=%s",
        task_type,
        {k: metrics[k] for k in report_keys},
    )
    logging.info("Artifacts: model=%s metrics=%s params=%s preds=%s", model_path, metrics_path, params_path, pred_path)

    return mpnn, TrainResult(model_path, params_path, metrics_path)

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
    try:
        import optuna
    except Exception as exc:
        raise ImportError("optuna is required for DL hyperparameter search.") from exc
    
    if X_val is None or y_val is None or len(y_val) == 0:
        raise ValueError("DL hyperparameter search requires an explicit validation split (X_val/y_val).")
    
    best_model = None
    best_score = float("-inf")
    pruned_reasons: list[str] = []
    
    def objective(trial: optuna.Trial) -> float:
        nonlocal best_model, best_score
        
        # Sample hyperparameters from search space
        params = {}
        for name, spec in config.search_space.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
        
        trial_seed = int(random_state) + int(trial.number) + 1
        _seed_dl_runtime(trial_seed)
        model = config.model_class(params)

        # Train
        result = _train_dl(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 32)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            patience=patience,
            random_state=trial_seed,
            task_type=task_type
        )
        
        # Evaluate on validation set
        trained_model = result["model"]

        # Predict
        y_pred = _predict_dl(trained_model, X_val)

        if task_type == "classification":
            try:
                y_true, y_score = _validate_classification_metric_inputs(
                    y_val,
                    np.asarray(y_pred).reshape(-1),
                    context="optuna validation scoring",
                )
            except ValueError as exc:
                pruned_reasons.append(str(exc))
                raise optuna.exceptions.TrialPruned(str(exc)) from exc
            y_proba = 1.0 / (1.0 + np.exp(-y_score))
            score = _safe_auc(y_true, y_proba)
            if score is None:
                message = "AUC undefined (single-class val set)"
                pruned_reasons.append(message)
                raise optuna.exceptions.TrialPruned(message)
        else:
            try:
                y_true, y_hat = _validate_regression_metric_inputs(
                    y_val,
                    y_pred,
                    context="optuna validation scoring",
                )
            except ValueError as exc:
                pruned_reasons.append(str(exc))
                raise optuna.exceptions.TrialPruned(str(exc)) from exc
            score = float(r2_score(y_true, y_hat))

        # Track best
        if score > best_score:
            best_score = score
            best_model = trained_model
            logging.info(f"New best score={score:.4f} with params: {params}")
        
        import gc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        del model, result, y_pred
        gc.collect()
        
        return score # Optuna maximizes by default when direction="maximize"
    
    # Create and run study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_evals, show_progress_bar=True)

    if best_model is None:
        detail = pruned_reasons[-1] if pruned_reasons else "no completed trials"
        raise ValueError(
            "DL hyperparameter search completed no valid trials; "
            f"last prune reason: {detail}"
        )

    best_params = study.best_params
    logging.info(f"Optuna complete. Best score={best_score:.4f}, params={best_params}")
    
    return best_model, best_params

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
    tuning_method = str(tuning_method or "fixed").strip().lower()
    if tuning_method not in {"train_cv", "fixed"}:
        raise ValueError(f"Unsupported model.tuning.method={tuning_method!r}; expected 'train_cv' or 'fixed'.")
    model_params = model_params or {}

    is_classification = str(task_type or "").strip().lower() == "classification"

    if model_type == "random_forest":
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }
        if is_classification:
            if tuning_method == "fixed":
                params = {"random_state": random_state, **model_params}
                params.setdefault("n_jobs", n_jobs)
                return RandomForestClassifier(**params)
            base_rf_cls = RandomForestClassifier(random_state=random_state)
            return RandomizedSearchCV(
                estimator=base_rf_cls,
                param_distributions=param_dist,
                n_iter=search_iters,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        if tuning_method == "fixed":
            params = {"random_state": random_state, **model_params}
            params.setdefault("n_jobs", n_jobs)
            return RandomForestRegressor(**params)
        base_rf = RandomForestRegressor(random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "svm":
        if is_classification:
            param_grid_svc = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.1, 0.01],
            }
            if tuning_method == "fixed":
                params = {"probability": True, **model_params}
                return SVC(**params)
            return GridSearchCV(
                estimator=SVC(kernel="rbf", probability=True),
                param_grid=param_grid_svc,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
            )
        param_grid_svm = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "epsilon": [0.1, 0.2, 0.5],
        }
        if tuning_method == "fixed":
            return SVR(**model_params)
        return GridSearchCV(
            estimator=SVR(kernel="rbf"),
            param_grid=param_grid_svm,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
        )
    if model_type == "decision_tree":
        param_dist_dt = {
            "max_depth": [int(x) for x in np.linspace(5, 50, num=10)] + [None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
        if is_classification:
            if tuning_method == "fixed":
                params = {"random_state": random_state, **model_params}
                return DecisionTreeClassifier(**params)
            base_dt_cls = DecisionTreeClassifier(random_state=random_state)
            return RandomizedSearchCV(
                estimator=base_dt_cls,
                param_distributions=param_dist_dt,
                n_iter=search_iters,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        if tuning_method == "fixed":
            params = {"random_state": random_state, **model_params}
            return DecisionTreeRegressor(**params)
        base_dt = DecisionTreeRegressor(random_state=random_state)
        return RandomizedSearchCV(
            estimator=base_dt,
            param_distributions=param_dist_dt,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "xgboost":
        param_dist_xgb = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.1, 1, 10, 100],
        }
        if is_classification:
            if tuning_method == "fixed":
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": random_state,
                    "n_jobs": n_jobs,
                    **model_params,
                }
                return XGBClassifier(**params)
            base_xgb_cls = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=n_jobs,
            )
            return RandomizedSearchCV(
                estimator=base_xgb_cls,
                param_distributions=param_dist_xgb,
                n_iter=search_iters,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=n_jobs,
                random_state=random_state,
            )
        if tuning_method == "fixed":
            params = {
                "objective": "reg:squarederror",
                "random_state": random_state,
                "n_jobs": n_jobs,
                **model_params,
            }
            return XGBRegressor(**params)
        base_xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist_xgb,
            n_iter=search_iters,
            cv=cv_folds,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_type == "ensemble":
        ensemble_cfg = model_params if isinstance(model_params, dict) else {}
        rf_override = (
            ensemble_cfg.get("rf_params")
            if isinstance(ensemble_cfg.get("rf_params"), dict)
            else {}
        )
        xgb_override = (
            ensemble_cfg.get("xgb_params")
            if isinstance(ensemble_cfg.get("xgb_params"), dict)
            else {}
        )
        voting = str(ensemble_cfg.get("voting", "soft")).strip().lower() or "soft"

        if is_classification:
            rf_params = {
                "n_estimators": 200,
                "max_depth": 30,
                "max_features": "sqrt",
                "bootstrap": False,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "random_state": random_state,
                "n_jobs": n_jobs,
                **rf_override,
            }
            xgb_params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": n_jobs,
                **xgb_override,
            }
            rf = RandomForestClassifier(**rf_params)
            xgb = XGBClassifier(**xgb_params)
            return VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)],
                voting=voting,
                n_jobs=n_jobs,
            )

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            max_features="sqrt",
            bootstrap=False,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=random_state,
            **rf_override,
        )
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            random_state=random_state,
            n_jobs=n_jobs,
            **xgb_override,
        )
        return VotingRegressor([("rf", rf), ("xgb", xgb)], n_jobs=n_jobs)
    
    if model_type == "dl_simple":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        try:
            from DLModels.simplenn import SimpleNN
        except Exception:
            # Backward compatibility for repos that still expose the legacy module/class.
            from DLModels.simpleregressionnn import SimpleRegressionNN as SimpleNN

        simple_params = inspect.signature(SimpleNN).parameters

        def _make_simple_model(params: Dict[str, Any]):
            kwargs = {
                "input_dim": input_dim,
                "hidden_dim": params.get("hidden_dim", 256),
            }
            if "use_tropical" in simple_params:
                kwargs["use_tropical"] = params.get("use_tropical", False)
            return SimpleNN(**kwargs)

        return DLSearchConfig(
            model_class=_make_simple_model,
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                **(
                    {"use_tropical": {"type": "categorical", "choices": [True, False]}}
                    if "use_tropical" in simple_params
                    else {}
                ),
            },
            default_params={
                "hidden_dim": 256,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )
    if model_type == "dl_deep":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.deepregressionnn import DeepRegressionNN
        return DLSearchConfig(
            model_class=lambda params: DeepRegressionNN(
                input_dim=input_dim,
                hidden_dims=[params.get("hidden_dim", 128)] * params.get("num_layers", 3),
                dropout_rate=params.get("dropout_rate", 0.2),
                use_tropical=params.get("use_tropical", False),
            ),
            search_space={
                "num_layers": {"type": "categorical", "choices": [2, 3, 4, 5]},
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "dropout_rate": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                "use_tropical": {"type": "categorical", "choices": [True, False]},
            },
            default_params={
                "num_layers": 3,
                "hidden_dim": 128,
                "dropout_rate": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    if model_type == "dl_gru":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.gruregressor import GRURegressor
        return DLSearchConfig(
            model_class=lambda params: GRURegressor(
                seq_len=input_dim,
                input_size=params.get("input_size", 1),
                hidden_size=params.get("hidden_size", 128),
                num_layers=params.get("num_layers", 2),
                bidirectional=params.get("bidirectional", True),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
                "num_layers": {"type": "categorical", "choices": [1, 2]},
                "bidirectional": {"type": "categorical", "choices": [True, False]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 8,
                "epochs": 200,
            },
        )
    if model_type == "dl_resmlp":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.resmlp import ResMLP
        return DLSearchConfig(
            model_class=lambda params: ResMLP(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_dim", 512),
                n_blocks=params.get("n_blocks", 4),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [128, 256, 512, 1024]},
                "n_blocks": {"type": "categorical", "choices": [2, 3, 4, 6, 8]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.6, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "hidden_dim": 512,
                "n_blocks": 4,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )
    
    if model_type == "dl_tabtransformer":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.tabtransformer import TabTransformer
        return DLSearchConfig(
            model_class=lambda params: TabTransformer(
                input_dim=input_dim,
                embed_dim=params.get("embed_dim", 128),
                n_heads=params.get("n_heads", 4),
                n_layers=params.get("n_layers", 2),
                dropout=params.get("dropout", 0.2),
            ),
            search_space={
                "embed_dim": {"type": "categorical", "choices": [64, 128, 256]},
                "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
                "n_layers": {"type": "categorical", "choices": [2, 3, 4]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.4, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "embed_dim": 128,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 8,
                "epochs": 200,
            },
        )
    
    if model_type == "dl_aereg":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.aeregressor import Autoencoder, AERegressor
        return DLSearchConfig(
            model_class=lambda params: AERegressor(
                pretrained_encoder=Autoencoder(
                    input_dim=input_dim,
                    bottleneck=params.get("bottleneck", 64),
                ).encoder,
                bottleneck=params.get("bottleneck", 64),
                dropout=params.get("dropout", 0.1),
            ),
            search_space={
                "bottleneck": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "bottleneck": 64,
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 200,
            },
        )

    raise ValueError(f"Unsupported model type: {model_type}")

# DL training helper functions
def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_dl_runtime(seed: int) -> None:
    """Seed Python/NumPy/PyTorch for deterministic DL training."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def _is_dl_model(model_type: str) -> bool:
    return model_type.startswith("dl_")
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
    """Train a PyTorch model. Returns dict with model and best_params."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    device = _get_device()
    model = model.to(device)
    
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    # y_t = torch.tensor(np.asarray(y_train).reshape(-1, 1), dtype=torch.float32, device=device)

    if X_val is None or y_val is None:
        raise ValueError("DL training requires X_val/y_val for early stopping.")
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    # y_val_t = torch.tensor(np.asarray(y_val).reshape(-1, 1), dtype=torch.float32, device=device)

    if task_type == "classification":
        y_t = torch.tensor(np.asarray(y_train).reshape(-1), dtype=torch.float32, device=device)
        y_val_t = torch.tensor(np.asarray(y_val).reshape(-1), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
    else:
        y_t = torch.tensor(np.asarray(y_train).reshape(-1, 1), dtype=torch.float32, device=device)
        y_val_t = torch.tensor(np.asarray(y_val).reshape(-1, 1), dtype=torch.float32, device=device)
        criterion = nn.MSELoss()
    
    dl_generator = torch.Generator()
    dl_generator.manual_seed(int(random_state))
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        generator=dl_generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()
    
    best_loss, best_state, wait = float('inf'), None, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            # loss = criterion(model(bx).view(-1, 1), by)
            out = model(bx)
            if task_type == "classification":
                loss = criterion(out.view(-1), by.view(-1))
            else:
                loss = criterion(out.view(-1, 1), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            # val_loss = criterion(model(X_val_t).view(-1, 1), y_val_t).item()
            outv = model(X_val_t)
            if task_type == "classification":
                val_loss = criterion(outv.view(-1), y_val_t.view(-1)).item()
            else:
                val_loss = criterion(outv.view(-1, 1), y_val_t).item()
        
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if epoch % 20 == 0:
                logging.info(f"[Epoch {epoch}] New best val_loss={val_loss:.4f}")
        else:
            wait += 1
            if wait >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return {"model": model, "best_params": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate}}

def _predict_dl(model, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Predict with a PyTorch model."""
    import torch
    device = _get_device()
    model = model.to(device).eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out = model(X_t[i:i + batch_size]).cpu().numpy().flatten()
            preds.append(out)
    return np.concatenate(preds)


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
    _ensure_dir(output_dir)
    if task_type == "classification":
        y_test = _ensure_binary_labels(y_test)
    is_dl = model_type.startswith("dl_")
    n_jobs = _resolve_n_jobs()
    X_train, X_test = _maybe_apply_xgboost_feature_map_for_explain(
        model_type=model_type,
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
    )

    try:
        import shap
    except Exception as exc:
        logging.warning("SHAP is not available; skipping SHAP explainability. %s", exc)
        return

    if is_dl:
        import torch
        class _SklearnWrapper:
            def __init__(self, model, task_type: str):
                self.model = model
                self.task_type = task_type

            def fit(self, X, y):
                return self

            def predict(self, X):
                y_out = _predict_dl(self.model, X.values if hasattr(X, "values") else X)
                if self.task_type == "classification":
                    scores = _validate_classification_score_values(
                        y_out,
                        context="dl explainability prediction",
                    )
                    proba = _sigmoid(scores)
                    return (proba >= 0.5).astype(int)
                return y_out

            def predict_proba(self, X):
                y_out = _predict_dl(self.model, X.values if hasattr(X, "values") else X)
                scores = _validate_classification_score_values(
                    y_out,
                    context="dl explainability prediction",
                )
                proba = _sigmoid(scores)
                return np.vstack([1.0 - proba, proba]).T

            def score(self, X, y):
                y_true = np.asarray(y).reshape(-1)
                if self.task_type == "classification":
                    y_true = _ensure_binary_labels(pd.Series(y_true)).to_numpy(dtype=int)
                    proba = self.predict_proba(X)[:, 1]
                    auc = _safe_auc(y_true, proba)
                    if auc is None:
                        pred = (proba >= 0.5).astype(int)
                        return float(accuracy_score(y_true, pred))
                    return float(auc)
                y_pred = _predict_dl(self.model, X.values if hasattr(X, "values") else X)
                y_true, y_pred = _validate_regression_metric_inputs(
                    y_true,
                    y_pred,
                    context="dl permutation importance scoring",
                )
                return float(r2_score(y_true, y_pred))
        
        wrapped_estimator = _SklearnWrapper(estimator, task_type=task_type)
        result = permutation_importance(
            wrapped_estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
        )
    else:
        scoring = "roc_auc" if task_type == "classification" else None
        result = permutation_importance(
            estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=n_jobs, scoring=scoring
        )


    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance": result.importances_mean}
    ).sort_values(by="importance", ascending=False)
    importance_path = os.path.join(output_dir, f"{model_type}_permutation_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    plt.figure(figsize=(10, 6))
    importance_df.head(20).plot.bar(x="feature", y="importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_type}_permutation_importance.png"))
    plt.close()

    try:
        if model_type in ["random_forest", "decision_tree", "xgboost", "catboost_classifier"]:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
            if task_type == "classification" and isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
        
        elif model_type == "ensemble":
            explainer = shap.Explainer(estimator.predict, X_test.iloc[:background_samples])
            shap_values = explainer(X_test)
        
        elif model_type == "svm":
            background = shap.sample(X_test, min(background_samples, len(X_test)))
            explainer = shap.KernelExplainer(estimator.predict, background)
            X_explain = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_explain)
            X_test = X_explain 
        
        elif model_type.startswith("dl_"): 
            device = _get_device()
            estimator.to(device).eval()
            
            n_bg = min(background_samples, len(X_train))
            n_ex = min(100, len(X_test))
            
            # Convert to numpy arrays for SHAP
            # X_bg_np = X_train.iloc[:n_bg].values.astype(np.float32)
            X_bg_np = X_train.sample(n=n_bg, random_state=42).values.astype(np.float32)
            X_ex_np = X_test.iloc[:n_ex].values.astype(np.float32)
            
            # Use KernelExplainer
            def model_predict(X):
                out = _predict_dl(estimator, X) 
                out = np.asarray(out).reshape(-1)
                if task_type == "classification":
                    return out  # <-- logits
                return out      
            
            explainer = shap.KernelExplainer(model_predict, X_bg_np)
            shap_values = explainer.shap_values(X_ex_np, nsamples=100)
            
            X_test = X_test.iloc[:n_ex]

        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, f"{model_type}_shap_summary.png"), bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logging.warning("SHAP explainability failed: %s", exc)
