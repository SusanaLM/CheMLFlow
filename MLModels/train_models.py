import json
import os
import logging
import shutil
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance


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
    os.makedirs(path, exist_ok=True)

def _resolve_n_jobs(model_config: Dict[str, Any] | None = None) -> int:
    """Resolve parallelism level for scikit-learn/joblib.

    Default to single-thread under pytest (subprocesses inherit PYTEST_CURRENT_TEST),
    since some sandboxes/CI environments disallow the syscalls loky uses to size its pool.
    """
    model_config = model_config or {}

    value = model_config.get("n_jobs")
    if value is None:
        env = os.environ.get("CHEMLFLOW_N_JOBS")
        if env:
            try:
                value = int(env)
            except ValueError:
                logging.warning("Invalid CHEMLFLOW_N_JOBS=%r; falling back to defaults.", env)

    if value is None:
        value = 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1

    try:
        value_int = int(value)
    except (TypeError, ValueError):
        logging.warning("Invalid n_jobs=%r; using 1.", value)
        return 1
    if value_int == 0:
        logging.warning("n_jobs=0 is invalid; using 1.")
        return 1

    # Some environments (notably sandboxed macOS setups) raise PermissionError on
    # os.sysconf("SC_SEM_NSEMS_MAX"), which joblib/loky calls when starting a
    # process pool. Fall back to single-thread execution to keep pipelines runnable.
    if value_int != 1:
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            logging.warning(
                "Parallel joblib backend is not permitted in this environment; forcing n_jobs=1."
            )
            return 1
        except Exception:
            # If sysconf is unavailable/unsupported, let joblib decide.
            pass
    return value_int


def _safe_auc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        logging.warning("AUROC undefined for single-class test set.")
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        logging.warning("AUPRC undefined for single-class test set.")
        return None
    return float(average_precision_score(y_true, y_score))


def _save_roc_curve(output_dir: str, model_type: str, y_true, y_score) -> str | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    roc_path = os.path.join(output_dir, f"{model_type}_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    return roc_path


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _safe_r2(y_true, y_pred) -> float | None:
    try:
        return float(r2_score(y_true, y_pred))
    except ValueError:
        return None


def _safe_mae(y_true, y_pred) -> float | None:
    try:
        return float(mean_absolute_error(y_true, y_pred))
    except ValueError:
        return None


def _save_split_metrics_artifacts(
    output_dir: str,
    model_type: str,
    split_metrics: Dict[str, Dict[str, float | None]],
) -> tuple[str | None, str | None]:
    if not split_metrics:
        return None, None

    split_order = [name for name in ("train", "val", "test") if name in split_metrics]
    if not split_order:
        split_order = list(split_metrics.keys())

    metrics_json_path = os.path.join(output_dir, f"{model_type}_split_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(split_metrics, f, indent=2)

    df = pd.DataFrame.from_dict(split_metrics, orient="index")
    df = df.reindex(split_order)
    df = df.apply(pd.to_numeric, errors="coerce")
    metric_names = [name for name in df.columns if df[name].notna().any()]
    if not metric_names:
        return metrics_json_path, None

    fig, axes = plt.subplots(
        nrows=len(metric_names),
        ncols=1,
        figsize=(8, max(3, 2.6 * len(metric_names))),
        squeeze=False,
    )
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx, 0]
        values = df[metric_name]
        ax.bar(df.index.astype(str), values.values)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_type}_split_metrics.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return metrics_json_path, plot_path


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
    foundation = str(model_config.get("foundation", "none")).strip().lower()
    if foundation in {"", "none"}:
        foundation = "none"
    if foundation not in {"none", "chemeleon"}:
        raise ValueError(
            "model.foundation must be one of: 'none', 'chemeleon'."
        )

    freeze_encoder = _as_bool(model_config.get("freeze_encoder", False))
    checkpoint_path: str | None = None
    if foundation == "chemeleon":
        raw_path = model_config.get("foundation_checkpoint")
        if not raw_path or not str(raw_path).strip():
            raise ValueError(
                "model.foundation_checkpoint is required when model.foundation='chemeleon'."
            )
        checkpoint_path = os.path.expanduser(str(raw_path).strip())
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"model.foundation_checkpoint does not exist or is not a file: {checkpoint_path}"
            )
    elif freeze_encoder:
        raise ValueError(
            "model.freeze_encoder=true requires model.foundation='chemeleon'."
        )

    return foundation, checkpoint_path, freeze_encoder


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

    This path is intentionally SMILES-native. It expects the curated dataset to include a
    `canonical_smiles` column (or an override via model_config.smiles_column).

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

    if task_type != "classification":
        raise ValueError("chemprop integration currently supports classification only.")

    smiles_column = model_config.get("smiles_column", "canonical_smiles")
    if smiles_column not in curated_df.columns:
        raise ValueError(
            f"chemprop requires smiles_column={smiles_column!r} in curated_df. "
            "Ensure curate emits canonical_smiles or override model.smiles_column."
        )
    if target_column not in curated_df.columns:
        raise ValueError(f"Target column {target_column!r} not found in curated_df.")

    y_all = _ensure_binary_labels(curated_df[target_column])
    # Keep a stable row_id (0..N-1) for joining predictions to the curated dataset.
    df = curated_df.reset_index(drop=True).copy()
    df["_row_id"] = df.index.astype(int)
    df["_label"] = y_all.reset_index(drop=True).astype(int)

    # Split indices are defined over curated_df row positions.
    tr_idx = [int(i) for i in split_indices.get("train", [])]
    va_idx = [int(i) for i in split_indices.get("val", [])]
    te_idx = [int(i) for i in split_indices.get("test", [])]
    if not te_idx and va_idx:
        te_idx, va_idx = va_idx, []
    if not tr_idx or not te_idx:
        raise ValueError("chemprop training requires non-empty train and test splits.")
    if not va_idx:
        raise ValueError(
            "chemprop training requires an explicit validation split from the split node. "
            "Set split.val_size > 0."
        )

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

    # Chemprop FFN constructor changed across versions:
    # - older: BinaryClassificationFFN(..., d_mp=..., d_h=...)
    # - newer: BinaryClassificationFFN(..., input_dim=..., hidden_dim=...)
    ffn_sig = inspect.signature(nn.BinaryClassificationFFN.__init__)
    ffn_kwargs: Dict[str, Any] = {"n_tasks": 1}
    if "d_mp" in ffn_sig.parameters:
        ffn_kwargs["d_mp"] = mp_output_dim
    if "input_dim" in ffn_sig.parameters:
        ffn_kwargs["input_dim"] = mp_output_dim
    if "d_h" in ffn_sig.parameters:
        ffn_kwargs["d_h"] = ff_hidden
    if "hidden_dim" in ffn_sig.parameters:
        ffn_kwargs["hidden_dim"] = ff_hidden
    ffn = nn.BinaryClassificationFFN(**ffn_kwargs)
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

    # Predict probabilities from the best checkpoint.
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

    def _predict_probs(loader) -> np.ndarray:
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
        if np.nanmin(preds) < 0.0 or np.nanmax(preds) > 1.0:
            # Some configs may emit logits; convert to probabilities for metrics.
            preds = 1.0 / (1.0 + np.exp(-preds))
        return preds

    preds = _predict_probs(test_loader)

    y_true = df.loc[te_idx, "_label"].to_numpy(dtype=int)
    y_pred = (preds >= 0.5).astype(int)

    metrics = {
        "auc": _safe_auc(y_true, preds),
        "auprc": _safe_auprc(y_true, preds),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
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
    if plot_split_performance:
        train_preds = _predict_probs(train_eval_loader)
        val_preds = _predict_probs(val_loader)
        train_y = df.loc[tr_idx, "_label"].to_numpy(dtype=int)
        val_y = df.loc[va_idx, "_label"].to_numpy(dtype=int)
        split_metrics: Dict[str, Dict[str, float | None]] = {
            "train": {
                "auc": _safe_auc(train_y, train_preds),
                "auprc": _safe_auprc(train_y, train_preds),
                "accuracy": float(accuracy_score(train_y, (train_preds >= 0.5).astype(int))),
                "f1": float(f1_score(train_y, (train_preds >= 0.5).astype(int))),
            },
            "val": {
                "auc": _safe_auc(val_y, val_preds),
                "auprc": _safe_auprc(val_y, val_preds),
                "accuracy": float(accuracy_score(val_y, (val_preds >= 0.5).astype(int))),
                "f1": float(f1_score(val_y, (val_preds >= 0.5).astype(int))),
            },
            "test": {
                "auc": metrics["auc"],
                "auprc": metrics["auprc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
            },
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
    joblib.dump(best_params, params_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(output_dir, "chemprop_predictions.csv")
    out_pred = df.loc[te_idx, ["_row_id", smiles_column, target_column]].copy()
    out_pred.rename(columns={target_column: "y_true", smiles_column: "smiles"}, inplace=True)
    out_pred["y_proba"] = preds
    out_pred["y_pred"] = y_pred
    out_pred.to_csv(pred_path, index=False)

    logging.info("Training complete (chemprop): metrics=%s", {k: metrics[k] for k in ["auc", "auprc", "accuracy", "f1"]})
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
) -> Tuple[object, dict]:
    try:
        import optuna
    except Exception as exc:
        raise ImportError("optuna is required for DL hyperparameter search.") from exc
    
    if X_val is None or y_val is None or len(y_val) == 0:
        raise ValueError("DL hyperparameter search requires an explicit validation split (X_val/y_val).")
    
    best_model = None
    best_score = float("-inf")
    
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
        )
        
        # Evaluate on validation set
        trained_model = result["model"]

        # Predict
        y_pred = _predict_dl(trained_model, X_val)

        # NaN/Inf handling
        y_true = np.asarray(y_val).reshape(-1)
        y_hat = np.asarray(y_pred).reshape(-1)

        if y_true.shape[0] != y_hat.shape[0]:
            raise optuna.exceptions.TrialPruned(
            f"Shape mismatch y_val={y_true.shape} y_pred={y_hat.shape}")

        if not np.isfinite(y_true).all():
            raise optuna.exceptions.TrialPruned("NaN/Inf in y_val (data issue)")

        if not np.isfinite(y_hat).all():
            # Common when training diverges for certain hyperparams
            raise optuna.exceptions.TrialPruned("NaN/Inf in y_pred (diverged training)")


        r2 = r2_score(y_val, y_pred)
        
        # Track best
        if r2 > best_score:
            best_score = r2
            best_model = trained_model
            logging.info(f"New best R2={r2:.4f} with params: {params}")
        
        import gc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        del model, result, y_pred
        gc.collect()
        
        return r2  # Optuna maximizes by default when direction="maximize"
    
    # Create and run study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_evals, show_progress_bar=True)
    
    best_params = study.best_params
    logging.info(f"Optuna complete. Best R2={best_score:.4f}, params={best_params}")
    
    return best_model, best_params

def _initialize_model(
    model_type: str,
    random_state: int,
    cv_folds: int,
    search_iters: int,
    input_dim: int = None,
    n_jobs: int = -1,
):
    if model_type == "random_forest":
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            "max_depth": [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }
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
        param_grid_svm = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "epsilon": [0.1, 0.2, 0.5],
        }
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
        base_xgb = XGBRegressor(objective="reg:squarederror", random_state=random_state)
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
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            max_features="sqrt",
            bootstrap=False,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=random_state,
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
        )
        return VotingRegressor([("rf", rf), ("xgb", xgb)], n_jobs=n_jobs)
    
    if model_type == "dl_simple":
        if input_dim is None:
            raise ValueError("input_dim required for DL models")
        from DLModels.simpleregressionnn import SimpleRegressionNN
        return DLSearchConfig(
            model_class=lambda params: SimpleRegressionNN(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_dim", 256),
                use_tropical=params.get("use_tropical", False),
            ),
            search_space={
                "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
                "use_tropical": {"type": "categorical", "choices": [True, False]},
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
                hidden_size=params.get("hidden_size", 512),
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
                "hidden_size": 512,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": 32,
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
                n_heads=params.get("n_heads", 8),
                n_layers=params.get("n_layers", 4),
                dropout=params.get("dropout", 0.1),
            ),
            search_space={
                "embed_dim": {"type": "categorical", "choices": [64, 128, 256]},
                "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
                "n_layers": {"type": "categorical", "choices": [3, 4]}, #  (deleted 6)
                "dropout": {"type": "float", "low": 0.0, "high": 0.4, "log": False},
                "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [16, 32]}, # Change to adapt to higher dimensional data (deleted 128, 64)
                "epochs": {"type": "categorical", "choices": [100, 200, 300]},
            },
            default_params={
                "embed_dim": 128,
                "n_heads": 8,
                "n_layers": 4,
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "batch_size": 32,
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
    ) -> Dict[str, Any]:
    """Train a PyTorch model. Returns dict with model and best_params."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    device = _get_device()
    model = model.to(device)
    
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(np.asarray(y_train).reshape(-1, 1), dtype=torch.float32, device=device)

    if X_val is None or y_val is None:
        raise ValueError("DL training requires X_val/y_val for early stopping.")
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(np.asarray(y_val).reshape(-1, 1), dtype=torch.float32, device=device)
    
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_loss, best_state, wait = float('inf'), None, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx).view(-1, 1), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t).view(-1, 1), y_val_t).item()
        
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
    model_config = model_config or {}
    plot_split_performance = _as_bool(model_config.get("plot_split_performance", False))
    debug_logging = _as_bool(model_config.get("_debug_logging", False))
    n_jobs = _resolve_n_jobs(model_config)

    logging.info(
        "Training start: model=%s task=%s X_train=%s X_test=%s",
        model_type,
        task_type,
        X_train.shape,
        X_test.shape,
    )

    if task_type == "classification" and model_type == "catboost_classifier":
        from catboost import CatBoostClassifier

        y_train = _ensure_binary_labels(y_train)
        y_test = _ensure_binary_labels(y_test)
        if y_val is not None:
            y_val = _ensure_binary_labels(y_val)

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
        metrics = {
            "auc": _safe_auc(y_test, y_pred_proba),
            "auprc": _safe_auprc(y_test, y_pred_proba),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        if plot_split_performance:
            split_metrics: Dict[str, Dict[str, float | None]] = {
                "train": {
                    "auc": _safe_auc(y_train, estimator.predict_proba(X_train)[:, 1]),
                    "auprc": _safe_auprc(y_train, estimator.predict_proba(X_train)[:, 1]),
                    "accuracy": float(accuracy_score(y_train, estimator.predict(X_train))),
                    "f1": float(f1_score(y_train, estimator.predict(X_train))),
                },
                "test": metrics.copy(),
            }
            if X_val is not None and y_val is not None and len(y_val) > 0:
                y_val_proba = estimator.predict_proba(X_val)[:, 1]
                y_val_pred = estimator.predict(X_val)
                split_metrics["val"] = {
                    "auc": _safe_auc(y_val, y_val_proba),
                    "auprc": _safe_auprc(y_val, y_val_proba),
                    "accuracy": float(accuracy_score(y_val, y_val_pred)),
                    "f1": float(f1_score(y_val, y_val_pred)),
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

        roc_path = _save_roc_curve(output_dir, model_type, y_test, y_pred_proba)
        if roc_path:
            metrics["roc_curve_path"] = roc_path
        params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
        joblib.dump(best_params, params_path)
        pd.Series(metrics).to_json(metrics_path)
        logging.info("Training complete (classification): metrics=%s", metrics)
        logging.info("Artifacts: model=%s metrics=%s params=%s", model_path, metrics_path, params_path)
        return estimator, TrainResult(model_path, params_path, metrics_path)

    if task_type == "classification":
        raise ValueError(f"Unsupported classification model type: {model_type}")

    model = _initialize_model(
        model_type,
        random_state,
        cv_folds,
        search_iters,
        input_dim=X_train.shape[1] if is_dl else None,
        n_jobs=n_jobs,
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
            )
        else:
            logging.info(f"Training DL model: {model_type} (default params)")
            nn_model = model.model_class(model.default_params)
            result = _train_dl(
                nn_model,
                X_train.values,
                y_train.values,
                X_val.values,
                y_val.values,
                epochs=model.default_params["epochs"],
                batch_size=model.default_params["batch_size"],
                learning_rate=model.default_params["learning_rate"],
                patience=patience,
            )
            estimator = result["model"]
            best_params = result["best_params"]
        
        y_pred = _predict_dl(estimator, X_test.values)
        import torch
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pth")
        torch.save(estimator.state_dict(), model_path)
    else:

        logging.info(f"Training ML model: {model_type}")
        model.fit(X_train, y_train)

        estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        y_pred = estimator.predict(X_test)
        model_path = os.path.join(output_dir, f"{model_type}_best_model.pkl")
        joblib.dump(estimator, model_path)
        best_params = model.best_params_ if hasattr(model, "best_params_") else {}

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    if plot_split_performance:
        if is_dl:
            y_train_pred = _predict_dl(estimator, X_train.values)
            y_test_pred = _predict_dl(estimator, X_test.values)
            y_val_pred = _predict_dl(estimator, X_val.values) if X_val is not None else None
        else:
            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)
            y_val_pred = estimator.predict(X_val) if X_val is not None else None

        split_metrics: Dict[str, Dict[str, float | None]] = {
            "train": {
                "r2": _safe_r2(y_train, y_train_pred),
                "mae": _safe_mae(y_train, y_train_pred),
            },
            "test": {
                "r2": _safe_r2(y_test, y_test_pred),
                "mae": _safe_mae(y_test, y_test_pred),
            },
        }
        if X_val is not None and y_val is not None and len(y_val) > 0 and y_val_pred is not None:
            split_metrics["val"] = {
                "r2": _safe_r2(y_val, y_val_pred),
                "mae": _safe_mae(y_val, y_val_pred),
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

    params_path = os.path.join(output_dir, f"{model_type}_best_params.pkl")
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    joblib.dump(best_params, params_path)   
    pd.Series(metrics).to_json(metrics_path)
    logging.info("Training complete (regression): metrics=%s", metrics)
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
            saved_params = joblib.load(params_path)
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
    return joblib.load(model_path)

def run_explainability(
    estimator: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    output_dir: str,
    background_samples: int = 100,
    ) -> None:
    _ensure_dir(output_dir)
    is_dl = model_type.startswith("dl_")
    n_jobs = _resolve_n_jobs()

    try:
        import shap
    except Exception as exc:
        logging.warning("SHAP is not available; skipping SHAP explainability. %s", exc)
        return

    if is_dl:
        import torch
        class _SklearnWrapper:
            def __init__(self, model):
                self.model = model
            def fit(self, X, y):
                return self
            def predict(self, X):
                return _predict_dl(self.model, X.values if hasattr(X, 'values') else X)
            def score(self, X, y):
                y_pred = self.predict(X)
                return r2_score(y, y_pred)
        
        wrapped_estimator = _SklearnWrapper(estimator)
        result = permutation_importance(
            wrapped_estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1
        )
    else:
        result = permutation_importance(
            estimator, X_test, y_test, n_repeats=10, random_state=42, n_jobs=n_jobs
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
            
            # Convert to numpy arrays (not tensors) for SHAP
            X_bg_np = X_train.iloc[:n_bg].values.astype(np.float32)
            X_ex_np = X_test.iloc[:n_ex].values.astype(np.float32)
            
            # Use KernelExplainer instead (more reliable for PyTorch)
            def model_predict(X):
                return _predict_dl(estimator, X)
            
            explainer = shap.KernelExplainer(model_predict, X_bg_np)
            shap_values = explainer.shap_values(X_ex_np, nsamples=100)
            
            # Keep DataFrame for feature names in plot
            X_test = X_test.iloc[:n_ex]
        
        else:
            background = shap.sample(X_test, min(background_samples, len(X_test)))
            explainer = shap.KernelExplainer(estimator.predict, background)
            X_explain = X_test.iloc[:min(100, len(X_test))]
            shap_values = explainer.shap_values(X_explain)
            X_test = X_explain

        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, f"{model_type}_shap_summary.png"), bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logging.warning("SHAP explainability failed: %s", exc)
