# MLModels Training Package

This package contains modular training utilities used by `MLModels/train_models.py`.

## Files

- `api.py`
  - Public API wrappers for training workflows.
  - Provides `DatasetSplit`, `TrainSpec`, `train`, `train_from_frames`, `load`, and `run_explainability`.
- `config.py`
  - Runtime config parsing and normalization helpers.
  - Includes `RuntimeTrainingOptions`, `as_bool`, `resolve_n_jobs`, and Chemprop foundation config parsing.
- `metrics.py`
  - Classification and regression metric helpers and validation.
  - Includes safe metric wrappers (AUC/AUPRC/R2/MAE) and split metric assembly helpers.
- `plots.py`
  - Plot and artifact writers for ROC/PR/confusion/split-metrics/parity plots.
- `persistence.py`
  - Model/metrics/params persistence helpers (`joblib`, JSON, torch state dict).
- `torch_models.py`
  - PyTorch DL helpers for device selection, deterministic seeding, training, prediction, and Optuna tuning.
- `__init__.py`
  - Package exports for the modules above.

## Usage

Current workflows still call through `MLModels.train_models`. You can also call the package API directly:

```python
from MLModels.training.api import DatasetSplit, TrainSpec, train

dataset = DatasetSplit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_val=X_val,
    y_val=y_val,
)

spec = TrainSpec(
    model_type="random_forest",
    output_dir="runs/example",
    task_type="regression",
)

model, result = train(dataset, spec)
```

## Compatibility

`MLModels/train_models.py` remains the compatibility surface for existing callers while logic is moved into this package.
