# Training Module Plan

`MLModels/train_models.py` currently mixes model construction, validation, metrics,
plotting, persistence, explainability, and multiple training backends. This folder
is the planned home for a smaller training package that can support both the
CheMLFlow workflow and standalone scripts.

## Goals

- Keep the existing workflow API stable while internals move into focused modules.
- Make metrics, validation, plotting, and model-specific training independently
  testable.
- Support command-line scripts without duplicating workflow logic.
- Keep model-specific behavior out of generic training code.

## Proposed Layout

```text
MLModels/training/
  api.py              # public train/load/explain functions used by main.py
  config.py           # dataclasses and model/training option parsing
  validation.py       # input, label, shape, finite-value checks
  metrics.py          # regression/classification metrics
  losses.py           # PyTorch/DL loss selection
  plots.py            # ROC, PR, and split-performance plots
  sklearn_models.py   # RF/SVM/DT/XGBoost/ensemble model construction/training
  xgboost_utils.py    # XGBoost feature-name sanitization
  chemprop_models.py  # Chemprop/Chemeleon training and reload helpers
  torch_models.py     # DL train loop and PyTorch utilities
  persistence.py      # model/params/metrics save-load helpers
  explainability.py   # permutation importance, SHAP/tree explainers
```

`MLModels/train_models.py` should remain as a compatibility shim during the
transition. Existing callers should continue to use:

```python
from MLModels import train_models

train_models.train_model(...)
train_models.load_model(...)
train_models.run_explainability(...)
```

## Class Guidance

Use small dataclasses for configuration and results. Avoid a single large
`Trainer` class that owns every backend.

Good candidates:

```python
@dataclass(frozen=True)
class TrainingConfig:
    model_type: str
    task_type: str
    random_state: int
    n_jobs: int
    tuning_method: str
    model_params: dict[str, Any]


@dataclass(frozen=True)
class PlotConfig:
    output_dir: Path
    model_type: str
    task_type: str
```

For standalone scripts, prefer thin CLI wrappers around these dataclasses and
pure functions:

```text
scripts/plot_split_metrics.py   -> calls MLModels.training.plots
scripts/evaluate_predictions.py -> calls MLModels.training.metrics
```

This keeps script behavior and workflow behavior identical.

## Refactor Order

1. Extract `validation.py`, `metrics.py`, and `plots.py`.
2. Add focused unit tests for extracted functions.
3. Extract `config.py` and `persistence.py`.
4. Extract `xgboost_utils.py`.
5. Extract `sklearn_models.py`.
6. Extract `torch_models.py` and `chemprop_models.py` last.

This order keeps risk low because metrics and plotting have fewer side effects
than training backends.

## QM9 Notes

QM9 showed that model options need to be explicit and backend-specific. For
example, random forest should support scoped options like:

```yaml
train:
  model:
    type: random_forest
    params:
      n_jobs: 1
      n_estimators: 100
```

The generic training layer should parse these options, but memory-sensitive
behavior should live in the relevant backend module.
