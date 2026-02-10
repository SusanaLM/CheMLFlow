# CheMLFlow

CheMLFlow is an open source software to develop, implement and apply modern cheminformatics workflows.

## Pipeline vision (dataset‑agnostic tabular ML)

The pipeline is intended to be dataset‑agnostic for tabular ML tasks in chemistry/materials. Datasets can be local files, ChEMBL, or other sources; the required contract is defined **between nodes**, not per dataset. The only required inputs are:

- a tabular file (CSV),
- a `target_column` defined in config (for supervised tasks),
- and an optional featurizer (e.g., RDKit if SMILES are present).

Downstream steps enforce only the minimum required columns for their node (e.g., `canonical_smiles` for RDKit, `target_column` for model training), and extra columns are allowed.

### SMILES handling (important)

- Raw SMILES strings are **never** used directly as numeric features.
- If you want SMILES to drive the model, use a featurizer (e.g., RDKit/Morgan) to convert SMILES into numeric descriptors/fingerprints.
- If you are using existing tabular descriptors, SMILES is only used for **canonicalization** and **scaffold splitting**, then dropped from the feature matrix.

### Tabular descriptors (no featurizer)

If your dataset already includes numeric descriptors, use the `use.curated_features` node to point training at the curated CSV directly (no RDKit/Morgan). You can also allowlist low‑cardinality categorical columns for one‑hot encoding via:

```
preprocess:
  categorical_features:
    - Family
```

`categorical_features` and `target_column` must match **column names in your dataset**.

## Installation

1. Clone the repository

git clone https://github.com/nijamudheen/CheMLFlow.git

2. Create conda environment (Python 3.12 recommended)

cd CheMLFlow

conda create -n chemlflow_env_py312 python=3.12

conda activate chemlflow_env_py312

3. Install dependencies (macOS‑reliable, single path)

Install compiled deps from conda-forge first:

conda install -c conda-forge numpy scipy scikit-learn matplotlib-base seaborn lightgbm xgboost catboost rdkit shap numba llvmlite

Then install Python deps from pip (don’t re-resolve compiled libs):

pip install -r requirements.txt --no-deps

Notes:
- CatBoost is stable on Python 3.12. Python 3.13 often lacks wheels.
- SHAP relies on numba/llvmlite. On macOS, conda-forge is the most reliable install path.
- If `conda activate` fails, run `conda init zsh` once and restart your terminal (or `exec zsh`).
- Run commands from the repo root (`CheMLFlow/`) so relative paths resolve.
- If you see NumPy binary incompat errors after pip installs, reinstall numpy/numba/llvmlite/rdkit from conda-forge.

4. Optional: Install PyTorch + Optuna for DL models

For Linux/Windows (check for cuda version available. for instance, cu121 here)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install pytorch-lightning

pip install optuna

For AMD GPU (ROCm, Linux only)

pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

pip install pytorch-lightning

pip install optuna

For Apple Silicon (M1/M2/M3), no CUDA/GPU available 

Can use the MPS backend (built into the default macOS wheels) when using mps as the device
 
pip install torch torchvision torchaudio

pip install pytorch-lightning 

pip install optuna

5. Remove additional install files

make clean

## Urease pIC50 regression benchmarks (context)

These are not directly comparable (different datasets, descriptors, and splitting strategies), but useful for rough context.

| Item | External benchmark (BindDB + CORAL) | This repo (ChEMBL + RDKit + RF) |
| --- | --- | --- |
| Dataset | 436 urease inhibitors from BindDB (IC50 → pIC50), 3 random splits into Train/InvTrain/Cal/Val | ChEMBL urease IC50 via API; size varies by run |
| Model / descriptors | Monte-Carlo QSAR (CORALSEA 17) with hybrid SMILES + GRAPH descriptors | RandomForestRegressor + RDKit descriptors |
| Validation metrics | Q2 (Val): 0.667–0.763; MAE (Val): 0.320–0.340 (across 3 splits) | Example run (2026-01-26): Test R2 0.507; MAE 0.462; Nested CV R2 0.319 ± 0.160 |
| Source | https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c74fcf4c8919392aad3c85/original/monte-carlo-method-based-qsar-model-to-discover-phytochemical-urease-inhibitors-using-smiles-and-graph-descriptors.pdf | Local run output (results/random_forest_evaluation_results.txt) |

## Running tests

Scripts to run tests in CLI formats are in tests directory

For E2E tests that spawn `main.py`, ensure the subprocess uses your conda Python:

CHEMLFLOW_PYTHON=$(which python) pytest tests/test_e2e_pipelines.py -q

For full test runs, install pytest in your env:

pip install pytest

## Running pipelines and finding results

Run a pipeline by setting `CHEMLFLOW_CONFIG` and executing `main.py` from repo root:

CHEMLFLOW_CONFIG=config/config.qm9.yaml python main.py

Outputs:
- If `global.runs.enabled: true`, results go to `runs/<timestamp>/`
- Otherwise, results go to `results/`
- Data artifacts always live under `data/<dataset>/`
- Each run writes a `run.log` file under the run directory.

## Config structure (node‑style)

Each node has its own config block, and global settings live under `global`:
# - global: shared defaults used by multiple nodes
# - pipeline: ordered list of nodes to execute
# - node configs: per-node parameters (get_data, split, featurize, model, etc.)

global:
  pipeline_type: qm9
  task_type: regression
  base_dir: data/qm9
  target_column: gap
  thresholds:
    active: 1000
    inactive: 10000

get_data:
  data_source: local_csv
  source:
    path: local_data/qm9.csv

split:
  strategy: scaffold
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  stratify: true

## Quick start (pipelines)

All pipelines are config-driven. You select the pipeline by setting `CHEMLFLOW_CONFIG`
and running `main.py` from the `CheMLFlow` directory.

### Urease (ChEMBL → pIC50 → RDKit → Train → Explain)

1. Activate your environment:

conda activate chemlflow_env

2. Run the pipeline:

CHEMLFLOW_CONFIG=config/config.chembl.yaml python main.py

3. Outputs:
- Data artifacts: `data/urease/`
- Models + metrics: `results/`
- Explainability PNGs (permutation importance + SHAP): `results/`

Notes:
- The ChEMBL API can be temporarily unavailable. If it returns a 500, retry later or
  switch `get_data.data_source` to `local_csv` with a cached file.

### QM9 (Local CSV → RDKit → Preprocess → Select → Train → Explain)

1. Activate your environment:

conda activate chemlflow_env

2. Run the pipeline:

CHEMLFLOW_CONFIG=config/config.qm9.yaml python main.py

3. Outputs:
- Data artifacts: `data/qm9/`
- Models + metrics: `results/`
- Explainability PNGs (permutation importance + SHAP): `results/`

Notes:
- Control dataset size via `get_data.max_rows` in `config/config.qm9.yaml`.
- Model choice is controlled by `model.type` (e.g., `random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble`).

### YSI (Sooting Index, local CSV → RDKit → Train → Explain)

1. Ensure the dataset exists at:

`local_data/ysi.csv`

Expected columns include `SMILES` and `YSI`.

2. Run the pipeline:

CHEMLFLOW_CONFIG=config/config.ysi.yaml python main.py

### PAH (logP, local CSV → RDKit → Train → Explain)

1. Ensure the dataset exists at:

`local_data/arockiaraj_pah_data.csv`

Expected columns include `smiles` and `log_p`.

2. Run the pipeline:

CHEMLFLOW_CONFIG=config/config.pah.yaml python main.py

### Pgp_Broccatelli (Local CSV → Morgan → CatBoost → AUROC)

1. Activate your environment:

conda activate chemlflow_env

2. Run the pipeline:

CHEMLFLOW_CONFIG=config/config.pgp.yaml python main.py

3. Outputs:
- Data artifacts: `data/pgp_broccatelli/`
- Models + metrics: `runs/<timestamp>/` (or `run_dir` if configured)
- Explainability PNGs (permutation importance + SHAP): `runs/<timestamp>/`

Notes:
- This config expects `local_data/pgp_broccatelli.csv`.
- Export it once via:
  - `python utilities/export_pgp_tdc.py local_data/pgp_broccatelli.csv`
- If `pytdc` is not installed in the main env, run the export script in a small temporary env.
- Split strategy defaults to `scaffold` (configurable in `split.strategy`).

### ARA (Androgen Receptor Antagonist, AR.csv → Morgan → CatBoost → AUROC)

1. Place the dataset at:

`local_data/AR.csv`

2. Run the pipeline:

CHEMLFLOW_CONFIG=config/config.ara.yaml python main.py

Notes:
- Expected columns: `Smiles` and `Activity` (`active`/`inactive`).
- Split strategy defaults to `scaffold` (configurable in `split.strategy`).
- SOTA reference from literature: AUROC ≈ 0.945 (DeepAR).

### Common config knobs

In `config/*.yaml`:
- `model.type`: model selection
- `model.cv_folds`, `model.search_iters`: CV + search effort
- `preprocess.*`: preprocessing thresholds and split settings
- `pipeline.nodes`: ordered list of steps (e.g., add/remove `explain`)
- `task_type`: `regression` or `classification`
- `split.*`: split strategy and sizes (e.g., `random`, `scaffold`, `tdc_scaffold`)
- `featurize.*`: featurizer settings (e.g., Morgan radius/n_bits)
- `global.runs.enabled`: use `runs/<timestamp>` instead of `results/`

## Chemprop backend (optional, classification only)

CheMLFlow supports an in-process Chemprop D-MPNN backend behind `model.type: chemprop`.
This path is SMILES-native (no descriptor generation) and uses CheMLFlow's `split_indices`
from the `split` node for apples-to-apples split comparability.

### Install

Chemprop is an optional dependency (it typically brings in PyTorch and Lightning):

- `pip install chemprop`
- or, if you install CheMLFlow as a package: `pip install -e ".[chemprop]"`

### Example config

Use the included example: `config/config.pgp_chemprop.yaml`:

```
CHEMLFLOW_CONFIG=config/config.pgp_chemprop.yaml python main.py
```

### Target column

Chemprop trains on the pipeline context's `target_column`:
- set `global.target_column` as the canonical target name.
- if you run `label.normalize`, you can set/override via `label.target_column` (the normalize node writes that column and updates the context).

### SMILES column

Chemprop expects SMILES in the curated dataset.
Default is `canonical_smiles`. Override via `model.smiles_column` if needed.

### Chemprop params

Put these under `model.params` (all optional; defaults exist):
- `max_epochs`
- `batch_size`
- `init_lr`, `max_lr`, `final_lr` (or `lr` as a fallback)
- `mp_hidden_dim`
- `mp_depth`
- `ffn_hidden_dim`

### Artifacts written

Under the run directory (e.g., `runs/<timestamp>/`):
- `chemprop_best_model.ckpt`
- `chemprop_best_params.pkl` (do not load untrusted pickle/joblib files)
- `chemprop_metrics.json`
- `chemprop_predictions.csv`
