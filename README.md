# CheMLFlow

CheMLFlow is an open source software to develop, implement and apply modern cheminformatics workflows.

## Pipeline vision (dataset‑agnostic tabular ML)

The pipeline is intended to be dataset‑agnostic for tabular ML tasks in chemistry/materials. Datasets can be local files, ChEMBL, or other sources; the required contract is defined **between nodes**, not per dataset. The only required inputs are:

- a tabular file (CSV),
- a `target_column` defined in config (for supervised tasks),
- and an optional featurizer (e.g., RDKit if SMILES are present).

Downstream steps enforce only the minimum required columns for their node (e.g., `canonical_smiles` for RDKit, `target_column` for model training), and extra columns are allowed.

## Installation

1. Clone the repository

git clone https://github.com/nijamudheen/CheMLFlow.git

2. Create conda environment 

cd CheMLFlow

conda create -n chemlflow_env python=3.13

conda activate chemlflow_env

3. Install dependencies

pip install -e .

4. Install RDKit from via conda or pip install

Conda installation (recommended)

conda install -c conda-forge rdkit

pip installation (works for most applications)

pip install rdkit 


5. Install PyTorch and PyTorch Lightning with GPU support, and Optuna for hyperparameter optimization

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

6. Remove additional install files

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
  switch to `data_source: local_csv` with a cached file.

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
- Control dataset size via `qm9.max_rows` in `config/config.qm9.yaml`.
- Model choice is controlled by `model.type` (e.g., `random_forest`, `svm`, `decision_tree`, `xgboost`, `ensemble`).

### Common config knobs

In `config/*.yaml`:
- `model.type`: model selection
- `model.cv_folds`, `model.search_iters`: CV + search effort
- `preprocess.*`: preprocessing thresholds and split settings
- `pipeline.nodes`: ordered list of steps (e.g., add/remove `explain`)
