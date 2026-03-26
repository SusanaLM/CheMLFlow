# DOE Config Generation

CheMLFlow supports generating many **runtime-valid** configs from one DOE YAML file.
The generator expands your search space, filters invalid combinations, and writes:

- `manifest.jsonl` (one row per attempted execution child, including skipped reasons)
- `parent_manifest.jsonl` (one row per scientific parent config)
- `summary.json` (counts, profile/task, selection metadata, DOE spec hash/snapshot path)
- one config YAML per valid case

Script:

```bash
python scripts/generate_doe.py --doe config/doe.example.yaml
```

## Why this exists

- Keeps split/evaluation policy consistent across models.
- Prevents known invalid combinations from reaching cluster jobs.
- Produces auditable case metadata (`status`, `issues`, `config_hash`).

## Supported profiles

- `reg_local_csv`
- `reg_chembl_ic50`
- `clf_local_csv`
- `clf_tdc_benchmark`

If `dataset.profile` is omitted, the generator infers a profile from:

- `dataset.task_type`
- `dataset.source.type`

For `task_type: auto`, set `dataset.auto_confirmed: true`.

## DOE schema (v1)

Top-level keys:

- `version` (must be `1`)
- `dataset`
- `search_space`
- `defaults` (optional)
- `constraints` (optional)
- `selection` (optional)
- `output`

Important conventions:

- `search_space` keys use dotted paths (for example `split.mode`, `train.model.type`).
- Values can be a scalar or list; scalar is treated as a single-value axis.
- `defaults` uses the same dotted-path style and is applied before `search_space` factors.
- For `split.mode: cv` or `nested_holdout_cv`, keep `n_splits` / `repeats` in `defaults`.
- If fold/repeat indices are omitted from both `defaults` and `search_space`, DOE generation expands all folds/repeats automatically.
- Set fold/repeat indices explicitly only when you intentionally want a single execution slice (for example debugging or retrying one failed fold).
- DOE uses a parent/child shape:
  - one scientific parent per logical config
  - one execution child per concrete split slice
  - holdout is a trivial one-child parent
- By default, generated cases are isolated per DOE spec hash + case id:
  - `global.base_dir` becomes `<base_dir>/<doe_spec_hash[:8]>/<case_id>`
  - `global.run_dir` becomes `<run_dir>/<doe_spec_hash[:8]>/<case_id>` (or `<output.dir>/runs/<doe_spec_hash[:8]>/<case_id>`)
  - `global.runs.id` is set to `case_id`
  Set `constraints.isolate_case_artifacts: false` only if you intentionally want shared artifacts.
- Safety cap: if `constraints.max_cases` is omitted and expanded combinations exceed 10,000, DOE generation fails fast.

## Required dataset fields (typical local CSV)

```yaml
dataset:
  task_type: classification   # or regression
  target_column: label
  source:
    type: local_csv
    path: local_data/my_data.csv
```

For regression with `source.type: local_csv`, `target_column` is required and must exist in the CSV.

For classification with non-binary raw labels, provide a label map:

```yaml
dataset:
  label_source_column: Activity
  label_map:
    positive: [active, "1", 1]
    negative: [inactive, "0", 0]
```

## Compatibility checks (skipped with reason codes)

Examples:

- `DOE_MODEL_TASK_MISMATCH`
- `DOE_SPLIT_STRATEGY_MODE_INVALID`
- `DOE_VALIDATION_SPLIT_REQUIRED`
- `DOE_SELECT_REQUIRES_PREPROCESS`
- `DOE_FEATURE_INPUT_REQUIRED`
- `DOE_FEATURE_INPUT_REQUIRED_FOR_PREPROCESS`
- `DOE_CHEMPROP_PREPROCESS_UNSUPPORTED`
- `DOE_SPLIT_PARAM_INVALID`
- `DOE_DATASET_COLUMN_MISSING`
- `DOE_TARGET_COLUMN_MISSING`
- `DOE_SMILES_COLUMN_MISSING`
- `DOE_CURATE_DEDUPE_INVALID`
- `DOE_CURATE_TARGET_DROPPED`
- `DOE_RUNTIME_SCHEMA_INVALID`

`manifest.jsonl` contains these codes per skipped execution child.

## Selection metric defaults

- Classification default: `auc`
- Regression default: `r2`

You can override in `selection.primary_metric`.

## Practical defaults

- `clf_local_csv` + non-chemprop models default to `pipeline.feature_input: featurize.morgan`.
- `chemprop` / `chemeleon` default to `pipeline.feature_input: smiles_native` when the DOE omits that axis.
- `reg_chembl_ic50` defaults `global.target_column` to `pIC50`.
- For DOE comparisons across pipelines, strongly consider:
  - `split.require_disjoint: true`
  - `split.require_full_test_coverage: true`

## Scientific selection guidance

- Avoid selecting the "best" config by comparing many configs on one fixed test split.
- Prefer `split.mode: nested_holdout_cv` (or repeated CV) and aggregate metrics across folds/repeats.
- Use the final untouched holdout only once for final reporting.

## DOE best practices

Use these rules when building DOE files for cluster runs.

### 1) Use one DOE file per split mode

- Keep `holdout`, `cv`, and `nested_holdout_cv` in separate DOE specs.
- Reason: DOE uses a cartesian grid and does not support conditional axes.
- If you mix modes in one grid, some execution settings become meaningless for some rows (for example CV fold settings in holdout mode).

### 2) Keep only meaningful search axes

- Put fixed choices in `defaults`.
- Put only true experiment axes in `search_space`.
- Good: vary `train.model.type`, `split.strategy`, and maybe one featurizer.
- Avoid large mixed grids unless you need them; they grow very quickly.
- For `chemprop` / `chemeleon`, keep `pipeline.feature_input: smiles_native`.
- If you keep `pipeline.preprocess: true` in a mixed-model DOE, only the no-op branch (`preprocess.scaler: none`) is meaningful for `chemprop` / `chemeleon`.
- Treat `smiles_native` as reserved for SMILES-native models. DOE will skip tabular models on that branch.

### 3) For chemistry model comparison, prefer scaffold CV

- Use `split.mode: cv` with `split.strategy: scaffold` for generalization-focused comparison.
- Set at least:
  - `split.cv.n_splits: 5`
  - `split.cv.repeats: 1` (increase if budget allows)
- Let DOE auto-expand folds/repeats unless you are debugging a specific failed fold.
- Requirement: scaffold CV needs `canonical_smiles` in curated data.

### 4) Separate hyperparameter search from evaluation design

- `train.tuning.method: train_cv` is inner model tuning.
- `split.mode: cv` or `nested_holdout_cv` is outer evaluation design.
- Do not treat inner tuning CV alone as final performance evidence.

### 5) Run a small pilot before full DOE

- First run with:
  - one model family
  - one feature mode
  - a few cases only
- Confirm:
  - metrics files are written
  - split metadata is produced
  - runtime is acceptable
  - no recurring OOM/timeouts

### 6) Make failure handling explicit

- Always inspect:
  - `summary.json`
  - `manifest.jsonl`
  - case `run_status.json` and cluster logs
- Treat obvious pathological runs (for example extreme MAE / huge negative R2) as failed for selection.
- Re-run only failed/stuck cases with adjusted resources or safer model params.

### 7) Report robustly, not by single best point

- Aggregate execution children back to the parent:
  - mean/median
  - spread (`std` or IQR)
- Compare models on matched splits where possible.
- Keep holdout-only sweeps for debugging or preliminary screening, not final claims.

### 8) Use deterministic, auditable settings

- Set `global.random_state` explicitly.
- Keep `constraints.isolate_case_artifacts: true`.
- Preserve generated configs + manifests with results bundles for reproducibility.
