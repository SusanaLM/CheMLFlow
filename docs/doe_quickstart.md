# DOE Quickstart (Ultra Simple)

This is the shortest path to generate DOE configs and run them.

## 1) Activate env

```bash
conda activate chemlflow_env_test
```

## 2) Edit the example DOE file

Open and set at least:

- `dataset.source.path`
- `dataset.target_column`
- `dataset.smiles_column` (for local CSV)
- `search_space` values you want to test

File:

```text
config/doe.example.yaml
```

## 3) Generate DOE configs

```bash
python scripts/generate_doe.py --doe config/doe.example.yaml
```

## 4) Check what was generated

```bash
cat config/generated/flash_doe/summary.json
head -n 20 config/generated/flash_doe/manifest.jsonl
ls config/generated/flash_doe/*.yaml
```

What these mean:

- `summary.json`: total cases, valid cases, skipped cases
- `manifest.jsonl`: per-case reason codes for skipped combos
- `*.yaml`: runnable configs for valid cases

## 5) Run one generated config

```bash
CHEMLFLOW_CONFIG=config/generated/flash_doe/<case_file>.yaml python main.py
```

Example:

```bash
CHEMLFLOW_CONFIG=config/generated/flash_doe/case_0001__reg_local_csv__random_forest__holdout__random.yaml python main.py
```

## 6) Where outputs go

- Working/intermediate files: `global.base_dir` (case-isolated by default in DOE)
- Run artifacts/logs/metrics: `global.run_dir`
- DOE generation artifacts: `output.dir` (`summary.json`, `manifest.jsonl`, generated YAML files)

## 7) Common quick checks

```bash
# show skip reasons
cat config/generated/flash_doe/manifest.jsonl | grep -E '"status": "skipped"|DOE_'

# show run status (after running a case; use global.run_dir from that case config)
cat config/generated/flash_doe/runs/<doe_spec_hash>/<case_id>/run_status.json
```

## 8) Clean generated DOE outputs (optional)

```bash
rm -rf config/generated/flash_doe
```
