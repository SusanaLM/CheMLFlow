# Agent Skills Quickstart

CheMLFlow skills are small operating manuals for agents. They teach an agent how to do
CheMLFlow-specific work consistently: create single configs, review DOE files, audit analysis
outputs, and avoid common manifest, row-count, scaler, and split-balance mistakes.

## 1. What is included

```text
skills/
+-- chemlflow-study-runner/
+-- chemlflow-config-builder/
+-- chemlflow-doe-designer/
+-- chemlflow-analysis-curator/
+-- chemlflow-molecular-analysis/
```

Each skill has:

- `SKILL.md`: concise instructions and trigger description
- optional `references/`: details loaded only when needed
- optional `scripts/`: deterministic helper checks

## 2. Use a skill in a prompt

Ask your agent to use the skill by path:

```text
Use the CheMLFlow Study Runner skill in skills/chemlflow-study-runner to coordinate a local DOE run and audited analysis.
```

```text
Use the CheMLFlow Config Builder skill in skills/chemlflow-config-builder to create one runtime config for a PGP random-forest baseline.
```

```text
Use the CheMLFlow DOE Designer skill in skills/chemlflow-doe-designer to review config/doe_pgp.yaml.
```

```text
Use the CheMLFlow Analysis Curator skill in skills/chemlflow-analysis-curator to audit pah/pah_analysis_6689856.
```

```text
Use the CheMLFlow Molecular Analysis skill in skills/chemlflow-molecular-analysis to inspect this curated SMILES/pIC50 dataset and create a dedicated molecular EDA config.
```

The molecular-analysis skill is opt-in. Do not use it merely because an
ordinary config or DOE dataset contains SMILES, IC50, or pIC50. Invoke it for an
explicit request for molecule inspection, chemical-space projection,
unsupervised clustering, activity-discontinuity analysis, or named molecular
publication figures.

## 3. Run the helper checks

The config-builder skill is currently an operating manual, not a scripted checker.

Summarize generated DOE artifacts:

```bash
python skills/chemlflow-doe-designer/scripts/summarize_doe.py tmp/pgp_hpcc_analysis/pgp_doe
```

Run generated DOE configs locally:

```bash
python scripts/run_doe_local.py --doe-dir config/generated/my_doe --max-workers 1 --resume
```

Analyze local DOE outputs:

```bash
python analysis.py --backend local --doe-dir config/generated/my_doe --output-dir config/generated/my_doe/analysis_local
```

Audit analysis outputs:

```bash
python skills/chemlflow-analysis-curator/scripts/audit_analysis.py pah/pah_analysis_6689856
```

## 4. What the agent should check

For single-config work, the agent should inspect:

- dataset shape: SMILES, tabular features, or non-molecular data
- task type: regression or classification
- curation/drop-row settings
- feature/model compatibility
- split mode, seed, scaler, and output paths
- whether full K-fold CV should be handled by DOE fanout
- Morgan vs RDKit assumptions, and random vs scaffold assumptions

For DOE work, the agent should inspect:

- `summary.json`
- `manifest.jsonl`
- `parent_manifest.jsonl`
- model, feature, scaler, and split compatibility
- valid, skipped, and parent case counts
- local vs Slurm execution backend

For analysis work, the agent should inspect:

- `report.json`
- `all_runs_metrics.csv`
- `all_runs_metrics_by_execution.csv`
- raw vs aggregated row counts
- `scaler`, Morgan/RDKit, model, and split balance
- failed or incomplete folds before discussing model performance
- whether `report.json` says `backend: local` or `backend: slurm`

For optional molecular-analysis work, the agent should check:

- that the user explicitly requested the optional analysis;
- SMILES, identifier, property, and units compatibility;
- that the work uses a dedicated dataset-analysis config rather than DOE
  children;
- that publication figures are explicitly selected;
- molecular EDA and publication manifest status, identity hashes, and artifact
  checksums.

## 5. Optional auto-discovery

Keep `skills/` in this repo as the source of truth. If your agent supports automatic skill
discovery, copy or symlink the skill folders into that agent's personal or project skill
directory.

Example:

```bash
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/chemlflow-doe-designer" ~/.codex/skills/chemlflow-doe-designer
ln -s "$(pwd)/skills/chemlflow-analysis-curator" ~/.codex/skills/chemlflow-analysis-curator
ln -s "$(pwd)/skills/chemlflow-config-builder" ~/.codex/skills/chemlflow-config-builder
ln -s "$(pwd)/skills/chemlflow-study-runner" ~/.codex/skills/chemlflow-study-runner
ln -s "$(pwd)/skills/chemlflow-molecular-analysis" ~/.codex/skills/chemlflow-molecular-analysis
```

After installing or symlinking skills, restart the agent session so it can reload available
skills.
