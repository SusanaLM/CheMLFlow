# Molecular Analysis Decision and Config Reference

## Decision table

| User intent | Data/source requirement | Action |
|---|---|---|
| Build or run an ordinary model config | Any dataset | Do not add either optional node. Use the config-builder skill. |
| Design or run a DOE/model comparison | Any dataset | Do not add either optional node to the spec or children. Use the DOE/study skills. |
| Audit or rank DOE results | Existing `analysis.py` bundle | Do not launch either node. Use the analysis-curator skill. |
| Inspect molecules and properties interactively/offline | CSV with resolvable SMILES and suitable cohort size | Use a dedicated `analyze.molecular_eda` config. |
| Explore chemical-space maps, unsupervised clusters, neighbours, or activity discontinuities | Compatible molecular CSV; meaningful property required for property-dependent views | Use a dedicated `analyze.molecular_eda` config. |
| Render named molecular publication figures now | Complete molecular EDA bundle | Use `analyze.publication_figures` with `source_dir` and an explicit `figures` list. |
| Build EDA and then selected figures | Compatible molecular CSV and explicit request for both | Run both nodes in one dedicated config, EDA first. |
| Produce parity, residual, ROC/PR, learning-curve, or SHAP plots | Completed model results | Do not use the molecular publication node; use the modelling/reporting path. |

Compatibility never overrides intent. A SMILES column alone is not a trigger.

## Molecular EDA compatibility gate

Confirm all applicable checks before promising execution:

1. The input is a readable CSV.
2. A SMILES column is explicitly configured or resolvable as one of
   `canonical_smiles`, `smiles`, `SMILES`, or `mol_smiles`.
3. The dataset is large enough for the requested methods. Treat 10 molecules as
   the lower configuration boundary and check method-specific constraints such
   as t-SNE perplexity and neighbourhood sizes.
4. The requested property exists. Structure-only EDA may omit a property, but
   property distributions and activity discontinuities may not.
5. A requested numeric property has usable finite values after curation.
6. Raw linear potency has one populated homogeneous units column. Do not combine
   nM, µM, M, or missing units without an explicit upstream standardization.
7. The output directory does not already exist unless the user explicitly chose
   `overwrite: true`.
8. Optional dependencies are available when execution is requested.

Resolve an identifier from `molecule_chembl_id`,
`parent_molecule_chembl_id`, `mol_id`, `compound_id`, `id`, or `name` when
available. Otherwise allow the workflow to generate deterministic in-memory row
identifiers; do not rewrite the source CSV solely to add IDs.

## Dedicated molecular EDA config

Use this shape when the input is already curated and contains pIC50. Adapt paths,
task metadata, thresholds, columns, and methods to the actual dataset.

```yaml
global:
  pipeline_type: molecular_dataset_analysis
  task_type: regression
  base_dir: artifacts/data/molecular_analysis
  run_dir: artifacts/runs/molecular_analysis
  target_column: pIC50
  random_state: 42
  thresholds:
    active: 6.0
    inactive: 5.0

pipeline:
  nodes: [analyze.molecular_eda]

analyze:
  molecular_eda:
    input_path: data/curated.csv
    smiles_column: canonical_smiles
    property_column: pIC50
    property_type: potency_log
    map_methods: [pca, umap]
    primary_map: umap
    embedding:
      random_state: 42
    report:
      advanced: true
      drug_discovery_panel: true
      activity_discontinuities: true
```

When starting from raw ChEMBL activity records, prefer a dedicated upstream
pipeline such as `get_data -> curate -> label.ic50 -> analyze.molecular_eda`, and
select the produced pIC50 column. If analyzing raw IC50 directly is intentional,
use `property_type: potency_linear` plus `units_column` and verify a single unit.

## Publication-only config

Use this only with a completed, checksum-valid molecular EDA bundle. Require the
user to select the figures.

```yaml
global:
  pipeline_type: molecular_publication_figures
  task_type: regression
  base_dir: artifacts/data/molecular_analysis
  run_dir: artifacts/runs/selected_figures
  target_column: pIC50
  random_state: 42
  thresholds:
    active: 6.0
    inactive: 5.0

pipeline:
  nodes: [analyze.publication_figures]

analyze:
  publication_figures:
    source_dir: artifacts/runs/molecular_analysis/molecular_eda
    figures: [chemical_space, qed, property_distribution]
    formats: [pdf, svg]
    on_missing: error
```

Use `on_missing: skip` only when partial output is explicitly acceptable. Report
every skipped requested figure and its reason.

## Combined dedicated config

When both operations were requested, order the nodes as:

```yaml
pipeline:
  nodes:
    - analyze.molecular_eda
    - analyze.publication_figures
```

Omit `publication_figures.source_dir` in this case; the second node consumes the
completed directory placed in the run context by the first node. Still require
`publication_figures.figures`.

## Validation and output review

Before execution, call the same node-order and strict config validators used by
CheMLFlow. Do not validate only by parsing YAML.

After molecular EDA, inspect:

- `run_manifest.json.status`;
- `analysis_identity.curated_input_sha256`;
- `analysis_identity.molecular_eda_config_sha256`;
- `analysis_identity.implementation_version`;
- row counts, invalid-SMILES counts, property cohort counts, warnings, and
  fingerprint-collision diagnostics;
- the artifact-manifest checksum verification result.

After publication rendering, inspect:

- the exact requested, produced, and skipped figure lists;
- the selected property and formats;
- source manifest and artifact-manifest hashes;
- every produced file's size and SHA-256.

Never claim that an HTML report is a self-contained single file when it depends
on its adjacent `eda/` assets.
