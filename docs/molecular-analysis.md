# Optional Molecular Dataset Analysis

CheMLFlow provides two opt-in, dataset-scoped analysis nodes:

- `analyze.molecular_eda` creates a molecular landscape analysis and offline
  interactive report bundle.
- `analyze.publication_figures` renders only the explicitly selected figures
  from a completed molecular EDA bundle.

These nodes are independent of the existing `analyze.eda` node. Adding them
does not change old configs, old output paths, or existing generic EDA reports.

Install their optional runtime dependencies only when needed:

```bash
pip install -e '.[molecular_eda]'
```

They are intentionally not part of CheMLFlow's core dependency set, so a fresh
installation for a legacy DOE does not resolve or upgrade scientific packages
solely because these optional nodes exist.

## Recommended execution model

Run these nodes in a dedicated dataset-analysis config. Do not add them to every
model/fold/seed child in a DOE. Molecular EDA is dataset-scoped, whereas model
DOE children are execution-scoped; repeating the analysis in every child wastes
runtime and storage without adding evidence.

Publication figures are also selection-scoped. Add the figure node only to a
dedicated analysis run or point it at the completed molecular EDA bundle chosen
for reporting. `figures` is required and has no implicit "render everything"
behavior in a CheMLFlow node.

CheMLFlow does not automatically decide which DOE model is "high performing"
and does not render these figures for every DOE child. After model-result audit,
the user selects the run or dataset bundle worth reporting and launches a small
publication-only config. The first integration covers figures backed by the
molecular EDA tidy exports; model parity, residual, and SHAP figure selection
remain part of the existing modelling/reporting workflow.

Future shared reuse can use the identity already persisted in
`molecular_eda/run_manifest.json`:

- curated input SHA-256;
- molecular EDA semantic-config SHA-256;
- molecular EDA implementation version.

No shared cache is enabled in the first integration.

## Example

```yaml
pipeline:
  nodes:
    - get_data
    - curate
    - label.ic50
    - analyze.molecular_eda
    - analyze.publication_figures

analyze:
  molecular_eda:
    property_column: pIC50
    property_type: potency_log
    map_methods: [pca, umap]
    primary_map: umap
    report:
      advanced: true
      drug_discovery_panel: true
      activity_discontinuities: true

  publication_figures:
    figures: [chemical_space, qed, property_distribution]
    formats: [pdf, svg, png]
    on_missing: error
```

The output directories are siblings under the CheMLFlow run directory:

```text
<run_dir>/
├── molecular_eda/
│   ├── run_manifest.json
│   ├── artifact_manifest.csv
│   ├── eda_report.html
│   └── eda/
└── publication_figures/
    ├── publication_figures_manifest.json
    ├── artifact_manifest.csv
    └── <selected figures only>
```

The HTML file and its `eda/` directory form an offline report bundle. The HTML
must not be moved away from its molecular SVG directory and described as a
standalone single-file report.

## Supported publication figures

- `chemical_space`
- `qed`
- `lipinski`
- `property_distribution` (`pic50` is accepted as an alias)
- `activity_discontinuities` (`activity_cliffs` is accepted as an alias)

The activity-discontinuity output is a similarity-and-property-difference
screen, not a matched-molecular-pair analysis. Folded-fingerprint collision
pairs are flagged in the neighbour tables and excluded from the default
activity-discontinuity publication table.

## Property and cohort contract

The report describes the post-curation molecular cohort supplied by the
pipeline. It cannot recover replicate assay rows removed upstream by curation.

`pIC50`, `pKi`, and pChEMBL-like properties are recognized as logarithmic
potency values. Raw `IC50`, `Ki`, `EC50`, and related linear potency columns
require `property_type: potency_linear` and one homogeneous units column. The
node never guesses units or silently converts raw potency values.

If no identifier column is available, the workflow creates a deterministic
row identifier in memory. It never rewrites the input CSV.

## Running publication figures later

A publication-only config can point to an already completed bundle:

```yaml
pipeline:
  nodes: [analyze.publication_figures]

analyze:
  publication_figures:
    source_dir: /path/to/selected/run/molecular_eda
    figures: [chemical_space, activity_discontinuities]
    formats: [pdf, svg]
```

The source must contain a complete `run_manifest.json`, an artifact manifest,
and the molecular EDA tidy exports. Every rendered file receives an embedded
provenance footer and is recorded with size and SHA-256 in the publication
manifest.

## Privacy and generated outputs

Generated molecular-analysis roots are ignored by Git when they use the
documented `artifacts/`, `outputs/`, or root-level `molecular_eda/` locations.
The root `publication_figures/` directory is the source package and is
therefore intentionally not ignored. Manifests retain filenames,
configuration, checksums, versions, and scientific provenance, but reduce
absolute host paths to their final file or directory names. Persisted
invocations retain only executable and entrypoint names; command arguments are
omitted.

The offline HTML report deliberately embeds the molecular table needed for
interactive inspection, including identifiers, SMILES, selected properties,
clusters, and neighbours. Treat the report bundle as dataset-bearing output:
review it before sharing and do not commit it merely because host paths have
been redacted.
