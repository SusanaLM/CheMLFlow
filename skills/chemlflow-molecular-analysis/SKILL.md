---
name: chemlflow-molecular-analysis
description: Create, review, run, or diagnose opt-in CheMLFlow dataset-analysis configs using `analyze.molecular_eda` and `analyze.publication_figures`. Use only when the user explicitly asks for interactive or offline molecular EDA, molecule inspection, chemical-space projection, unsupervised clustering, nearest-neighbour or activity-discontinuity analysis, or selected publication-quality molecular figures from a compatible SMILES CSV or completed molecular EDA bundle. Do not use for ordinary runtime configs, default end-to-end studies, DOE/model-search children, model training, generic `analyze.eda`, `analysis.py` result auditing, or model-performance/SHAP figures unless the user explicitly requests these optional molecular-analysis nodes.
---

# CheMLFlow Molecular Analysis

## Purpose

Use CheMLFlow's two optional, dataset-scoped analysis nodes without changing the
default modelling workflow:

- `analyze.molecular_eda` builds the molecular landscape and offline report.
- `analyze.publication_figures` renders only explicitly selected figures from a
  completed molecular EDA bundle.

Treat both as opt-in. Dataset compatibility permits their use; it does not by
itself authorize adding them to a config or running them.

## Hard Default Boundary

- Do not add either node to an ordinary single-model config, DOE spec, generated
  DOE child, benchmark, training workflow, or analysis audit by default.
- Do not infer molecular EDA merely because a dataset contains SMILES, IC50, or
  pIC50 columns.
- Do not replace or modify the existing generic `analyze.eda` node.
- Do not make molecular EDA or publication rendering a DOE search-space axis.
- Do not render figures for every fold, seed, or model child.
- Use a dedicated dataset-analysis config. A publication-only config may point
  to an already completed molecular EDA bundle.
- Require an explicit figure list. Never interpret the CheMLFlow node as
  "render every available figure."

When the request is an ordinary config, DOE, study, or result audit, continue
with the corresponding CheMLFlow skill and leave these nodes absent.

## Decision Workflow

1. Confirm explicit intent. Look for a direct request to inspect molecules,
   explore molecular properties, project or cluster chemical space, find
   neighbours/activity discontinuities, open the molecular report, or render
   named publication figures.
2. Select the operation:
   - Use `analyze.molecular_eda` for dataset-level molecular exploration.
   - Use `analyze.publication_figures` for named figures from a complete bundle.
   - Use both, in that order, only when both outputs are explicitly requested.
3. Apply the compatibility gate in `references/decision-and-config.md`.
4. Inspect the input and state the resolved SMILES, identifier, property, and
   units columns. Do not guess ambiguous activity semantics.
5. Create a separate config and keep its `global.base_dir`, `global.run_dir`, and
   output directories away from existing DOE children and reports.
6. If execution is requested, preflight the optional dependencies and validate
   the config before running it.
7. Verify the completed manifests and required artifacts before reporting
   success.

## Scientific Guardrails

- Describe the analyzed population as the post-curation molecular cohort.
- Do not claim the node can recover replicate assay rows removed by upstream
  curation.
- Treat pIC50, pKi, and pChEMBL-like values as logarithmic potency only when the
  selected property actually has that meaning.
- For raw IC50, Ki, EC50, or related linear activity values, require
  `property_type: potency_linear` and one populated homogeneous units column.
  Prefer upstream `label.ic50` and analysis of the resulting pIC50 when that is
  the intended CheMLFlow workflow.
- Never relabel raw linear activity as `potency_log` and never silently convert
  mixed or missing units.
- Call activity-discontinuity results similarity screens, not
  matched-molecular-pair analyses.
- Preserve folded-fingerprint collision flags and do not present
  collision-derived zero-distance pairs as default activity discontinuities.
- Treat PCA/UMAP/t-SNE/PaCMAP/TriMap as exploratory projections; do not infer
  causal or mechanistic relationships from two-dimensional proximity.

## Publication-Figure Boundary

Use `analyze.publication_figures` only for molecular EDA exports. It does not
produce model parity, residual, learning-curve, ROC/PR, or SHAP figures.

After a DOE audit, a user may choose a dataset or result for reporting. That
selection may trigger a separate publication-only config, but ranking a model
does not automatically trigger this skill or the figure node.

Supported selections are:

- `chemical_space`
- `qed`
- `lipinski`
- `property_distribution` (`pic50` alias)
- `activity_discontinuities` (`activity_cliffs` alias)

## Execution and Verification

For a requested local run, preflight the optional stack in the active
environment:

```bash
python -c "import rdkit, umap, plotly; print('molecular EDA stack ok', rdkit.__version__, umap.__version__, plotly.__version__)"
```

Install the optional extra only when the user authorizes installation:

```bash
python -m pip install -e '.[molecular_eda]'
```

Validate node order and strict config semantics before execution. Afterward,
require:

- molecular EDA: `run_manifest.json` with `status: complete`,
  `artifact_manifest.csv`, `eda_report.html`, `eda/molecule_table.csv`, and the
  expected analysis tables;
- publication figures: `publication_figures_manifest.json` with
  `status: complete`, `artifact_manifest.csv`, and every requested non-skipped
  output;
- checksum verification through the native artifact-manifest validation.

Do not describe a dependency import as a completed analysis proof.

## Dataset-Level Reuse

Do not enable shared caching automatically. If reuse is explicitly requested,
key it by all three persisted identity fields:

- curated input SHA-256;
- molecular EDA semantic-config SHA-256;
- molecular EDA implementation version.

Reject reuse when any identity component differs.

## References

- Read `references/decision-and-config.md` for eligibility rules and canonical
  dedicated-config shapes.
- Read `docs/molecular-analysis.md` for the user-facing node contract and
  interpretation notes.
- Read `docs/config-options.md` and inspect
  `utilities/config_validation.py` when exact runtime validation behavior is
  needed.
- Inspect `tests/test_molecular_analysis_nodes.py` and
  `tests/test_publication_figure_runner.py` for executable integration examples.
