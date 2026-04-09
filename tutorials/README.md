# Tutorials

This folder is the low-friction onboarding path for CheMLFlow.

The goal is simple:

- start with one runnable config
- move to DOE generation once the config shape makes sense
- move to phase-2 style tuning only after the DOE workflow is clear

## Tutorial Roadmap

1. `01_single_config_colab`
   Run one PGP classification config in a Colab-style workflow.
   This tutorial uses:
   - the bundled `local_data/pgp_broccatelli.csv` dataset
   - Morgan fingerprints
   - `preprocess.scaler: standard`
   - `split.mode: cv` with `n_splits: 5`
   - a single runnable slice: `fold_index: 0`, `repeat_index: 0`

2. `02_submit_doe`
   Planned. Generate a DOE from one dataset profile and inspect the manifest/summary outputs.

3. `03_submit_phase2`
   Planned. Launch a focused phase-2 tuning DOE from a parent winner.

## Important CV Note

CheMLFlow treats `split.mode: cv` as one fold per run.

That means tutorial 1 is still a "single config" tutorial, but the config is one execution slice from a 5-fold CV design. The full fold fanout belongs in the DOE tutorial.
