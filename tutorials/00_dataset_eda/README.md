# Tutorial 00: Dataset EDA

This tutorial is the shortest path to "load a dataset and make plots" in CheMLFlow.

It uses a two-node workflow:

- `get_data`
- `analyze.eda`

For the first example, it uses the bundled PGP dataset:

- `local_data/pgp_broccatelli.csv`

## Config

- `configs/pgp_raw_eda.yaml`
- `CheMLFlow_Tutorial_00_Dataset_EDA.ipynb`

## What it makes

For this raw PGP dataset, the generic EDA flow should create:

- dataset overview
- missingness plot
- numeric histograms
- correlation heatmap
- target distribution
- class balance plot

## Run

From the repo root:

```bash
CHEMLFLOW_CONFIG=tutorials/00_dataset_eda/configs/pgp_raw_eda.yaml python main.py
```

Outputs go to:

- `tutorials/00_dataset_eda/artifacts/run/eda`

## Colab

The notebook keeps the config visible in a code cell, writes it to disk, then launches:

```bash
python main.py
```
