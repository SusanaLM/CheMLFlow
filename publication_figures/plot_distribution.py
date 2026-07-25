#!/usr/bin/env python3
"""Config-driven, counted-data distribution figures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:  # package use
    from . import figutils as fu
except ImportError:  # direct script use
    import figutils as fu  # type: ignore


def _resolve(path: str, config_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()


def _truthy_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - {"", "0", "1", "false", "true", "no", "yes"})
    if invalid:
        raise ValueError(f"Cohort flag contains non-boolean values: {invalid[:10]}")
    return normalized.isin({"1", "true", "yes"})


def render(config: dict[str, Any], *, config_dir: str | Path = ".") -> dict[str, Any]:
    """Render one distribution and return exact row/bin accounting."""
    base = Path(config_dir).resolve()
    inp = config.get("input") or {}
    run_dir = _resolve(str(inp.get("run_dir", ".")), base)
    table_key = inp.get("molecule_table")
    if not table_key:
        raise ValueError("Distribution config requires input.molecule_table.")
    table = run_dir / str(table_key)
    if not table.is_file():
        raise FileNotFoundError(f"Required publication-figure table not found: {table}")
    prop = str(inp.get("property_column") or "").strip()
    if not prop:
        raise ValueError("Distribution config requires input.property_column.")

    df = pd.read_csv(table, low_memory=False)
    required = [prop]
    flag = inp.get("cohort_flag_column")
    class_col = inp.get("class_column")
    required.extend(str(value) for value in (flag, class_col) if value)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Distribution table is missing required columns: {missing}")
    input_rows = len(df)
    if flag:
        df = df[_truthy_mask(df[str(flag)])]
    cohort_rows = len(df)
    numeric = pd.to_numeric(df[prop], errors="coerce")
    df = df.loc[numeric.notna()].copy()
    values = numeric.loc[numeric.notna()].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No finite numeric values are available for '{prop}'.")

    binning = config.get("binning") or {}
    bw = float(binning.get("bin_width", 0.0))
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError("binning.bin_width must be a positive finite number.")
    discrete = bool(binning.get("discrete", False))
    if discrete:
        lo, hi = int(np.floor(values.min())), int(np.ceil(values.max()))
        centers = np.arange(lo, hi + 1, dtype=float)
        edges = np.append(centers - 0.5, centers[-1] + 0.5)
        bw = 1.0
    else:
        lower = np.floor(values.min() / bw) * bw
        upper = np.ceil(values.max() / bw) * bw
        if np.isclose(lower, upper):
            lower -= bw / 2
            upper += bw / 2
        edges = np.arange(lower, upper + bw * 1.01, bw)
        if len(edges) < 2:
            edges = np.array([lower, lower + bw], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])

    labels = config.get("labels") or {}
    layout = config.get("layout") or {}
    horizontal = layout.get("orientation", "vertical") == "horizontal"
    fu.apply_style()
    fig, ax = plt.subplots(figsize=tuple(layout.get("figsize", [6.6, 5.2])))

    def draw(counts, base_counts, color, label):
        if horizontal:
            ax.barh(centers, counts, left=base_counts, height=bw * 0.9,
                    color=color, edgecolor="white", linewidth=0.3, label=label)
        else:
            ax.bar(centers, counts, bottom=base_counts, width=bw * 0.9,
                   color=color, edgecolor="white", linewidth=0.3, label=label)

    class_counts: dict[str, int] = {}
    if class_col:
        classes = config.get("classes") or {}
        observed = df[str(class_col)].fillna("<missing>").astype(str)
        configured = [str(value) for value in classes.get("order", [])]
        order = configured + sorted(set(observed) - set(configured))
        colors = {str(key): value for key, value in classes.get("colors", {}).items()}
        display = {str(key): value for key, value in classes.get("labels", {}).items()}
        base_counts = np.zeros(len(centers))
        for index, category in enumerate(order):
            selected = observed.eq(category).to_numpy()
            counts, _ = np.histogram(values[selected], bins=edges)
            class_counts[category] = int(selected.sum())
            draw(counts, base_counts, colors.get(
                category, fu.PALETTE[index % len(fu.PALETTE)]), display.get(category, category))
            base_counts += counts
        ax.legend(frameon=True, edgecolor="#CCCCCC", framealpha=0.9)
    else:
        counts, _ = np.histogram(values, bins=edges)
        draw(counts, np.zeros(len(centers)), layout.get("color", fu.PALETTE[0]), None)

    property_label = labels.get("property", prop)
    count_label = labels.get("count", "Count")
    if horizontal:
        ax.set_ylabel(property_label)
        ax.set_xlabel(count_label)
        ax.grid(axis="x", color=fu.GRID, alpha=0.35, lw=0.6)
    else:
        ax.set_xlabel(property_label)
        ax.set_ylabel(count_label)
        ax.grid(axis="y", color=fu.GRID, alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    fu.despine(ax)
    if discrete:
        (ax.set_yticks if horizontal else ax.set_xticks)(centers)
    if labels.get("title"):
        ax.set_title(labels["title"])

    provenance = config.get("provenance") or {}
    if provenance:
        fu.provenance_footer(fig, **provenance)
    output = config.get("output") or {}
    stem = _resolve(str(output.get("stem") or "figure"), base)
    formats = tuple(str(value).lower() for value in output.get("formats", ["pdf", "svg", "png"]))
    fig.tight_layout(rect=(0, 0.035 if provenance else 0, 1, 1))
    fu.save_vector(fig, str(stem), formats=formats)
    return {
        "input_rows": int(input_rows),
        "cohort_rows": int(cohort_rows),
        "rows_plotted": int(len(values)),
        "rows_dropped_non_numeric": int(cohort_rows - len(values)),
        "bins": int(len(centers)),
        "class_counts": class_counts,
        "source_table": str(table),
        "files": [str(stem.with_suffix(f".{extension}")) for extension in formats],
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python plot_distribution.py CONFIG.yaml")
    config_path = Path(os.path.abspath(sys.argv[1]))
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = render(config, config_dir=config_path.parent)
    print(f"rows plotted: {result['rows_plotted']}  bins: {result['bins']}")


if __name__ == "__main__":
    main()
