#!/usr/bin/env python3
"""Config-driven scatter figures over validated molecular EDA exports."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from . import figutils as fu
    from .plot_distribution import _truthy_mask
except ImportError:
    import figutils as fu  # type: ignore
    from plot_distribution import _truthy_mask  # type: ignore


def _resolve(path: str, config_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()


def render(config: dict[str, Any], *, config_dir: str | Path = ".") -> dict[str, Any]:
    base = Path(config_dir).resolve()
    inp = config.get("input") or {}
    run_dir = _resolve(str(inp.get("run_dir", ".")), base)
    table_key = inp.get("table")
    if not table_key:
        raise ValueError("Scatter config requires input.table.")
    table = run_dir / str(table_key)
    if not table.is_file():
        raise FileNotFoundError(f"Required publication-figure table not found: {table}")
    xcol, ycol = str(inp.get("x_column") or ""), str(inp.get("y_column") or "")
    if not xcol or not ycol:
        raise ValueError("Scatter config requires input.x_column and input.y_column.")
    color_col = inp.get("color_column")
    flag = inp.get("cohort_flag_column")

    df = pd.read_csv(table, low_memory=False)
    required = [xcol, ycol, *[str(value) for value in (color_col, flag) if value]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Scatter table is missing required columns: {missing}")
    input_rows = len(df)
    if flag:
        df = df[_truthy_mask(df[str(flag)])]
    cohort_rows = len(df)
    df = df.copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])
    if df.empty:
        raise ValueError(f"No finite x/y values are available for '{xcol}' and '{ycol}'.")

    labels = config.get("labels") or {}
    layout = config.get("layout") or {}
    point_size = float(layout.get("point_size", 14))
    alpha = float(layout.get("alpha", 0.8))
    fu.apply_style()
    fig, ax = plt.subplots(figsize=tuple(layout.get("figsize", [6.8, 6.0])))
    color_type = str(inp.get("color_type", "continuous"))
    category_counts: dict[str, int] = {}
    if color_col and color_type == "continuous":
        colors = pd.to_numeric(df[str(color_col)], errors="coerce")
        if inp.get("drop_missing_color", True):
            keep = colors.notna()
            df, colors = df.loc[keep], colors.loc[keep]
        if df.empty:
            raise ValueError(f"No finite color values are available for '{color_col}'.")
        colormap = layout.get("colormap", "viridis")
        if isinstance(colormap, (list, tuple)):
            from matplotlib.colors import LinearSegmentedColormap
            colormap = LinearSegmentedColormap.from_list("custom", list(colormap))
        scatter = ax.scatter(df[xcol], df[ycol], c=colors, cmap=colormap,
                             s=point_size, alpha=alpha, linewidths=0, rasterized=True)
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.015, fraction=0.046)
        colorbar.set_label(labels.get("color", color_col))
        colorbar.outline.set_linewidth(0.6)
    elif color_col and color_type == "categorical":
        classes = config.get("classes") or {}
        observed = df[str(color_col)].fillna("<missing>").astype(str)
        configured = [str(value) for value in classes.get("order", [])]
        order = configured + sorted(set(observed) - set(configured))
        colors = {str(key): value for key, value in classes.get("colors", {}).items()}
        display = {str(key): value for key, value in classes.get("labels", {}).items()}
        for index, category in enumerate(order):
            selected = observed.eq(category)
            subset = df.loc[selected]
            category_counts[category] = int(len(subset))
            if subset.empty:
                continue
            ax.scatter(subset[xcol], subset[ycol], s=point_size, alpha=alpha,
                       linewidths=0, color=colors.get(
                           category, fu.PALETTE[index % len(fu.PALETTE)]
                       ),
                       label=display.get(category, category), rasterized=True)
        ax.legend(title=labels.get("color"), frameon=True, edgecolor="#CCCCCC", framealpha=0.9)
    else:
        ax.scatter(df[xcol], df[ycol], s=point_size, alpha=alpha, linewidths=0,
                   color=layout.get("color", fu.PALETTE[0]), rasterized=True)

    ax.set_xlabel(labels.get("x", xcol))
    ax.set_ylabel(labels.get("y", ycol))
    if labels.get("title"):
        ax.set_title(labels["title"])
    if layout.get("hide_ticks"):
        ax.set_xticks([])
        ax.set_yticks([])
    if layout.get("grid", True):
        ax.grid(color=fu.GRID, alpha=0.30, lw=0.6)
        ax.set_axisbelow(True)
    fu.despine(ax, keep=("left", "bottom"))
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
        "rows_plotted": int(len(df)),
        "rows_dropped": int(cohort_rows - len(df)),
        "category_counts": category_counts,
        "source_table": str(table),
        "files": [str(stem.with_suffix(f".{extension}")) for extension in formats],
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python plot_scatter.py CONFIG.yaml")
    config_path = Path(os.path.abspath(sys.argv[1]))
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = render(config, config_dir=config_path.parent)
    print(f"points plotted: {result['rows_plotted']}")


if __name__ == "__main__":
    main()
