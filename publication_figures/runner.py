"""Native publication-figure runner with strict provenance and output contracts."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from molecular_landscape import __version__ as molecular_eda_version
from molecular_landscape.io_utils import (
    atomic_output_directory,
    dependency_versions,
    redact_host_paths,
    sha256_file,
    verify_artifact_manifest,
    write_artifact_manifest,
    write_json,
)

from .plot_distribution import render as render_distribution
from .plot_scatter import render as render_scatter


_PACKAGE_DIR = Path(__file__).resolve().parent
FIGURE_SPECS: dict[str, tuple[str, str]] = {
    "chemical_space": ("scatter", "molecular_space_umap.yaml"),
    "qed": ("distribution", "qed_distribution.yaml"),
    "lipinski": ("distribution", "lipinski_distribution.yaml"),
    "property_distribution": ("distribution", "pic50_distribution.yaml"),
    "activity_discontinuities": ("scatter", "activity_cliffs.yaml"),
}
_ALIASES = {
    "pic50": "property_distribution",
    "activity_cliffs": "activity_discontinuities",
}
DEFAULT_FIGURES = [
    "chemical_space",
    "qed",
    "lipinski",
    "property_distribution",
    "activity_discontinuities",
]
_FORMATS = {"pdf", "svg", "png"}


def available_figures() -> list[str]:
    return list(FIGURE_SPECS)


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_source_manifest(exports: Path) -> tuple[Path, dict[str, Any]]:
    path = exports / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Molecular EDA source is missing run_manifest.json: {exports}"
        )
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Molecular EDA source manifest is not complete: {manifest.get('status')!r}"
        )
    required = (exports / "artifact_manifest.csv", exports / "eda" / "molecule_table.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Molecular EDA source is incomplete; missing: {missing}")
    verify_artifact_manifest(exports, exports / "artifact_manifest.csv")
    return path, manifest


def _canonical_requested(figures: list[str] | None) -> list[str]:
    raw = DEFAULT_FIGURES if figures is None else figures
    if not raw:
        raise ValueError("At least one publication figure must be requested.")
    requested = [_ALIASES.get(str(name), str(name)) for name in raw]
    unknown = sorted(set(requested) - set(FIGURE_SPECS))
    if unknown:
        raise ValueError(
            f"Unknown publication figures requested: {unknown}. "
            f"Available: {available_figures()}"
        )
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate publication figures requested: {duplicates}")
    return requested


def _canonical_overrides(
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    canonical: dict[str, dict[str, Any]] = {}
    for raw_name, value in (overrides or {}).items():
        name = _ALIASES.get(str(raw_name), str(raw_name))
        if name not in FIGURE_SPECS:
            raise ValueError(f"Override provided for unknown figure: {raw_name}")
        if name in canonical:
            raise ValueError(f"Duplicate overrides resolve to figure: {name}")
        if not isinstance(value, dict):
            raise ValueError(f"Override for figure '{raw_name}' must be a mapping.")
        canonical[name] = value
    return canonical


def _source_table_path(exports: Path, config: dict[str, Any]) -> Path:
    inputs = config["input"]
    relative = inputs.get("table") or inputs.get("molecule_table")
    return exports / str(relative)


def _adapt_config(
    *,
    name: str,
    config: dict[str, Any],
    exports: Path,
    staging: Path,
    property_column: str | None,
    formats: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    inputs = config.setdefault("input", {})
    inputs["run_dir"] = str(exports)
    config.setdefault("output", {})["stem"] = str(staging / name)
    config["output"]["formats"] = formats
    config["provenance"] = provenance
    if name == "chemical_space":
        if not property_column:
            raise ValueError("chemical_space requires a selected numeric property.")
        inputs["color_column"] = property_column
        config.setdefault("labels", {})["color"] = property_column
    elif name == "property_distribution":
        if not property_column:
            raise ValueError("property_distribution requires a selected numeric property.")
        inputs["property_column"] = property_column
        config.setdefault("labels", {})["property"] = property_column
    elif name == "activity_discontinuities":
        label = f"|Δ {property_column}|" if property_column else "|Δ property|"
        config.setdefault("labels", {})["y"] = label

    table_path = _source_table_path(exports, config)
    if table_path.is_file():
        columns = set(pd.read_csv(table_path, nrows=0).columns)
        class_column = inputs.get("class_column")
        if class_column and class_column not in columns:
            inputs.pop("class_column", None)
            config.pop("classes", None)
    return config


def run_publication_figures(
    *,
    exports_dir: str,
    output_dir: str,
    figures: list[str] | None = None,
    formats: list[str] | None = None,
    property_column: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    on_missing: str = "error",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render explicitly selected figures from one completed molecular EDA bundle."""
    exports = Path(exports_dir).resolve()
    output = Path(output_dir).resolve()
    source_manifest_path, source_manifest = _load_source_manifest(exports)
    requested = _canonical_requested(figures)
    normalized_overrides = _canonical_overrides(overrides)
    selected_formats = [str(value).lower() for value in (formats or ["pdf", "svg", "png"])]
    invalid_formats = sorted(set(selected_formats) - _FORMATS)
    if invalid_formats:
        raise ValueError(f"Unsupported publication figure formats: {invalid_formats}")
    if on_missing not in {"error", "skip"}:
        raise ValueError("on_missing must be 'error' or 'skip'.")
    if property_column is None:
        properties = (
            source_manifest.get("result", {}).get("schema", {}).get("property_cols", [])
        )
        property_column = str(properties[0]) if properties else None

    source_input = source_manifest.get("input", {})
    provenance = {
        "source": "molecular EDA post-curation cohort",
        "version": str(source_manifest.get("workflow_version", molecular_eda_version)),
        "sha256": str(source_input.get("sha256", "")),
        "date": str(source_manifest.get("completed_at", "")),
        "extra": f"EDA manifest {sha256_file(source_manifest_path)[:10]}…",
    }
    started_at = datetime.now(timezone.utc)
    produced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rendered_configs: dict[str, Any] = {}
    with atomic_output_directory(output, overwrite=overwrite) as staging:
        for name in requested:
            renderer_name, template_name = FIGURE_SPECS[name]
            with (_PACKAGE_DIR / template_name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            figure_overrides = normalized_overrides.get(name) or {}
            _deep_update(config, copy.deepcopy(figure_overrides))
            config = _adapt_config(
                name=name,
                config=config,
                exports=exports,
                staging=staging,
                property_column=property_column,
                formats=selected_formats,
                provenance=provenance,
            )
            source_table = _source_table_path(exports, config)
            if not source_table.is_file():
                reason = f"missing required source table: {source_table}"
                if on_missing == "skip":
                    skipped.append({"figure": name, "reason": reason})
                    continue
                raise FileNotFoundError(reason)
            manifest_config = copy.deepcopy(config)
            manifest_config["output"]["stem"] = name
            rendered_configs[name] = manifest_config
            renderer = render_scatter if renderer_name == "scatter" else render_distribution
            try:
                result = renderer(config, config_dir=_PACKAGE_DIR)
            except (FileNotFoundError, ValueError) as exc:
                if on_missing == "skip":
                    skipped.append({"figure": name, "reason": str(exc)})
                    continue
                raise
            files = []
            for extension in selected_formats:
                artifact = staging / f"{name}.{extension}"
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise RuntimeError(
                        f"Figure '{name}' did not produce the required artifact: {artifact}"
                    )
                files.append(
                    {
                        "relative_path": artifact.relative_to(staging).as_posix(),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                )
            produced.append(
                {
                    "figure": name,
                    "renderer": renderer_name,
                    "template": template_name,
                    "source_table": source_table.relative_to(exports).as_posix(),
                    "accounting": {
                        key: value
                        for key, value in result.items()
                        if key not in {"files", "source_table"}
                    },
                    "files": files,
                }
            )

        manifest = {
            "status": "complete",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "implementation_version": molecular_eda_version,
            "output_dir": str(output),
            "source": {
                "molecular_eda_dir": str(exports),
                "run_manifest": str(source_manifest_path),
                "run_manifest_sha256": sha256_file(source_manifest_path),
                "artifact_manifest_sha256": sha256_file(exports / "artifact_manifest.csv"),
                "curated_input_sha256": source_input.get("sha256"),
                "molecular_eda_config_sha256": source_manifest.get(
                    "analysis_identity", {}
                ).get("molecular_eda_config_sha256"),
                "cohort_scope": "post-curation molecular cohort",
            },
            "request": {
                "figures": requested,
                "formats": selected_formats,
                "property_column": property_column,
                "on_missing": on_missing,
                "overrides": normalized_overrides,
            },
            "rendered_configs": rendered_configs,
            "produced": produced,
            "skipped": skipped,
            "n_figures": len(produced),
            "runtime": dependency_versions(),
            "interpretation_notes": [
                "Figures describe a post-curation molecular cohort.",
                "Activity-discontinuity figures are similarity screens, not "
                "matched-molecular-pair analyses.",
                "If activity classes are used to stack the property distribution, "
                "those classes may be derived from the same property axis.",
            ],
            "privacy": {
                "host_paths": "absolute paths reduced to file or directory names",
                "invocation": "not recorded by the native figure runner",
            },
        }
        manifest = redact_host_paths(manifest)
        write_json(staging / "publication_figures_manifest.json", manifest)
        write_artifact_manifest(staging, staging / "artifact_manifest.csv")
    return manifest
