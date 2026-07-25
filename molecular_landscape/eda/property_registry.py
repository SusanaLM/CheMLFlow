"""Semantic inference for selected molecular properties."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .config import PROPERTY_TYPES


@dataclass(frozen=True)
class PropertyProfile:
    column: str
    semantic_type: str
    display_name: str
    units: str | None
    higher_is_better: bool | None
    is_potency: bool
    is_log_scaled: bool
    suggested_active_threshold: float | None
    suggested_high_threshold: float | None
    interpretation_label_high: str
    interpretation_label_low: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _infer_units(metadata: pd.DataFrame | None) -> str | None:
    if metadata is None:
        return None
    # Prefer standardized unit fields. A populated authoritative field must not
    # be overridden by a mixed or stale legacy `units` column.
    for column in ("standard_units", "qudt_units", "units"):
        if column not in metadata.columns:
            continue
        values = metadata[column].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        if values.nunique() == 1:
            return str(values.iloc[0])
        if values.nunique() > 1:
            return "mixed"
    return None


def _auto_type(column: str, values: pd.Series, units: str | None) -> str:
    name = _normalized(column)
    if name in {"pchemblvalue", "pic50", "pki", "pec50", "pkd", "pactivity"}:
        return "potency_log"
    if name in {"ic50", "ki", "ec50", "gi50", "kd"}:
        return "potency_linear"
    if any(token in name for token in ("homolumogap", "bandgap")) or name == "gap":
        return "qm_gap"
    if any(token in name for token in ("homo", "lumo", "energy", "enthalpy")):
        return "qm_energy"
    if any(
        token in name
        for token in (
            "solubility",
            "logs",
            "logd",
            "logp",
            "permeability",
            "tpsa",
            "molwt",
        )
    ):
        return "physchem"
    if any(
        token in name
        for token in ("clearance", "tox", "admet", "bioavailability", "half life")
    ):
        return "admet"
    numeric = pd.to_numeric(values, errors="coerce")
    nonmissing = values.dropna()
    if len(nonmissing) and numeric.notna().sum() == len(nonmissing):
        return "generic_numeric"
    if 1 < nonmissing.nunique() <= max(10, int(len(nonmissing) * 0.1)):
        return "classification"
    return "generic_categorical"


def infer_property_profile(
    column: str,
    values: pd.Series,
    metadata: pd.DataFrame | None = None,
    requested_type: str = "auto",
    higher_is_better: str | bool = "auto",
) -> PropertyProfile:
    """Infer cautious semantics used by report language and thresholds."""
    if requested_type not in PROPERTY_TYPES:
        raise ValueError(f"Unsupported property type: {requested_type}")
    if higher_is_better not in {"auto", True, False}:
        raise ValueError("higher_is_better must be 'auto', true, or false.")
    units = _infer_units(metadata)
    semantic_type = (
        _auto_type(column, values, units)
        if requested_type == "auto"
        else requested_type
    )
    if semantic_type == "potency_log":
        units = None
    is_potency = semantic_type in {"potency_log", "potency_linear"}
    is_log_scaled = semantic_type == "potency_log"
    inferred_direction: bool | None
    if semantic_type == "potency_log":
        inferred_direction = True
    elif semantic_type == "potency_linear":
        inferred_direction = False
    elif semantic_type in {"classification", "generic_categorical"}:
        inferred_direction = None
    else:
        inferred_direction = None
    direction = (
        inferred_direction if higher_is_better == "auto" else bool(higher_is_better)
    )
    if semantic_type == "potency_log":
        high_label, low_label = "higher potency", "lower potency"
        active_threshold, high_threshold = 6.0, 8.0
    elif semantic_type == "potency_linear":
        high_label, low_label = "weaker potency", "stronger potency"
        active_threshold, high_threshold = None, None
    else:
        high_label, low_label = "high property", "low property"
        active_threshold, high_threshold = None, None
    return PropertyProfile(
        column=column,
        semantic_type=semantic_type,
        display_name=column.replace("_", " ").strip().title(),
        units=units,
        higher_is_better=direction,
        is_potency=is_potency,
        is_log_scaled=is_log_scaled,
        suggested_active_threshold=active_threshold,
        suggested_high_threshold=high_threshold,
        interpretation_label_high=high_label,
        interpretation_label_low=low_label,
    )
