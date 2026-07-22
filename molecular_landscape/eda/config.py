"""Configuration for the optional interactive EDA report."""

from __future__ import annotations

from dataclasses import dataclass


PROPERTY_TYPES = {
    "auto",
    "potency_log",
    "potency_linear",
    "physchem",
    "admet",
    "qm_energy",
    "qm_gap",
    "classification",
    "generic_numeric",
    "generic_categorical",
}


@dataclass(frozen=True)
class EDAConfig:
    enabled: bool = False
    advanced: bool = False
    open_report: bool = False
    property_type: str = "auto"
    higher_is_better: str | bool = "auto"
    include_drug_discovery_panel: bool = False
    include_model_readiness: bool = True
    include_nearest_neighbors: bool = True
    include_activity_cliffs: bool = True
    include_property_descriptor_plots: bool = True
    top_scaffolds: int = 20
    singleton_scaffold_warning_fraction: float = 0.30
    representative_molecules: int = 48
    max_svg_molecules: int = 5000
    nearest_neighbors: int = 10
    activity_cliff_similarity: float = 0.70
    activity_cliff_delta: float = 1.0
    map_method: str = "umap"
    qed_low_threshold: float = 0.35
    lipinski_violation_warning_threshold: int = 2
    max_points_for_svg_hover: int = 5000
    use_scattergl: bool = True
    export_selected_template: bool = False

    def validate(self) -> None:
        if self.open_report and not self.enabled:
            raise ValueError("Opening the EDA report requires EDA reporting to be enabled.")
        if self.advanced and not self.enabled:
            raise ValueError("Advanced EDA requires EDA reporting to be enabled.")
        if self.include_drug_discovery_panel and not self.advanced:
            raise ValueError("The drug-discovery panel requires advanced EDA.")
        if self.property_type not in PROPERTY_TYPES:
            raise ValueError(f"Unsupported EDA property type: {self.property_type}")
        if self.higher_is_better not in {"auto", True, False}:
            raise ValueError("EDA higher_is_better must be 'auto', true, or false.")
        for name, value in (
            ("top_scaffolds", self.top_scaffolds),
            ("representative_molecules", self.representative_molecules),
            ("max_svg_molecules", self.max_svg_molecules),
            ("nearest_neighbors", self.nearest_neighbors),
            ("max_points_for_svg_hover", self.max_points_for_svg_hover),
        ):
            if value < 1:
                raise ValueError(f"EDA {name} must be at least 1.")
        if not 0.0 <= self.activity_cliff_similarity <= 1.0:
            raise ValueError(
                "Activity-cliff similarity threshold must be between 0 and 1."
            )
        if self.activity_cliff_delta < 0.0:
            raise ValueError("Activity-cliff property difference must be non-negative.")
        for fraction_name, fraction_value in (
            ("singleton_scaffold_warning_fraction", self.singleton_scaffold_warning_fraction),
            ("qed_low_threshold", self.qed_low_threshold),
        ):
            if not 0.0 <= fraction_value <= 1.0:
                raise ValueError(f"EDA {fraction_name} must be between 0 and 1.")
        if self.lipinski_violation_warning_threshold < 0:
            raise ValueError(
                "EDA Lipinski violation warning threshold must be non-negative."
            )
        if self.map_method not in {"pca", "umap", "tsne", "pacmap", "trimap"}:
            raise ValueError(
                "EDA map method must be 'pca', 'umap', 'tsne', 'pacmap', or 'trimap'."
            )
