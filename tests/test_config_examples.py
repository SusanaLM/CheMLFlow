from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from main import validate_pipeline_nodes


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes(config: dict) -> list[str]:
    nodes = ((config.get("pipeline") or {}).get("nodes")) or []
    if not isinstance(nodes, list):
        return []
    return [str(n) for n in nodes]


@pytest.mark.parametrize(
    "config_path",
    sorted((REPO_ROOT / "config").glob("config.*.yaml")),
)
def test_example_configs_have_valid_pipeline_split_rules(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    nodes = _load_nodes(config)
    if not nodes:
        pytest.skip(f"No pipeline nodes in {config_path}")
    validate_pipeline_nodes(nodes)


def test_validate_pipeline_nodes_requires_split_before_training() -> None:
    with pytest.raises(ValueError):
        validate_pipeline_nodes(["train"])
    with pytest.raises(ValueError):
        validate_pipeline_nodes(["train", "split"])
    with pytest.raises(ValueError):
        validate_pipeline_nodes(["curate", "split", "train", "split", "explain"])
    with pytest.raises(ValueError):
        validate_pipeline_nodes(["split", "split"])


def test_pgp_chemprop_example_config_is_valid_yaml() -> None:
    config_path = REPO_ROOT / "config" / "config.pgp_chemprop.yaml"
    assert config_path.exists(), f"Missing example config: {config_path}"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)

    global_cfg = config.get("global") or {}
    assert global_cfg.get("task_type") == "classification"
    assert global_cfg.get("target_column"), "Expected global.target_column to be set."

    model_cfg = config.get("model") or {}
    assert model_cfg.get("type") == "chemprop"

    nodes = _load_nodes(config)
    assert "split" in nodes, "Chemprop training requires split_indices; include the split node."
    assert "train" in nodes
    assert nodes.index("split") < nodes.index("train")
