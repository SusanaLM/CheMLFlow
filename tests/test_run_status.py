from __future__ import annotations

import json
from pathlib import Path

import pytest

from main import NODE_REGISTRY, run_configured_pipeline_nodes


def _base_config(tmp_path: Path) -> dict:
    return {
        "global": {
            "pipeline_type": "doe_status_test",
            "task_type": "regression",
            "base_dir": str(tmp_path / "data"),
            "run_dir": str(tmp_path / "run"),
            "target_column": "target",
            "thresholds": {"active": 1, "inactive": 2},
            "runs": {"enabled": True, "id": "status_test"},
        },
        "pipeline": {"nodes": ["get_data"]},
        "get_data": {
            "data_source": "local_csv",
            "source": {"path": str(tmp_path / "missing.csv")},
        },
    }


def test_run_status_written_on_node_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)

    def _boom(_context: dict) -> None:
        raise RuntimeError("node exploded")

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _boom)

    with pytest.raises(RuntimeError, match="node exploded"):
        run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    run_dir = Path(config["global"]["run_dir"])
    run_config_path = run_dir / "run_config.yaml"
    status_path = run_dir / "run_status.json"
    assert run_config_path.exists()
    assert status_path.exists()

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["failed_node"] == "get_data"
    assert status["exception_type"] == "RuntimeError"
    assert "node exploded" in status["message"]
    assert status.get("traceback")


def test_run_status_written_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)

    def _ok(_context: dict) -> None:
        return None

    monkeypatch.setitem(NODE_REGISTRY, "get_data", _ok)

    assert run_configured_pipeline_nodes(config, str(tmp_path / "config.yaml"))

    run_dir = Path(config["global"]["run_dir"])
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert "failed_node" not in status
