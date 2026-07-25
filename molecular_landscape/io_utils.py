"""Reproducible and atomic workflow IO."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import tempfile
import unicodedata
import uuid
from csv import DictWriter
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename_token(value: str, max_length: int = 80) -> str:
    """Return a readable, deterministic, path-safe token without silent collisions."""
    raw = str(value)
    ascii_value = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_value).strip("._-")
    token = token[:max_length].rstrip("._-") or "unnamed"
    if token != raw or len(raw) > max_length:
        token = f"{token[: max_length - 9]}_{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
    return token


def public_manifest_path(value: str | Path) -> str:
    """Remove host-specific parents while retaining a useful file/directory name."""
    text = str(value)
    native = Path(text)
    if native.is_absolute():
        return native.name or "<filesystem-root>"
    windows = PureWindowsPath(text)
    if windows.is_absolute():
        return windows.name or "<filesystem-root>"
    return text


def redact_host_paths(payload: Any) -> Any:
    """Return a JSON-friendly copy with absolute filesystem paths reduced to names."""
    if isinstance(payload, dict):
        return {key: redact_host_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [redact_host_paths(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(redact_host_paths(value) for value in payload)
    if isinstance(payload, Path):
        return public_manifest_path(payload)
    if isinstance(payload, str):
        return public_manifest_path(payload)
    return payload


def public_invocation(argv: Sequence[str | Path]) -> str:
    """Identify the executable/entrypoint while omitting command arguments."""
    if not argv:
        return ""
    public = [public_manifest_path(argv[0])]
    if len(argv) > 1:
        public.append(public_manifest_path(argv[1]))
    if len(argv) > 2:
        public.append("arguments-redacted")
    return shlex.join(public)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )


def dependency_versions() -> dict:
    names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "plotly",
        "rdkit",
        "umap-learn",
    ]
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def configure_runtime_caches(output_dir: Path) -> Path:
    """Place writable library caches beside, but outside, the atomic output."""
    cache_root = output_dir.parent / f".{output_dir.name}.runtime-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    return cache_root


def write_artifact_manifest(root: Path, path: Path) -> None:
    """Write checksums for every artifact except the manifest itself."""
    rows = []
    excluded = path.resolve()
    for artifact in sorted(root.rglob("*")):
        if not artifact.is_file() or artifact.resolve() == excluded:
            continue
        rows.append(
            {
                "relative_path": artifact.relative_to(root).as_posix(),
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_artifact_manifest(root: Path, path: Path) -> dict[str, int]:
    """Verify every listed artifact and reject missing, changed, or unsafe paths."""
    root = root.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Artifact manifest is missing columns: {sorted(required - set(frame.columns))}"
        )
    checked = 0
    for row in frame.to_dict(orient="records"):
        relative = Path(str(row["relative_path"]))
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Artifact manifest path escapes its root: {relative}") from exc
        if not artifact.is_file():
            raise FileNotFoundError(f"Manifest-listed artifact is missing: {artifact}")
        expected_size = int(row["size_bytes"])
        if artifact.stat().st_size != expected_size:
            raise ValueError(
                f"Artifact size mismatch for {relative}: expected {expected_size}, "
                f"got {artifact.stat().st_size}"
            )
        expected_hash = str(row["sha256"])
        actual_hash = sha256_file(artifact)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Artifact SHA-256 mismatch for {relative}: expected {expected_hash}, "
                f"got {actual_hash}"
            )
        checked += 1
    return {"artifacts_verified": checked}


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def atomic_output_directory(final_path: Path, overwrite: bool) -> Iterator[Path]:
    final_path = final_path.resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {final_path}. Pass --overwrite "
            "to replace it."
        )
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.staging-", dir=final_path.parent)
    )
    backup_path: Optional[Path] = None
    try:
        yield temp_path
        if final_path.exists():
            backup_path = final_path.parent / (
                f".{final_path.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(final_path, backup_path)
        try:
            os.replace(temp_path, final_path)
        except Exception:
            if backup_path is not None and not final_path.exists():
                os.replace(backup_path, final_path)
                backup_path = None
            raise
        if backup_path is not None:
            _remove_path(backup_path)
    except Exception:
        _remove_path(temp_path)
        raise
