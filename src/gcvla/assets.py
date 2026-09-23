"""Pinned public benchmark-asset provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> dict[str, int | str]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = 0
    bytes_total = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(f"{file_sha256(child)}  {size}  ./{relative}\\n".encode())
        files += 1
        bytes_total += size
    if not files:
        raise ValueError(f"Empty benchmark asset directory: {path}")
    return {"sha256": digest.hexdigest(), "files": files, "bytes": bytes_total}


def write_asset_lock(config_path: Path, lock_path: Path, *, benchmark: str, revision: str) -> None:
    config = yaml.safe_load(config_path.read_text())
    required = {"assets", "bddl_files", "init_states"}
    if not required <= set(config):
        raise ValueError("Benchmark config is missing asset paths")
    paths = {name: Path(config[name]).resolve() for name in sorted(required)}
    lock = {
        "schema_version": 1,
        "benchmark": benchmark,
        "benchmark_revision": revision,
        "config_sha256": file_sha256(config_path),
        "paths": {name: tree_sha256(path) for name, path in paths.items()},
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def verify_asset_lock(config_path: Path, lock_path: Path, *, benchmark: str, revision: str) -> str:
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing benchmark asset lock: {lock_path}")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != 1 or lock.get("benchmark") != benchmark:
        raise ValueError("Benchmark asset lock does not match selected benchmark")
    if lock.get("benchmark_revision") != revision:
        raise ValueError("Benchmark checkout revision does not match asset lock")
    config = yaml.safe_load(config_path.read_text())
    if lock.get("config_sha256") != file_sha256(config_path):
        raise ValueError("Benchmark configuration changed after asset lock creation")
    for name, expected in lock["paths"].items():
        actual = tree_sha256(Path(config[name]).resolve())
        if actual != expected:
            raise ValueError(f"Benchmark asset digest mismatch: {name}")
    return file_sha256(lock_path)
