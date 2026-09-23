from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

RELEASE_SHA256 = "838309cae88aebff37e3cbb993a9b5658ab55f5277b4a5fb2302afd53ad49707"
HF_MODEL_REPO = "Tommoro-AI/HABILIS-Brain-0"
HF_MODEL_REVISION = "480d698f8f62191e2713fd91d5b965627abc8f53"
HF_MODEL_FILENAME = "gc-vla-gcrf.ckpt"
MEMBERS = {"gc-vla/config.yaml", "gc-vla/model.pt", "gcrf/router.pt", "gcrf/residual.pt"}


def _stream_digest(stream) -> str:
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def verify_unified_checkpoint(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "gc-vla-gcrf" or manifest.get("schema_version") != 1:
            raise ValueError("unsupported unified checkpoint")
        names = set(archive.namelist())
        if set(manifest["files"]) != MEMBERS or names != MEMBERS | {"manifest.json"}:
            raise ValueError("unexpected checkpoint members")
        if len(names) != len(archive.namelist()):
            raise ValueError("duplicate checkpoint members")
        for name, record in manifest["files"].items():
            if name not in names:
                raise ValueError(f"missing checkpoint member: {name}")
            info = archive.getinfo(name)
            if info.file_size != record["size"]:
                raise ValueError(f"checkpoint member size mismatch: {name}")
            with archive.open(name) as stream:
                if _stream_digest(stream) != record["sha256"]:
                    raise ValueError(f"checkpoint member hash mismatch: {name}")
    return manifest


def extract_unified_checkpoint(path: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    manifest = verify_unified_checkpoint(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gc-vla-", dir=destination.parent))
    try:
        with zipfile.ZipFile(path) as archive:
            for name in manifest["files"]:
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 16 * 1024 * 1024)
        (temporary / "manifest.json").write_text(json.dumps(manifest))
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def resolve_checkpoint(path: str | Path, cache_dir: str | Path) -> Path:
    """Resolve a local checkpoint or the official public Hugging Face artifact."""
    reference = str(path)
    if not reference.startswith("hf://"):
        return Path(reference).expanduser()

    spec = reference.removeprefix("hf://")
    parts = spec.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError("Invalid Hugging Face checkpoint reference")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or HF_MODEL_FILENAME
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required for hf:// checkpoints; install requirements-runtime.txt"
        ) from exc
    revision = HF_MODEL_REVISION if repo_id == HF_MODEL_REPO else None
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(Path(cache_dir).expanduser()),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN"),
    )
    return Path(downloaded)


def prepare_checkpoint(path: str | Path, destination: Path) -> Path:
    archive = resolve_checkpoint(path, destination.parent / "downloads")
    with archive.open("rb") as stream:
        if _stream_digest(stream) != RELEASE_SHA256:
            raise ValueError("Not the frozen release checkpoint: SHA256 mismatch")
    manifest = verify_unified_checkpoint(archive)
    if not destination.exists():
        return extract_unified_checkpoint(archive, destination)
    for name, record in manifest["files"].items():
        with (destination / name).open("rb") as stream:
            if _stream_digest(stream) != record["sha256"]:
                raise ValueError(f"Extracted checkpoint changed: {name}")
    return destination
