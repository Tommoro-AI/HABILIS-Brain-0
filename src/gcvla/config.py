from dataclasses import dataclass
from pathlib import Path

import yaml

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
AXES = ("lan", "object", "swap", "task")


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark: str
    checkpoint: str | Path
    cache: Path
    output_dir: Path
    python: Path
    libero_repo: Path
    libero_config: Path
    gpu: int = 0
    dependencies: Path | None = None
    asset_lock: Path | None = None

    @property
    def runtime(self):
        return Path(__file__).with_name("_runtime")

    def tasks(self):
        for suite in SUITES:
            for axis in AXES if self.benchmark == "libero-pro" else (None,):
                for task in range(10):
                    yield f"{suite}_{axis}" if axis else suite, task


def load_config(path: str | Path) -> EvaluationConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    allowed = set(EvaluationConfig.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Unsupported configuration fields")
    if raw.get("benchmark") not in {"libero", "libero-pro"}:
        raise ValueError("benchmark must be libero or libero-pro")
    for key in allowed - {"benchmark", "gpu", "checkpoint"}:
        if key in {"dependencies", "asset_lock"} and raw.get(key) is None:
            continue
        value = Path(raw[key]).expanduser()
        # Keep virtualenv interpreter symlinks intact; resolving them loses the venv.
        raw[key] = (path.parent / value).absolute() if not value.is_absolute() else value
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, (str, Path)):
        raise TypeError("checkpoint must be a local path or an hf:// reference")
    if isinstance(checkpoint, str) and checkpoint.startswith("hf://"):
        raw["checkpoint"] = checkpoint
    else:
        value = Path(checkpoint).expanduser()
        raw["checkpoint"] = (path.parent / value).absolute() if not value.is_absolute() else value
    if type(raw.get("gpu", 0)) is not int or raw.get("gpu", 0) < 0:
        raise ValueError("gpu must be a nonnegative integer")
    return EvaluationConfig(**raw)
