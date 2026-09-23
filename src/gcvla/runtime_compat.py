import hashlib
import json
import sys
from pathlib import Path


def activate_runtime():
    root = Path(__file__).with_name("_runtime").resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Bundled runtime is incomplete or modified: {name}")
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"olmo", "lerobot"}:
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise RuntimeError(f"External runtime already imported: {name}")
    sys.path.insert(0, str(root))


def register_public_model_name():
    activate_runtime()
    from olmo.models import model_config

    original = model_config.get_model_types
    if "gc_vla" not in original():

        def get_model_types():
            types = original()
            types["gc_vla"] = types["molmoact2"]
            return types

        model_config.get_model_types = get_model_types
