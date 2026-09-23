"""Install pinned public benchmark code and register separately obtained official assets."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

try:
    from gcvla.assets import write_asset_lock
except ModuleNotFoundError:
    from src.gcvla.assets import write_asset_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("libero", "libero-pro"))
    parser.add_argument("--assets", type=Path, required=True, help="Official mesh/texture assets")
    parser.add_argument(
        "--data", type=Path, required=True, help="Contains bddl_files and init_files"
    )
    parser.add_argument("--root", type=Path, default=Path("external"))
    args = parser.parse_args()
    spec = json.loads((Path(__file__).parents[1] / "release/benchmarks.json").read_text())[
        args.benchmark
    ]
    for path in (args.assets, args.data / "bddl_files", args.data / "init_files"):
        if not path.is_dir() or not any(path.iterdir()):
            raise ValueError(f"Missing official benchmark assets: {path}")
    repo = args.root / ("LIBERO" if args.benchmark == "libero" else "LIBERO-PRO")
    if not repo.exists():
        subprocess.run(["git", "clone", "--no-checkout", spec["repository"], str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", spec["commit"]], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
    )
    if head != spec["commit"] or dirty:
        raise ValueError(
            "Existing benchmark checkout differs; use a new --root, do not overwrite it"
        )
    destination = args.root / f"{args.benchmark}-config"
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str((repo / "libero/libero").resolve()),
        "assets": str(args.assets.resolve()),
        "bddl_files": str((args.data / "bddl_files").resolve()),
        "init_states": str((args.data / "init_files").resolve()),
        "datasets": str(args.data.resolve()),
    }
    with (destination / "config.yaml").open("x") as f:
        yaml.safe_dump(config, f)
    write_asset_lock(
        destination / "config.yaml",
        destination / "assets.lock.json",
        benchmark=args.benchmark,
        revision=spec["commit"],
    )
    print(destination.resolve())


if __name__ == "__main__":
    main()
