import argparse
import json
from pathlib import Path

from gcvla.checkpoint import prepare_checkpoint
from gcvla.config import load_config
from gcvla.runner import environment, evaluate, run_identity, summarize


def parser():
    root = argparse.ArgumentParser(prog="gcvla")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("config", type=Path)
    run = commands.add_parser("eval")
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    retry = commands.add_parser("retry")
    retry.add_argument("config", type=Path)
    retry.add_argument("--suite", required=True)
    retry.add_argument("--task", type=int, required=True)
    retry.add_argument("--attempt", required=True)
    retry.add_argument("--dry-run", action="store_true")
    result = commands.add_parser("summarize")
    result.add_argument("config", type=Path)
    result.add_argument(
        "--retry", nargs=3, action="append", default=[], metavar=("SUITE", "TASK", "ATTEMPT")
    )
    return root


def main():
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        print(prepare_checkpoint(cfg.checkpoint, cfg.cache))
        print(run_identity(cfg, environment(cfg, cfg.output_dir)))
    elif args.command == "summarize":
        print(json.dumps(summarize(cfg, args.retry), indent=2))
    elif args.command == "retry":
        evaluate(cfg, dry_run=args.dry_run, retry=(args.suite, args.task, args.attempt))
    else:
        evaluate(cfg, args.shard_index, args.shard_count, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
