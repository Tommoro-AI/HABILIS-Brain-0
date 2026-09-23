import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from gcvla.assets import verify_asset_lock
from gcvla.checkpoint import RELEASE_SHA256, prepare_checkpoint
from gcvla.provenance import git_revision

SPLITS = "15206,15430,15654,15878,16102,16438,16806,17182,17294,17430,17846,18262,19286,20310,21334,23382,24406,26454,26466"


def _config_file(path: Path) -> Path:
    return path / "config.yaml" if path.is_dir() else path


def _stable_contract_environment(env: dict[str, str]) -> dict[str, str]:
    """Exclude GPU and output-specific paths from a cross-shard run identity."""
    volatile = {
        "MOLMOACT2_ACTION_TRACE_PATH",
        "MOLMOACT2_GCRF_ONPOLICY_TRACE_PATH",
        "MOLMOACT2_GCRF_ROLLOUT_ID",
        "LEROBOT_BATCH_AUDIT_PATH",
        "HF_MODULES_CACHE",
        "CUDA_VISIBLE_DEVICES",
    }
    return {
        key: value
        for key, value in env.items()
        if key not in volatile
        and (key.startswith(("MOLMOACT2_", "LIBERO_")) or key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"))
    }


IDENTITY_FIELDS = frozenset(
    {
        "checkpoint_sha256",
        "runtime_manifest_sha256",
        "wrapper_source_sha256",
        "benchmark_source_sha256",
        "benchmark_revision",
        "asset_lock_sha256",
        "contract_environment_sha256",
    }
)


def _python_source_digest(root: Path, *, exclude=()) -> str:
    """Hash relative source paths and bytes, including uncommitted/untracked Python."""
    ignored = {".git", ".venv", "venv", "__pycache__", "build", "dist", *exclude}
    records = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in ignored)
        for name in sorted(files):
            if name.endswith(".py"):
                path = Path(directory) / name
                records.append(
                    (
                        path.relative_to(root).as_posix(),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    if not records:
        raise ValueError(f"No Python sources found for provenance: {root}")
    return hashlib.sha256(json.dumps(sorted(records), separators=(",", ":")).encode()).hexdigest()


def run_identity(cfg, env: dict[str, str]) -> dict[str, str | None]:
    config = _config_file(cfg.libero_config)
    asset_lock = cfg.asset_lock or config.with_name("assets.lock.json")
    asset_lock_hash = verify_asset_lock(
        config, asset_lock, benchmark=cfg.benchmark, revision=git_revision(cfg.libero_repo) or ""
    )
    return {
        "checkpoint_sha256": RELEASE_SHA256,
        "runtime_manifest_sha256": hashlib.sha256(
            (cfg.runtime / "manifest.json").read_bytes()
        ).hexdigest(),
        "wrapper_source_sha256": _python_source_digest(
            Path(__file__).resolve().parent, exclude=("_runtime",)
        ),
        "benchmark_source_sha256": _python_source_digest(cfg.libero_repo),
        "benchmark_revision": git_revision(cfg.libero_repo),
        "asset_lock_sha256": asset_lock_hash,
        "contract_environment_sha256": hashlib.sha256(
            json.dumps(_stable_contract_environment(env), sort_keys=True).encode()
        ).hexdigest(),
    }


def environment(cfg, output):
    # Preserve the frozen runtime ABI; public configuration exposes no research knobs.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            ("MOLMOACT2_", "GCVLA_", "LIBERO_", "LEROBOT_", "PAPER_", "PRO_", "REUSE")
        )
    }
    flags = {
        "ACTION_EXPERT_DROP_DEPTH_TOKENS": "1",
        "DEPTH_DECODE_FALLBACK": "1",
        "GCVLA_FIXED_QUERY_DEPTH": "1",
        "PRESERVE_RAW_GRIPPER": "1",
        "GCRF_RESIDUAL_PATH": str(cfg.cache / "gcrf/residual.pt"),
        "GCRF_ROUTER_PATH": str(cfg.cache / "gcrf/router.pt"),
        "GCRF_ROUTER_ENABLED": "1",
        "GCRF_ROUTER_OVERRIDE": "1",
        "GCRF_ONPOLICY": "1",
        "EPISODE_FLOW_RNG": "1",
        "GCRF_PRESERVE_BASE_RNG": "1",
        "GCRF_POLICY_SPLIT_BASE_WIDTH": "14870",
        "GCRF_POLICY_SPLIT_BOUNDARIES": SPLITS,
        "GCRF_POLICY_DISABLE_TF32": "1",
    }
    env.update({"MOLMOACT2_" + k: v for k, v in flags.items()})
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(cfg.gpu),
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LIBERO_INIT_STATE_FROM_SEED_BASE": "1000",
            "LIBERO_CONFIG_PATH": str(cfg.libero_config),
            "LEROBOT_BATCH_AUDIT_PATH": str(output / "batch_audit.jsonl"),
            "HF_MODULES_CACHE": str(output / "hf_modules"),
            "PYTHONPATH": os.pathsep.join(
                map(
                    str,
                    [Path(__file__).resolve().parents[1], cfg.runtime, cfg.libero_repo],
                )
            ),
        }
    )
    if cfg.dependencies is not None:
        env["PYTHONPATH"] = str(cfg.dependencies / "deps") + os.pathsep + env["PYTHONPATH"]
        env["LD_LIBRARY_PATH"] = str(cfg.dependencies / "system/usr/lib/x86_64-linux-gnu")
    return env


def command(cfg, suite, task, output):
    return [
        str(cfg.python),
        str(Path(__file__).with_name("runtime_eval.py")),
        "--seed=1000",
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={cfg.cache / 'gc-vla'}",
        "--policy.device=cuda",
        "--policy.norm_tag=libero",
        "--policy.enable_depth_reasoning=true",
        "--policy.num_depth_tokens_per_image=200",
        "--policy.inference_action_mode=continuous",
        "--policy.num_steps=10",
        "--policy.enable_inference_cuda_graph=false",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids=[{task}]",
        "--env.init_states=true",
        "--env.num_steps_wait=50",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=5",
        "--eval.n_episodes=50",
        f"--output_dir={output}",
    ]


def read_result(output, suite, task):
    info = json.loads((output / "eval_info.json").read_text())
    rows = info["per_task"]
    if len(rows) != 1 or rows[0]["task_group"] != suite or rows[0]["task_id"] != task:
        raise ValueError(f"Unexpected task identity: {output}")
    values = rows[0]["metrics"]["successes"]
    batches = [json.loads(x) for x in (output / "batch_audit.jsonl").read_text().splitlines()]
    if len(values) != 50 or len(batches) != 10:
        raise ValueError(f"Incomplete N50: {output}")
    observed = []
    for i, batch in enumerate(batches):
        ids = list(range(i * 5, i * 5 + 5))
        if (
            batch["batch_ix"] != i
            or batch["init_indices"] != ids
            or batch["seeds"] != [1000 + j for j in ids]
        ):
            raise ValueError(f"Unexpected batch metadata: {output}")
        observed.extend(batch["success"])
    if observed != values or any(type(v) is not bool for v in values):
        raise ValueError(f"Result/audit mismatch: {output}")
    return values


def _retry_output(cfg, suite, task, attempt):
    if (suite, task) not in set(cfg.tasks()):
        raise ValueError("Task is outside the configured benchmark")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", attempt):
        raise ValueError("Attempt must be 1-64 letters, digits, underscores or hyphens")
    return cfg.output_dir / "attempts" / attempt / f"{suite}_task{task}"


def evaluate(cfg, shard_index=0, shard_count=1, dry_run=False, retry=None):
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    tasks = list(cfg.tasks())[shard_index::shard_count]
    retry_output = None
    if retry is not None:
        if shard_index != 0 or shard_count != 1:
            raise ValueError("Single-task retry cannot be sharded")
        suite, task, attempt = retry
        retry_output = _retry_output(cfg, suite, task, attempt)
        if retry_output.exists():
            raise FileExistsError(retry_output)
        tasks = [(suite, task)]
    if dry_run:
        print(
            json.dumps(
                [
                    command(cfg, s, t, retry_output or cfg.output_dir / f"{s}_task{t}")
                    for s, t in tasks
                ],
                indent=2,
            )
        )
        return
    if not (cfg.runtime / "lerobot/scripts/lerobot_eval.py").is_file():
        raise FileNotFoundError("Bundled inference runtime is missing; reinstall the package")
    prepare_checkpoint(cfg.checkpoint, cfg.cache)

    # The asset lock covers the immutable benchmark tree. Verify it once per
    # evaluator process rather than re-hashing the entire tree for every task.
    identity = run_identity(cfg, environment(cfg, cfg.output_dir))
    for suite, task in tasks:
        output = retry_output or cfg.output_dir / f"{suite}_task{task}"
        # Never silently reuse old results or overwrite interrupted experiments.
        output.mkdir(parents=True, exist_ok=False)
        cmd = command(cfg, suite, task, output)
        env = environment(cfg, output)
        (output / "launch.json").write_text(
            json.dumps(
                {
                    "command": cmd,
                    "attempt": retry[2] if retry is not None else None,
                    "checkpoint_sha256": RELEASE_SHA256,
                    "benchmark": cfg.benchmark,
                    "suite": suite,
                    "task": task,
                    "runtime": str(cfg.runtime),
                    "run_identity": identity,
                    "libero_repo": str(cfg.libero_repo),
                    "contract_environment": {
                        k: v
                        for k, v in env.items()
                        if k.startswith(("MOLMOACT2_", "LIBERO_", "LEROBOT_"))
                        or k in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM")
                    },
                },
                indent=2,
            )
        )
        with (output / "eval.log").open("x") as log:
            subprocess.run(
                cmd, env=env, cwd=cfg.runtime, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        read_result(output, suite, task)


def summarize(cfg, retries=()):
    selected = {}
    for suite, task, attempt in retries:
        task = int(task)
        if (suite, task) in selected:
            raise ValueError("Duplicate retry selection")
        selected[suite, task] = _retry_output(cfg, suite, task, attempt)
    episodes = []
    expected_identity = None
    for suite, task in cfg.tasks():
        output = selected.get((suite, task), cfg.output_dir / f"{suite}_task{task}")
        if (suite, task) in selected and not (output / "eval_info.json").is_file():
            raise ValueError(f"Selected retry is incomplete: {output}")
        if not (output / "eval_info.json").exists():
            continue
        provenance = json.loads((output / "launch.json").read_text())
        if provenance["checkpoint_sha256"] != RELEASE_SHA256:
            raise ValueError(f"Checkpoint mismatch: {output}")
        identity = provenance.get("run_identity")
        if not isinstance(identity, dict) or not IDENTITY_FIELDS.issubset(identity):
            raise ValueError(f"Missing or incomplete run identity: {output}")
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(f"Mixed runtime or asset provenance: {output}")
        episodes.extend(
            {"suite": suite, "task": task, "init": i, "success": v}
            for i, v in enumerate(read_result(output, suite, task))
        )
    expected = len(list(cfg.tasks())) * 50
    n = len(episodes)
    return {
        "benchmark": cfg.benchmark,
        "selected_retries": [
            {"suite": suite, "task": task, "output": str(output)}
            for (suite, task), output in sorted(selected.items())
        ],
        "completed": n,
        "expected": expected,
        "complete": n == expected,
        "successes": sum(r["success"] for r in episodes),
        "success_pct": 100 * sum(r["success"] for r in episodes) / n if n else None,
        "failures": [r for r in episodes if not r["success"]],
    }
