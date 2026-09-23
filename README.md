# HABILIS Brain 0

This package provides the GC-VLA + GCRF runtime and an optional LIBERO benchmark runner.

## Paper

[arXiv:2609.25558](https://arxiv.org/abs/2609.25558)

## Reproduce the Reported Evaluations

- **LIBERO — 99.55% (1,991/2,000):** `gcvla prepare configs/eval_libero.yaml`, then `gcvla eval configs/eval_libero.yaml`.
- **LIBERO-Pro — 51.09% (four-axis macro-average):** `gcvla prepare configs/eval_libero_pro.yaml`, then `gcvla eval configs/eval_libero_pro.yaml`.

Both commands use the public checkpoint below; each config selects its benchmark.

## Checkpoint

The default input is the public Hugging Face artifact
[`Tommoro-AI/HABILIS-Brain-0`](https://huggingface.co/Tommoro-AI/HABILIS-Brain-0).
The pinned file is `gc-vla-gcrf.ckpt` (ZIP64, 27,339,289,222 bytes). It contains the
base model, causal binary router and unified residual. Its SHA256 is:

```text
838309cae88aebff37e3cbb993a9b5658ab55f5277b4a5fb2302afd53ad49707
```

The example YAMLs use `hf://Tommoro-AI/HABILIS-Brain-0`. The first command
downloads the pinned revision into the Hugging Face cache and verifies the archive
before extracting the runtime cache. Set `HF_TOKEN` only when using a private fork.
For offline use, `checkpoint` may instead be an absolute path to the same archive.
The source archive is read-only. When updating the package, remove the old
extracted cache at `~/.cache/gcvla/habilis-brain-0/extracted` before running
`prepare`; incompatible cached weights are rejected.
Allow approximately 28 GB additional space for extraction.

## Standalone Setup

Model, ActionExpert, GCRF, preprocessing and evaluator source are bundled inside
`gcvla/_runtime`.

Use Linux, Python 3.12 and an NVIDIA GPU with enough memory for FP32 inference.
The benchmark renderer also requires the operating-system OSMesa library. On
Ubuntu, install `libosmesa6` before running evaluation (`sudo apt-get install
libosmesa6`). Python requirements do not install this system library.
```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --require-hashes --no-deps -r requirements-build.txt
.venv/bin/python -m pip install --require-hashes --no-deps --no-build-isolation -r requirements-runtime.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation .
```

The manual commands require a CUDA PyTorch environment compatible with the pinned
runtime requirements. The hash-checked `requirements-runtime.txt` is the Linux
installation contract.
The Qwen/Qwen3-4B **tokenizer only** is fetched/cached by Transformers on first use;
all model/router/residual weights come from the unified checkpoint.

LIBERO benchmark code, mesh/texture assets, BDDL and initial states are separate
public dependencies. Download the exact revisions below from the project root:

```bash
git clone --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
git -C external/LIBERO fetch origin f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
git -C external/LIBERO checkout --detach f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
git clone --no-checkout https://github.com/Zxy-MLlab/LIBERO-PRO.git external/LIBERO-PRO
git -C external/LIBERO-PRO fetch origin eafdb809426b13153aa1e4c42d6601844217dfec
git -C external/LIBERO-PRO checkout --detach eafdb809426b13153aa1e4c42d6601844217dfec
.venv/bin/python - <<'PYDATA'
from huggingface_hub import snapshot_download
snapshot_download(
    "zhouxueyang/LIBERO-Pro", repo_type="dataset",
    revision="c86fc3b8293185a6f373677018ff3e37f8391602",
    allow_patterns=["bddl_files/**", "init_files/**", "SHA256SUMS.txt"],
    local_dir="external/libero-pro-data",
)
PYDATA
.venv/bin/python scripts/setup_benchmark.py libero \
  --assets external/LIBERO/libero/libero/assets \
  --data external/LIBERO/libero/libero
.venv/bin/python scripts/setup_benchmark.py libero-pro \
  --assets external/LIBERO-PRO/libero/libero/assets \
  --data external/libero-pro-data
```

Use each benchmark's own assets directory; Pro includes additional objects.
The setup command records hashes of the supplied files to detect later changes.
The example YAMLs already point to the public model. Change `checkpoint` only when
using a local archive or a compatible mirror.

Without administrator access on Ubuntu, install OSMesa into a local directory:

```bash
mkdir -p dependencies/packages dependencies/system
(cd dependencies/packages && apt download libosmesa6 libllvm20 libdrm2)
for package in dependencies/packages/*.deb; do
  dpkg-deb -x "$package" dependencies/system
done
```

This package list applies to Ubuntu 24.04's Mesa 25.1.7 build; other distributions
may require different dependencies. Set `dependencies: ../dependencies` in each
evaluation YAML when using this local installation.

## Run

```bash
gcvla prepare configs/eval_libero.yaml
gcvla eval configs/eval_libero.yaml --dry-run
gcvla eval configs/eval_libero.yaml
gcvla summarize configs/eval_libero.yaml

gcvla eval configs/eval_libero_pro.yaml
gcvla summarize configs/eval_libero_pro.yaml
```

Paths are resolved relative to each config file. For eight GPUs, give each process
a config with its GPU index and the same output directory, and run shard indices
0 through 7 with `--shard-count 8`. Each task remains a separate continuous B5/N50
process; no episode-level slicing is performed. Run `prepare` once before starting
parallel workers. Output directories are exclusive: interrupted or completed runs
are not overwritten or silently resumed. Use a new output directory for reruns.

## Inference API

```python
from gcvla.config import load_config
from gcvla.inference import load_policy

policy = load_policy(load_config("configs/eval_libero.yaml"))
policy.reset(episode_seeds=[1000])  # One seed per observation batch slot.
action = policy.select_action(observation)
```

Use the LeRobot observation schema and preprocessing. Call `load_policy`
before importing torch, and use a dedicated inference process.

## Benchmark Contract

Seed 1000; B5; continuous initial states 0--49; 50 settling steps; relative
control; ten flow steps; CUDA graphs off; episode-local flow RNG; causal routing
at every replan with per-stream memory reset at episode start; fixed residual
split boundaries; TF32 disabled for residual policy;
Terminal/reset lifecycle is provided by the
evaluator. Pro instructions are read from BDDL.

### Early termination and reset handling

The bundled LIBERO evaluator follows the double-reset fix from
[LeRobot PR #4273](https://github.com/huggingface/lerobot/pull/4273):
`LiberoEnv.step()` returns the terminal observation without resetting, while
`SyncVectorEnv` owns the subsequent `AutoresetMode.NEXT_STEP` reset. This keeps
the terminal observation intact and prevents a second reset from advancing the
initial-state sequence unexpectedly.
