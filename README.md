# bench-wan

Standalone benchmark runner for diffusion models on vLLM-Omni.
Each configuration is a single `.env` file — the runner code is generic.

## Quick Start

```bash
cd /home/yiliu7/workspace/bench-wan

# Default (TP2, 1280x720, 81 frames, 40 steps, 5 prompts)
python run.py

# Smoke test (fast iteration)
python run.py smoke.env

# Full quality
python run.py full.env

# Eager mode (no CUDA graphs)
python run.py eager.env

# Dry run (print commands only)
python run.py default.env --dry-run
```

## Env File Format

Each `.env` file is a complete configuration. Copy and edit to create new ones:

```bash
cp default.env my_experiment.env
# edit my_experiment.env
python run.py my_experiment.env
```

### Keys

| Key | Default | Description |
|-----|---------|-------------|
| `MODEL` | `/media/hf_models/Wan2.2-T2V-A14B-Diffusers` | Model path |
| `PYTHON` | `/home/yiliu7/workspace/venvs/omni/bin/python` | Python binary |
| `VLLM_OMNI_ROOT` | `/home/yiliu7/workspace/vllm-omni` | Path to vllm-omni repo |
| `TP` | `2` | Tensor parallel size |
| `CUDA_DEVICES` | `0,1` | GPU selection |
| `ENFORCE_EAGER` | `false` | Disable CUDA graphs |
| `ENABLE_CPU_OFFLOAD` | `false` | CPU offload |
| `EXTRA_SERVE_ARGS` | | Extra `vllm serve` flags (space-separated) |
| `TASK` | `t2v` | Benchmark task |
| `WIDTH` / `HEIGHT` | `1280` / `720` | Resolution |
| `NUM_FRAMES` | `81` | Frame count |
| `NUM_INFERENCE_STEPS` | `40` | Denoising steps |
| `NUM_PROMPTS` | `5` | Number of prompts |

## CLI Flags

```bash
python run.py [env_file] [--no-server] [--server-only] [--dry-run] [--timeout N]
```

| Flag | Description |
|------|-------------|
| `--no-server` | Skip server launch, connect to existing |
| `--server-only` | Launch server only, no benchmark |
| `--dry-run` | Print commands without executing |
| `--timeout` | Server startup timeout in seconds |

## Comparing Eval Results

```bash
# Compare all sibling results under final-score/ against the baseline directory
python compare_benchmarks.py final-score/default_bf16_subject_consistency.json

# Compare specific results
python compare_benchmarks.py \
  final-score/default_bf16_subject_consistency.json \
  final-score/default_fp8_linear_subject_consistency.json \
  final-score/default_mxfp4_linear_only_subject_consistency.json
```

The script resolves directories to the latest `*_eval_results.json`, prints the
overall score delta versus the baseline, and shows the largest per-prompt
regressions and improvements.

## Files

```
default.env   # TP2, full quality, 5 prompts
smoke.env     # TP2, small res, 4 steps, 2 prompts
full.env      # TP2, full quality, 10 prompts
eager.env     # TP2 + --enforce-eager
run.py        # Generic runner (reads any .env)
```
