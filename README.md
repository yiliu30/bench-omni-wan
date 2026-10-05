# bench-wan

Standalone benchmark runner for diffusion models on vLLM-Omni.
Each configuration is a single `.env` file — the runner code is generic.

## Lightning VBench-I2V: 50 paired inputs

`i2v_lightning_subset.py` is a separate, resumable I2V runner following the
same env-file convention. Its paired campaign uses the shared
`i2v_lightning_50.env` plus two complete, recipe-specific env files:
[`i2v_lightning_bf16.env`](i2v_lightning_bf16.env) for BF16/Flash and
[`i2v_lightning_sage.env`](i2v_lightning_sage.env) for MXFP4/Sage+Flash.
The runner reads both recipe files, rejects mismatched shared settings or
backend settings, and pins their contents in `recipe_envs.json` in the
output directory. Pass the shared env file (the default) to the paired
runner; the recipe files are not separate commands. It takes the **first
50 entries tagged `imaging_quality`** from the pinned upstream
`vbench2_beta_i2v/vbench2_i2v_full_info.json`, one video per image–prompt
pair per recipe: BF16 Flash self-attention versus **MXFP4 weights** with Sage
V3 Hybrid self-attention and Flash fallback on blocks `0,33,34,38`.
Cross-attention uses SDPA in both cases. Both recipes use the Lightning
checkpoint, 640x640 from official square crops, 81 frames, 4 Euler steps,
guidance 1.0, seed 0, no cache, the upstream Wan VAE decoder
(`VLLM_OMNI_WAN_VAE_FAST_DECODE=0`), and `--enforce-eager` on XPUs 0–3. The
upstream decoder is used identically in both recipes after a repeatable
XPU 2 fast-decoder device reset; partial fast-decoder results are retained
separately in `results/i2v_lightning_50/` and are not part of this campaign. This
is an ordered, background-heavy exploratory subset, **not** the full 355-pair
VBench-I2V imaging-quality evaluation. VBench recommends five videos per
pair; these recipes produce one.

Fetch the selected official `crop/1-1` JPEGs directly from the upstream
[VBench-I2V Google Drive folder](https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx)
using the host's isolated Python environment with `gdown`:

```bash
cd /home/yiliu7/bench-omni-wan
/home/yiliu7/vbench_i2v/.venv/bin/python fetch_i2v_official.py
```

The downloader records official per-file Drive IDs and SHA-256 digests,
resumes interrupted downloads, and does not accept untracked files.
Alternatively, download the official
[`crop.zip`](https://drive.google.com/uc?id=1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ)
to `/home/yiliu7/vbench_i2v/data/crop.zip`, then use
`i2v_lightning_subset.py extract-official`; the archive may be quota-blocked.
**Do not substitute reconstructed inputs.**

After ensuring `yi-wan` exposes **all four distinct physical XPUs 0–3** and
mounts the Lightning checkpoint read-only, run. This machine currently
uses a read-only bind mount of a zero-copy hardlink mirror under
`/home/yiliu7/hf_models/` at the model path in the env file; the container
bind mount and restored DRM nodes must be rechecked after a restart.

```bash
docker exec yi-wan bash -c 'source /opt/gfx-deps/env.sh && \
  cd /home/yiliu7/bench-omni-wan && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py dry-run'

# Only after obtaining all 50 official crops (or first run extract-official):
docker exec yi-wan bash -c 'source /opt/gfx-deps/env.sh && \
  cd /home/yiliu7/bench-omni-wan && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py prepare && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py preflight && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py smoke && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py run'
```

The runner keeps one model server loaded per GPU and recipe, assigns pairs
`index % 4`, saves `manifest.json`, per-XPU logs and videos under
`results/i2v_lightning_50_upstream_vae/`, verifies each decoded MP4, and only skips
completed records matching the manifest on restart. The `smoke` action
generates the first pair and requires actual Sage/Flash/SDPA call counters
before `run` can proceed. Use `run --retry-failed`
to archive a failed record/partial video before retry. The successful run
creates `comparison/` with matched frames 0/20/40/60/80, per-frame and
per-pair decoded-pixel MAE/SSIM, and a summary. These **are not VBench
imaging-quality scores**. The campaign index and later XPU MUSIQ-SPAQ
subset scores are in [I2V_LIGHTNING_50.md](I2V_LIGHTNING_50.md);
the [imaging-quality results](I2V_IMAGING_QUALITY_RESULTS.md) summarize
the three-recipe comparison.

Never publish source images, checkpoint weights or scoring-model weights
in this repository.

### MXFP4 Sage without forced block fallback

[`i2v_lightning_sage_no_fallback.env`](i2v_lightning_sage_no_fallback.env)
is a standalone third arm using the **same** 50 official crops, prompts,
MXFP4 model weights, Sage V3 Hybrid kernel, upstream VAE, seed, sampling
parameters and four XPU assignments. It sets
`SAGE_ATTN_FORCE_SDPA_BLOCKS=` to disable forced Flash on blocks
`0,33,34,38`; cross-attention still uses SDPA. It does not regenerate the
paired BF16/four-block results or change their output directory.

```bash
docker exec yi-wan bash -c 'source /opt/gfx-deps/env.sh && \
  cd /home/yiliu7/bench-omni-wan && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py preflight i2v_lightning_sage_no_fallback.env && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py smoke i2v_lightning_sage_no_fallback.env && \
  /opt/gfx-deps/venv/bin/python i2v_lightning_subset.py run i2v_lightning_sage_no_fallback.env'
```

The separate `results/i2v_lightning_50_no_fallback/` run has its own
manifest, runtime, recipe env digest, logs and video records. A successful
run requires 50 validated videos and **zero forced self-attention fallback
calls** on every worker. Use `run i2v_lightning_sage_no_fallback.env
--retry-failed` only for recorded failures.
This arm completed **50/50 videos**, 81 frames each at 640x640/16 FPS on
XPUs 0–3. One XPU-2 pair initially contained near-black frames; the failed
attempt is archived, and an identical same-XPU retry produced a valid
video. All 4,050 frames in the final videos were checked for black/flat
frames. This arm does not alter the existing BF16/four-block comparison
metrics; it is included in the separate XPU MUSIQ-SPAQ scoring below.

### XPU imaging-quality scoring

All three completed arms were scored with the upstream MUSIQ-SPAQ
checkpoint and VBench's default `longer` (512x512) preprocessing using
[`score_i2v_imaging_quality_xpu.py`](score_i2v_imaging_quality_xpu.py).
The scorer verifies the 50-pair manifests and all 150 video digests,
decodes with `decord` in separate CPU subprocesses (this version cannot
coexist with initialized XPU in one process), and scores all 81 frames
per video on XPUs. The isolated `pyiqa==0.1.16`, `decord==0.6.0` and
`timm==1.0.30` packages are at
`/home/yiliu7/vbench_i2v/scoring_pkgs`; Intel XPU PyTorch remains in
`/opt/gfx-deps/venv`. The checkpoint is at
`/home/yiliu7/vbench_i2v/weights/musiq_spaq_ckpt-358bb6af.pth`.

```bash
docker exec yi-wan bash -c 'source /opt/gfx-deps/env.sh && \
  cd /home/yiliu7/bench-omni-wan && \
  /opt/gfx-deps/venv/bin/python score_i2v_imaging_quality_xpu.py run'
```

The runner resumes missing per-video score records. Repeated XPU-2
resets required scoring the final ten jobs on XPU 3; only the scoring
device changed, not the source videos or method. Results and recovery
provenance are in
[`results/i2v_imaging_quality_xpu/`](results/i2v_imaging_quality_xpu/).
These are VBench's `imaging_quality` **method** on a nonstandard subset,
not official full-suite VBench scores; input-image fidelity is not scored.

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
