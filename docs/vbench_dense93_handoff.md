# VBench dense93 on vLLM-Omni (Intel XPU) — Runbook & Handoff

Branch: `vbench-dense93-3xpu`. Written 2026-09-12 for the CRI node D93 migration;
valid for any node that satisfies section 3.

## 1. What this is

Generate the full VBench **dense93** prompt set (the 93 general prompts; per
`docs/vbench_dimensions.md`, `imaging_quality` and `aesthetic_quality` use the
identical 93-prompt set) with **Wan2.2-T2V-A14B-Diffusers** served by
**vLLM-Omni** on Intel XPU, using the quantized kernel recipe:

- `--quantization mxfp8` (ModelOptMxFp8Config in the container vllm build)
- `--cache-backend cache_dit --enable-cache-dit-summary`
- SageAttention V3 hybrid kernel (`SAGE_ATTN_3`) from `deepklox-sage3`, with
  forced plain-SDPA fallback on blocks 33,34,38,39

Generation spec (pinned): 1280x720, 81 frames @ 16 fps, 40 steps, seed 42,
dual-stage guidance 4.0 (low-noise) / 3.0 (high-noise), boundary_ratio 0.875,
flow_shift 5.0, 137-char Chinese negative prompt. One video per prompt, named
`{prompt}-0.mp4` (VBench standard naming — this is what makes resume/skip and
later scoring work).

## 2. Repo layout (bench-omni-wan)

| File | Purpose |
|---|---|
| `run.py` | Generic runner: start server from `.env`, run benchmark, kill server |
| `generate.py` | Generation-only driver: same `.env`, sync `/v1/videos/sync`, resume-safe skip of existing outputs. Flags: `--prompt-file`, `--prompt`, `--num-prompts`, `--output-dir`, `--no-server`, `--server-log`, `--timeout`, `--dry-run` |
| `xpu_sagev3_mxfp8_cachedit_fb33343839_xpu{0,1,2}.env` | Full-quality server+generation recipe, one per XPU (ports 8098/8099/8100) |
| `xpu_sagev3_mxfp8_cachedit_fb33343839_xpu{0,1,2}_smoke.env` | Smoke variant (4 steps / 5 frames) |
| `scripts/vbench_watchdog_3xpu.sh` | Campaign watchdog per XPU (canonical copy also at `vllm-omni-inner/wan/vbench_benchomni_watchdog3.sh`) |
| `vbench_eval.py` / `vbench_eval.sh` / `vbench_eval_all.py` / `vbench_eval_folder.py` / `vbench_compare_common.py` | VBench scoring (section 8) |
| `docs/vbench_dimensions.md` | VBench dimension reference (models, prompt counts, mechanics) |
| `docs/vbench_common10_results.md` | Earlier 10-prompt result comparisons |

`generate.py` reads `.env` files (KEY=VALUE + `[env]` section = variables
exported to the server subprocess). `EXTRA_VIDEO_FIELDS_JSON` is a single-line
JSON of extra per-request `/v1/videos` form fields (guidance 4.0/3.0,
boundary_ratio, negative prompt).

Prompt lists (NOT in this repo — keep them next to the outputs):
`<base>/prompts.txt` (93), `prompts_xpu{0,1}_3way.txt` (31/31/31),
`prompts_smoke_xpu{0,1}.txt` (2 each). On D93:
`/workspace/tmp_yi_yiwan/vbench_dense93/` (host: `/home/mengke/tmp_yi_yiwan/`).

## 3. Environment prerequisites (node + container)

A working XPU node (D93 has 3 XPUs; 2 also works — use xpu0/xpu1 only).

**Container**: `yi-wan:latest` (or equivalent) with:
- mounts: `<workspace-host-dir>` → `/workspace`, model store →
  `/workspace/hf_models` (D93: `/home/mengke` and `/home/hf_models`; the
  Wan2.2 dir is a symlink into the big store, e.g. `/mnt/disk3/hf_models`),
  `/dev/dri`, `--shm-size=32g --ipc=host`
- `/opt/gfx-deps/venv` — Python 3.12 + torch 2.15.0a0 (XPU) + vllm-omni
  (import resolves to `VLLM_OMNI_ROOT`, see below)
- `/opt/gfx-deps/env.sh` — oneAPI (umf/oneccl/dpcpp); auto-sourced by login
  shells in this image
- `/home/ubuntu/main_cri_toolchain/env.sh` — **must be sourced before
  anything touches torch** (provides `LD_LIBRARY_PATH` for `libpti_view.so.1`,
  `libmpicxx.so.12`, dnnl/oneccl libs; without it `import torch` fails with
  `ImportError: libpti_view.so.1`)

**Paths the `.env` files expect** (all container-visible):

| Path | Content |
|---|---|
| `/workspace/hf_models/Wan2.2-T2V-A14B-Diffusers` | model (118G: transformer 54G + transformer_2 54G + text_encoder 11G + vae + tokenizer); symlink into the real store is fine |
| `/workspace/vllm-omni-inner` | vLLM-Omni repo (`VLLM_OMNI_ROOT`; `vllm_omni` package) |
| `/workspace/deepklox-sage3` | SageAttn3 kernel (`PYTHONPATH`). Authoritative kernel identity: `deepklox/_C.cpython-312-x86_64-linux-gnu.so`, **md5 `f200d765`** (the `build/` copy is a different, older build) |

**Quick container health check** (from host):

```bash
docker exec yi-wan bash -c '
source /opt/gfx-deps/env.sh >/dev/null 2>&1
source /home/ubuntu/main_cri_toolchain/env.sh >/dev/null 2>&1
/opt/gfx-deps/venv/bin/python3 -c "import torch; assert torch.xpu.is_available(); print(\"xpu devices:\", torch.xpu.device_count())"
/opt/gfx-deps/venv/bin/python3 -c "import vllm_omni; print(vllm_omni.__file__)"
md5sum /workspace/deepklox-sage3/deepklox/_C.cpython-312-x86_64-linux-gnu.so'
# expect: device count = number of /dev/dri render nodes, vllm_omni from
# /workspace/vllm-omni-inner, md5 f200d765
```

## 4. Recipe (env file anatomy)

`xpu_sagev3_mxfp8_cachedit_fb33343839_xpu<N>.env`:

- `MODEL`, `PYTHON=/opt/gfx-deps/venv/bin/python3`,
  `VLLM_OMNI_ROOT=/workspace/vllm-omni-inner`, `HOST=127.0.0.1`,
  `PORT=809{8,9}/8100`, `TP=1`
- `EXTRA_SERVE_ARGS=--quantization mxfp8 --cache-backend cache_dit
  --enable-cache-dit-summary --boundary-ratio 0.875 --flow-shift 5.0`
- generation block: `WIDTH=1280 HEIGHT=720 NUM_FRAMES=81 FPS=16
  NUM_INFERENCE_STEPS=40 SEED=42 OUTPUT_DIR=<base dir>`
- `EXTRA_VIDEO_FIELDS_JSON` (one line): `guidance_scale=4.0`,
  `guidance_scale_2=3.0` (high-noise stage guidance, see
  `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py`), `boundary_ratio=0.875`,
  negative prompt (137 chars, Chinese)
- `[env]` (exported to the server subprocess; read at module import, so they
  must exist **before** the server starts):
  - `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=<N>` (XPU pin)
  - `PYTHONPATH=/workspace/deepklox-sage3`
  - `DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3`
  - `SAGE_ATTN_FORCE_SDPA_BLOCKS=33,34,38,39`
  - `DEEPKLOX_SAGEATTN_HYBRID_FUSED_PREP=1`, `SAGE_ATTN_XPU_LAYOUT=auto`,
    `SAGE_ATTN_REPORT_FALLBACKS=1`
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`

Server entry point is `vllm-omni serve <model> --omni ...` (generate.py uses
`python -m vllm_omni.entrypoints.cli.main serve ...`), **not** plain
`vllm serve`: the container's `vllm` is the Intel XPU build without the omni
app wiring; the omni CLI only activates when `--omni` is in argv.

Job API: `POST /v1/videos` → poll `GET /v1/videos/{id}`
(queued|in_progress|completed|failed) → `GET /v1/videos/{id}/content`;
health `GET /health`.

## 5. How to run

All commands run on the **host**, targeting the `yi-wan` container. `<base>`
= campaign output dir (D93: `/workspace/tmp_yi_yiwan/vbench_dense93` inside
the container).

### 5.1 Env prelude (needed for every launch)

Every launcher must source the toolchain environments before running. The
watchdog script applies the full prelude itself (see its header); when
launching `generate.py` manually, mirror the same prelude. If localhost job
submissions fail, compare your prelude against
`scripts/vbench_watchdog_3xpu.sh` line by line.

### 5.2 Smoke (per XPU, ~5 min wall incl. server start)

```bash
docker exec -d yi-wan bash -c "cd /workspace/bench-omni-wan && \
  source /opt/gfx-deps/env.sh >/dev/null 2>&1 && \
  source /home/ubuntu/main_cri_toolchain/env.sh >/dev/null 2>&1 && \
  /opt/gfx-deps/venv/bin/python3 generate.py \
    xpu_sagev3_mxfp8_cachedit_fb33343839_xpu${XPU}_smoke.env \
    --prompt-file <base>/prompts_smoke_xpu${XPU}.txt \
    --output-dir <base>/smoke3_xpu${XPU} \
    --server-log <base>/logs/xpu${XPU}_smoke3_server.log --timeout 900 \
    > <base>/logs/xpu${XPU}_smoke3_driver.log 2>&1"
```

Pass on section 6 criteria. Do this on **every XPU before the full run**,
especially on a new node.

### 5.3 Single XPU full run (no watchdog)

```bash
docker exec -d yi-wan bash -c "cd /workspace/bench-omni-wan && <prelude> && \
  /opt/gfx-deps/venv/bin/python3 generate.py \
    xpu_sagev3_mxfp8_cachedit_fb33343839_xpu${XPU}.env \
    --prompt-file <base>/prompts_xpu${XPU}_3way.txt \
    --server-log <base>/logs/xpu${XPU}_full_server.log --timeout 2700 \
    > <base>/logs/xpu${XPU}_full_driver.log 2>&1"
```

### 5.4 Multi-XPU campaign with watchdog (recommended)

Watchdog per XPU = `generate.py` (owns its server: start → all jobs → kill)
retried up to 5 times while any video in the prompt list is still missing,
with an XPU probe between attempts. Resume-safe: existing
`{prompt}-0.mp4` files are skipped, so relaunch after a crash/cutover just
fills the gaps.

```bash
for x in 0 1 2; do
  docker exec -d yi-wan bash -c "cd /workspace/bench-omni-wan && \
    bash scripts/vbench_watchdog_3xpu.sh $x"
done
```

The watchdog applies the env prelude itself (see script header). It logs to
`<base>/logs/xpu${XPU}_watchdog3.log` (per-attempt missing counts) and
`<base>/logs/xpu${XPU}_full_server_n2.log` (server).

Two D93 lessons baked into `scripts/vbench_watchdog_3xpu.sh` (2026-09-12):
- per-job poll timeout is `--timeout 2700` (45 min). The earlier 900 s cap
  matched the old node's ~6 min/video, but D93 runs ~14 min/video — a
  900 s cap made slow jobs "timed out" client-side, took the server down
  (connection reset), and cascaded connection-refused failures across the
  rest of the list.
- the XPU probe must use the env the script sourced **at top level** (the
  same env the server subprocess inherits). Re-sourcing the two env files on
  top of an already-sourced shell reorders `LD_LIBRARY_PATH` and breaks
  torch with `undefined symbol: _ZN4sycl3_V17handler...`. On D93 both the
  original inline version and a heredoc "fresh subshell" version failed
  (the heredoc body still re-sourced inside the inherited env); the working
  form is the probe with NO source lines at all (verified 2026-09-12).
- 2026-09-12 14:06: a job on xpu1 hung ~45 min (device stall), only the
  45-min client timeout surfaced it. The campaign monitor
  (`tmp_yi_yiwan/vbench_dense93/monitor/monitor_black.sh`) now also does
  **step-stall detection**: if a running XPU's denoise step counter doesn't
  advance for 25 min it quarantines nothing but restarts that XPU's server
  (same kill sequence as the black-video path) so the watchdog resumes with
  a fresh server.

For 2 XPU: same, `for x in 0 1`, and either keep the 3-way lists (xpu2's 31
stay missing until later) or use the original 47/46 split
(`prompts_xpu0.txt` / `prompts_xpu1.txt` with the legacy
`vllm-omni-inner/wan/vbench_benchomni_watchdog.sh`).

### 5.5 Progress & ETA

- authoritative progress = `ls <base>/*.mp4 | wc -l` (93 = done) + server log
  step progress bars
- driver/watchdog stdout is block-buffered when redirected — "Done in ..."
  lines may lag; don't wait for them to believe a video finished
- per-video time observed: ~6 min warm (old node, 40 steps) up to ~15 min
  including first-compile; 85 remaining on 3 XPUs ⇒ ETA ≈ 3.5–6 h
- server start is ~3.5–4 min (weights load once per XPU; this is why the
  campaign uses long-lived servers per XPU instead of offline per-prompt runs)

## 6. Verification checklist (after each smoke / periodically in full)

From the server log (`logs/xpu<N>_*_server*.log`):

1. kernel identity: `Sage attention V3 kernel build: deepklox: ...
   md5=f200d765 layout_mode=auto` (md5 must match; if not, wrong .so on
   PYTHONPATH — check you point at the repo-root build, not `build/`)
2. `Building quantization config: mxfp8`
3. `Cache-dit enabled successfully on Wan22Pipeline`
4. per job: `Video sampling params: steps=4 guidance=4.0 guidance_2=3.0
   seed=42` (smoke) / `steps=40 ...` (full)
5. `Sage attention V3 call counts: sage=..., forced_sdpa=...,
   cross_sdpa=..., sdpa_fallback=0, nonfinite_sdpa=0, layout_fallback=0`
   (smoke, 4 steps/5 frames: exactly `sage=612, forced_sdpa=68,
   cross_sdpa=680`; non-zero `sdpa_fallback`/`nonfinite` = investigate)
6. no `Traceback`, `UR_RESULT_ERROR`, `DEVICE_LOST`

Videos (any container python with `av`/`imageio`):
- full: 1280x720, 81 frames, 16 fps; smoke: 1280x720, 5 frames, 16 fps
- naming `{prompt}-0.mp4` at the base dir root (VBench standard mode)
- driver prints wall time per video; the `inference: Ns` number is
  stage_durations in **ms** (cosmetic bug, wall time is the real one)

## 7. Ops notes & failure modes

- **XPU health**: the old node had two self-recovering
  `UR_RESULT_ERROR_DEVICE_LOST` hangs under sustained multi-hour load
  (recovered in ~1 min). Mitigations already in place: watchdog restarts the
  server, probes the device between attempts, resume-skip refills gaps. If a
  device hangs, also try a container-level reset before blaming the run.
- Container has **no jq/pgrep**: JSON parsing in scripts uses
  `/opt/gfx-deps/venv/bin/python3`; check server processes with
  `docker exec yi-wan ps aux | grep -i serve` (titles:
  `vLLM-Omni::DiffusionWorker`, `APIServer`).
- Server binds 127.0.0.1 inside the container only — all drivers must run
  **inside** the container.
- The `NVML init failed ... profiling fallback` warning and the Gloo hostname
  warning are benign on XPU.
- Don't `rm`/move the model symlink target without updating
  `/workspace/hf_models/Wan2.2-T2V-A14B-Diffusers`; a broken chain dies at
  startup with `HFValidationError`.
- Known flake (old node, unexplained): one generated mp4 vanished after
  verification. The watchdog's missing-count check is the safety net; if
  videos keep disappearing, check the container's disk/quota and sync jobs.

## 8. VBench scoring (post-generation)

Not yet done as of this writing. Requirements:

- `vbench` python package + a VBench checkout. **Gotcha**: `vbench_eval.py`
  hardcodes `DEFAULT_VBENCH_EVALUATE=/home/yiliu7/workspace/VBench/evaluate.py`
  (yiliu7's old path). On a new node: `git clone VBench` somewhere and either
  edit that constant or `mkdir -p /home/yiliu7/workspace && ln -s <clone>
  /home/yiliu7/workspace/VBench`.
- Model weights download to `~/.cache/vbench` on first run (DINO, CLIP,
  MUSIQ-SPAQ, RAFT, AMT-S, ...). They come from HuggingFace and are
  multi-GB — test the node's model-download network early
  (a mirror endpoint via HF_ENDPOINT may be needed, as for the model
  weights themselves).
- Videos must be flat in one dir named `{prompt}-0.mp4` (our base dir is
  exactly that; point the eval at the base dir, not at smoke subdirs).

Commands (inside container, after prelude):

```bash
# one dimension (quality dims don't need prompts; semantic ones do):
./vbench_eval.sh <base> "<shared prompt or prompts.json>" <out_dir> 0

# all subfolders / dimensions: see vbench_eval_all.py (DIMS list) and
# compare results against a baseline with compare_benchmarks.py
```

For this campaign's goal: dimension `imaging_quality` (MUSIQ-SPAQ,
no-reference, no prompt needed) on the 93-video base dir. Compare against a
bf16 baseline (recipe: `default_bf16.env`) if one is generated.

## 9. Node-switch checklist

On the new node, verify in order (fail fast):

1. XPU devices: `sycl-ls | grep -c level_zero:gpu` (inside container, after
   oneAPI setup) = expected count
2. section 3 health check (torch XPU, vllm_omni import, kernel md5 f200d765)
3. model symlink chain resolves to the full 118G model (check
   `transformer/` ~54G exists)
4. `bench-omni-wan` on branch `vbench-dense93-3xpu` (self-contained:
   generate.py patch + all env files + watchdog script in `scripts/`)
5. campaign base dir with prompt lists and existing videos (copy from the
   old node's `tmp_yi_yiwan/vbench_dense93/`; videos are resume-skipped)
6. `vllm-omni-inner` + `deepklox-sage3` repos present (kernel .so md5!)
7. smoke on every XPU you will use (section 5.2)
8. launch watchdogs (5.4); watch `*_watchdog3.log`

Things that do NOT need to move: model weights (re-download or re-share if
the store isn't mounted), venv (comes with the container image), this repo's
git history.

## 10. Status (2026-09-12)

- 2026-09-11 (old node, 2 XPU): smoke passed both ways; full campaign ran
  00:48–~01:57 UTC → 8/93 videos, cut off by migration.
- 2026-09-12 (node D93): model symlink checked (was already correct),
  smokes passed on xpu0/1/2 (kernel md5, cache-dit, fallback counts, video
  specs all verified). A 3-way campaign (31/31/31, ports 8098/8099/8100)
  launched 04:16:57 UTC under `scripts/vbench_watchdog_3xpu.sh`.
- 2026-09-12 ~04:50 UTC: per operator preference the run was reduced to
  **2 XPUs** (xpu0/xpu1 only; xpu2 job stopped cleanly). xpu2's prompts were
  merged into the active lists: `prompts_xpu0_3way.txt` = 46 prompts,
  `prompts_xpu1_3way.txt` = 47 prompts (no overlap, full 93 coverage;
  originals backed up as `*.bak-3way`). No job restart needed: each running
  `generate.py` finishes its in-memory list, then its watchdog re-reads the
  updated file on the next attempt and resume-skips what exists.
- Next: monitor to 93/93, then VBench `imaging_quality` scoring (section 8).
