# BKC — VBench on Wan2.2-T2V (vLLM-Omni, Intel XPU)

Best-known configuration for generating and scoring the VBench **dense93** set
(93 general prompts; `imaging_quality` / `aesthetic_quality` use this same set)
with **Wan2.2-T2V-A14B-Diffusers** served by **vLLM-Omni** on Intel XPU:
SageAttention V3 hybrid + MXFP8 linear + cache-dit, with a curated set of
transformer blocks forced off the quantized attention path.

Node-agnostic by design: every machine-specific value is `<A_PLACEHOLDER>`;
fill them into one `.env` file per experiment and everything else runs from
`bench-omni-wan/`.

Validated 2026-10-05 on an 8-XPU CRI node (containerized, TP=1): smoke
(5f/4-step) and 17f/40-step arms both green — see §7 for the pass criteria and
§9 for reference timings.

## 1. Component matrix

| Layer | Requirement | Notes |
|---|---|---|
| Model | `Wan2.2-T2V-A14B-Diffusers` (~118 GB) | dual-expert `WanPipeline`; must contain `transformer/` + `transformer_2/` (12 safetensors shards each), `text_encoder/` (UMT5, 3 shards), `vae/`, `tokenizer/`, `scheduler/`, and `model_index.json` with `boundary_ratio: 0.875` |
| Serving | vLLM-Omni (`vllm-omni` repo) with the `SAGE_ATTN_3` diffusion attention backend | entry point is `python -m vllm_omni.entrypoints.cli.main serve … --omni` (plain `vllm serve` does not carry the omni wiring on XPU builds) |
| vllm core | wheel in the container venv with `mxfp8` + `modelopt_mxfp8` in `QUANTIZATION_METHODS` | required for `--quantization mxfp8` (online quant of the bf16 checkpoint) |
| vllm omni |https://github.com/intel-innersource/applications.ai.gpu.vllm-omni/tree/bench-wan|  |
| Attention kernel | `deepklox` checkout exposing `deepklox.sageattn_interface.sageattn_v3_hybrid(q, k, v, tensor_layout, sparsity)` | the V3-hybrid binding must be built in (`DEEPKLOX_SAGEATTN_V3=1` build flag); backend calls it with `tensor_layout="NHD"` — no lse/compat shim needed against a current interface |
| Step cache | `cache_dit` importable in the venv | enabled via `--cache-backend cache_dit`; gives the bulk of the 40-step speedup (§9) |
| Linear | MXFP8 | |
| Negative Prompt | empty `""` | |
| Fallback Blocks | 0,33,34,38 | |


## 2. Machine prerequisites (fill in once per node)

- `<CONTAINER>`: the wan image, started with the XPU device nodes
  (`/dev/dri`), `--ipc=host --shm-size=32g`, and mounts that expose:
  - `<VLLM_OMNI_ROOT>` — vLLM-Omni checkout (§1)
  - `<DEEPKLOX_ROOT>` — deepklox checkout with the V3-hybrid build
  - `<MODEL_DIR>` — the diffusers checkpoint (symlink OK; if it is not in a
    bind-mounted subtree, a privileged container can bind the host directory
    in at runtime — no copy needed, and usually no room for one)
  - `<OUT_BASE>` — writable results dir
- `<bench-omni-wan>` checkout present in the container (it is under the same
  mount as `<OUT_BASE>` in the validated setup).
- `<N>`: a free XPU index; `<PORT>`: a free localhost port.
- Confirm before anything else:

```bash
docker exec <CONTAINER> bash -lc '/opt/gfx-deps/venv/bin/python3 - <<EOF
import torch; assert torch.xpu.is_available(); print("xpus:", torch.xpu.device_count())
import vllm_omni; print("vllm_omni ->", vllm_omni.__file__)          # must be <VLLM_OMNI_ROOT>
import deepklox.sageattn_interface as dk; print("deepklox ->", dk.__file__)  # must be <DEEPKLOX_ROOT>
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as Q
assert "mxfp8" in Q
import cache_dit; print("stack OK")
EOF'
```

If `vllm_omni`/`deepklox` resolve to a different checkout than intended, an
editable install in the venv is shadowing (or being shadowed by) `PYTHONPATH` —
re-test with the env-file `PYTHONPATH` exported explicitly; the resolution
order is box-specific. Pin kernel identity by md5-ing
`<DEEPKLOX_ROOT>/deepklox/_C*.so` and checking its mtime is the build you made,
not a stale one (header-only edits do NOT relink — touch the `.cpp`).

## 3. The recipe (env file)

Start from the template below (instantiated examples live beside it as
`xpu_sagev3_mxfp8_cachedit_fb*_xpu*.env`; the negative-free arms are the
`*_noneg_*` files).
Pinned generation spec — **do not vary it if scoring against the dense93
baselines**: 1280×720, 81 frames @ 16 fps, 40 steps, seed 42, dual-stage
guidance 4.0 (low-noise) / 3.0 (high-noise), boundary_ratio 0.875, flow_shift
5.0, **no negative prompt** — CFG runs against the server default
(empty-string) negative embeddings. The original campaign arm appended a
137-char Chinese negative prompt (matching the offline reference runners);
scores against that baseline will shift, so re-generate the baseline on the
same spec if you compare absolute numbers.

```ini
# --- Server ---
MODEL=<MODEL_DIR>
PYTHON=/opt/gfx-deps/venv/bin/python3
VLLM_OMNI_ROOT=<VLLM_OMNI_ROOT>
HOST=127.0.0.1
PORT=<PORT>
TP=1
ENFORCE_EAGER=false
ENABLE_CPU_OFFLOAD=false
EXTRA_SERVE_ARGS=--quantization mxfp8 --cache-backend cache_dit --enable-cache-dit-summary --boundary-ratio 0.875 --flow-shift 5.0

# --- Generation (full-quality arm) ---
TASK=t2v
BACKEND=v1/videos
DATASET=vbench
MAX_CONCURRENCY=1
WIDTH=1280
HEIGHT=720
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=40
SEED=42
ENABLE_NEGATIVE_PROMPT=false
OUTPUT_DIR=<OUT_BASE>/<arm>

# Dual-stage guidance + boundary as per-request /v1/videos fields, NO
# negative_prompt key (empty-string negative is the server default):
EXTRA_VIDEO_FIELDS_JSON={"guidance_scale": "4.0", "guidance_scale_2": "3.0", "boundary_ratio": "0.875"}

# --- Server subprocess environment ---
[env]
ZE_FLAT_DEVICE_HIERARCHY=FLAT
ZE_AFFINITY_MASK=<N>
PYTHONPATH=<VLLM_OMNI_ROOT>/aaawork/pyshims:<VLLM_OMNI_ROOT>:<DEEPKLOX_ROOT>
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3
SAGE_ATTN_FORCE_SDPA_BLOCKS=0,33,34,38
DEEPKLOX_SAGEATTN_HYBRID_FUSED_PREP=1
SAGE_ATTN_REPORT_FALLBACKS=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Notes:

- `SAGE_ATTN_FORCE_SDPA_BLOCKS` is the tuning knob of this recipe: those
  blocks run the backend fallback (**flash** in current vllm-omni, name of the
  knob is legacy) instead of sage V3. The fallback block set (0,33,34,38) was
  chosen by accuracy sweeps; keep the *count* of blocks in mind when reading
  the call-count verification (§7).
- `SAGE_ATTN_XPU_LAYOUT` is a no-op on current deepklox (layout comes from the
  caller, NHD). Do not rely on it.
- Smoke variant: same file with `NUM_FRAMES=5`, `NUM_INFERENCE_STEPS=4`
  (expected counts in §7), tiny `OUTPUT_DIR`.
- If the container environment sets a corporate HTTP proxy, unset
  `http(s)_proxy` and set `no_proxy=localhost,127.0.0.1` in the launching
  shell — urllib otherwise 403s on the localhost job API.

## 4. Run

Everything below runs **inside** the container (the server binds 127.0.0.1);
wrap in `docker exec -d <CONTAINER> bash -lc '… > <OUT_BASE>/logs/….log 2>&1'`
for long jobs. `generate.py` owns its server (start → all prompts → kill) and
is resume-safe (existing `{prompt}-0.mp4` skipped), so crash/relaunch just
fills gaps.

```bash
cd <bench-omni-wan>

# 1) smoke (seconds per video; run before anything full-quality)
/opt/gfx-deps/venv/bin/python3 generate.py <arm>_smoke.env \
  --prompt-file <OUT_BASE>/prompts_smoke.txt --timeout 2400 \
  --server-log <OUT_BASE>/logs/smoke_server.log \
  > <OUT_BASE>/logs/smoke_driver.log 2>&1

# 2) intermediate arm (recommended): 17f @ 40 steps — real per-step timing, ~1 min/video
/opt/gfx-deps/venv/bin/python3 generate.py <arm>_smoke17.env \
  --prompt-file <OUT_BASE>/prompts_smoke.txt --timeout 2400 ...

# 3) full quality, per-video or campaign
/opt/gfx-deps/venv/bin/python3 generate.py <arm>.env \
  --prompt-file <OUT_BASE>/prompts.txt --timeout 2400 ...

# perf-only alternative (throughput/latency report instead of videos):
/opt/gfx-deps/venv/bin/python3 run.py <arm>.env
```

Campaign-scale runs: use `scripts/vbench_watchdog_3xpu.sh` (retry while any
prompt is missing, XPU probe between attempts) and optionally
`scripts/monitor_black.sh` + `scripts/scan_black.py` (per-video black-frame
quarantine, step-stall server restart). Keep the per-job poll at ≥2700 s if
the node is slower than the reference one — a client-side timeout mid-job
takes the server down and cascades (hard-won lesson from the original
campaign).

Prompt lists are kept next to the outputs, not in the repo: one prompt per
line, full 93 = the VBench master list (`prompts/master.txt` in the VBench
repo; fetch through the egress proxy if direct GitHub is unreachable).

## 5. Video API used (for driver authors)

`POST /v1/videos` (async; used by `generate.py`) → poll `GET /v1/videos/{id}`
(`queued|in_progress|completed|failed`) → `GET /v1/videos/{id}/content` →
`DELETE`. `/v1/videos/sync` also exists. `guidance_scale_2` and
`boundary_ratio` in the form data drive the dual-expert stages; a request
guidance ≤ 1.0 disables CFG and a negative prompt is then irrelevant; omitting
`negative_prompt` altogether (this BKC) keeps CFG but against the server's
default empty-string negative embeddings.

## 6. Black-frame / artifact guard

```bash
/opt/gfx-deps/venv/bin/python3 scripts/scan_black.py "<OUT_BASE>/<arm>/{prompt}-0.mp4"
# BLACK = min < 12 AND std < 8 over 9 sampled frames; healthy T2V videos: min >= ~50
```

## 7. Verification checklist (every run)

From the server log:

1. `Building quantization config: mxfp8`
2. `Cache-dit enabled successfully on Wan22Pipeline`
3. `SageAttention3 XPU forcing the flash fallback for transformer blocks: [0, 33, 34, 38]`
   and `Resolved diffusion attention backend 'SAGE_ATTN_3'` for roles self+cross
4. per job: `Video sampling params: steps=40 guidance=4.0 guidance_2=3.0 seed=42`
   (smoke: `steps=4 …`)
5. final counters: `Sage attention V3 call counts: sage=…, forced_sdpa=…,
   cross_sdpa=…, sdpa_fallback=0, nonfinite_sdpa=0, fallback_sdpa=0`
   — `sdpa_fallback`/`nonfinite_*` must be **zero**.
   Reference absolute values, 4-step × 5-frame smoke (1 dummy + 2 jobs):
   `sage=612, forced_sdpa=68, cross_sdpa=680`. Per full block-sweep there are
   36 sage + 4 forced (4-block set) + 40 cross calls; cache-dit skips
   recomputation on most of the 40-step schedule, so the 40-step counts scale
   with *executed* sweeps, not with `steps` (validated 17f run:
   `sage=2196, forced_sdpa=344, cross_sdpa=2540` for 2×40-step jobs).
6. driver exits `Done. Generated: N, Skipped: M`, `EXITED rc=0`; no
   `Traceback` / `UR_RESULT_ERROR` / `DEVICE_LOST` in the log.
7. Videos: `<WIDTH>x<HEIGHT>` (frame_shape `(720, 1280, 3)`), frames
   `NUM_FRAMES`, 16 fps, named `{prompt}-0.mp4` flat in the output dir
   (VBench standard mode), black-scan ok.

## 8. VBench scoring

- Requires a VBench checkout + `vbench` package and its model weights
  (MUSIQ-SPAQ, DINO, CLIP, RAFT, …) downloaded to `~/.cache/vbench` on first
  run — multi-GB over the network; validate the node's egress early
  (HF_ENDPOINT mirror / proxy may be needed).
- `vbench_eval.sh <videos_dir> "<prompts>" <out_dir> <gpu>` wraps
  `VBench/evaluate.py`. **Gotcha**: `vbench_eval.py` hardcodes a local
  `VBench/evaluate.py` path — fix the constant or symlink it.
- `imaging_quality` / `aesthetic_quality` need only the flat video dir (no
  prompt pairing); semantic dimensions need the prompt files.
- Compare arms with `compare_benchmarks.py` / `vbench_compare_common.py`
  against a baseline arm (e.g. bf16 + cache-dit: `bf16_cachedit_dense93_*.env`).

## 9. Reference timings (validated node, TP=1, per video)

| Arm | Steps | Time |
|---|---|---|
| server boot (incl. mxfp8 online-quant of both experts) | — | ~4–5 min |
| 5f/4-step smoke | 4 | ~7–8 s |
| 17f × 40 steps | 40 | ~70 s (≈1.7 s/it with cache-dit active) |
| 81f × 40 steps | 40 | est. 5–9 min (~4.2× tokens vs 17f arm) |

93-prompt campaign on one XPU ≈ 8–12 h at that pace (resume-safe, so
splittable across cards by prompt shard with per-card envs on distinct ports).

## 10. Deviations from the original campaign recipe (keep in mind for cross-node comparison)

- Forced fallback blocks use the **flash** implementation (first-generation build: SDPA).
  The blocks under test (the 36 sage ones) are unaffected.
- No `kernel build … md5=` identity line in current vllm-omni logs — verify
  kernel identity from the venv side (§2) instead.
- `SAGE_ATTN_XPU_LAYOUT` ignored (layout fixed to NHD by the backend).
- The `inference: N…s` figure in the driver's `Done in` line is stage
  milliseconds (cosmetic); trust the wall figure / `OmniTiming` lines.
