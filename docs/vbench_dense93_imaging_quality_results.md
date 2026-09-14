# VBench dense93 — `imaging_quality` results (bf16 baseline vs XPU sagev3 mxfp8)

Date: 2026-09-14 · Branch: `vbench-dense93-3xpu`

## Scores

| Set | Recipe | imaging_quality | min | median | max |
|---|---|---|---|---|---|
| **BF16 baseline** (B300) | bf16 weights + `cache_dit` | **70.66 / 100** | 49.94 | 71.36 | 80.11 |
| **XPU run** (D93) | sagev3 hybrid + MXFP8 linear + `cache_dit` | **68.08 / 100** | 27.62 | 71.85 | 79.49 |

Δ (XPU − bf16) = **−2.58**. The medians are essentially tied (XPU even slightly
higher); the whole gap comes from a low tail on the XPU set (min 27.62 vs 49.94).

## Generation recipe (identical for both sets)

Wan2.2-T2V-A14B, 93 dense prompts (the VBench `imaging_quality`/`aesthetic_quality`
set, from `VBench/vbench/VBench_full_info.json`), one video per prompt named
`{prompt}-0.mp4`:

- 1280x720, 81 frames @ 16 fps, 40 denoising steps, seed 42
- dual-stage guidance 4.0 (low-noise) / 3.0 (high-noise)
- `--boundary-ratio 0.875 --flow-shift 5.0`, 137-char Chinese negative prompt
- `--cache-backend cache_dit --enable-cache-dit-summary`

Differences between the two runs:
- BF16 baseline: no quantization (bf16 weights), plain attention, NVIDIA B300 (TP=1)
- XPU run: `--quantization mxfp8` + SageAttention V3 hybrid (`SAGE_ATTN_3`,
  forced SDPA fallback blocks 33,34,38,39), Intel XPU (TP=1)

## Provenance

**BF16 baseline (generated 2026-09-13, scored 2026-09-13 14:11 UTC)**
- Node: `llm-server` (4x B300 SXM6 AC), generation on GPUs 2 and 3 (47/46 split,
  one TP=1 server per GPU, ports 8090/8091)
- Env: `/models/yiliu7/envs/omni-bf16` — vllm 0.26.0, torch 2.11.0+cu130,
  vllm-omni clean worktree `/models/yiliu7/vllm-omni-baseline` pinned to
  `da4a08b65bfe717d8c4bc1a98decd0c8ad88976b` (last `origin/main` commit before
  the vllm 0.28.0 rebase `651b03225`; base image `vllm/vllm-openai:v0.26.0`)
- Env files: `bf16_cachedit_dense93_gpu2.env` / `bf16_cachedit_dense93_gpu3.env`
- 93/93 generated, 0 failures; wall time 7336 s (GPU2) / 7155 s (GPU3)
- Model: `/models/Wan2.2-T2V-A14B-Diffusers` (118 GB)

**XPU run (D93 campaign, 2026-09-12/13, see `vbench_dense93_handoff.md`)**
- 93/93 generated on D93 xpu0/xpu1 with
  `xpu_sagev3_mxfp8_cachedit_fb33343839_xpu{0,1}.env`
- Videos taken from HF dataset `Yi30/vbench-wan22-t2v-dense93-xpu`
  (downloaded to `/models/yiliu7/wan-res/vbench_dense93_xpu`, 93/93 spec-checked:
  1280x720/81f/16fps)
- **Caveat**: that set was produced by three different `deepklox` kernel builds
  during the campaign (pre-07:27 build, 07:37 build `315eea85`, and 5-param
  `f200d765` + compat shim). The low-score tail is consistent with that
  non-uniformity; regenerate with a single build if uniformity matters.

## Scoring setup (both sets, identical)

- VBench `imaging_quality` (MUSIQ-SPAQ, no-reference IQA), `--mode custom_input`
- VBench checkout: `/models/yiliu7/VBench` @ `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`
- Scoring env: `/models/yiliu7/envs/vbench` (torch 2.11.0+cu130, pyiqa 0.1.16),
  weights under `VBENCH_CACHE_DIR=/models/yiliu7/cache/vbench` (MUSIQ-SPAQ ckpt
  `musiq_spaq_ckpt-358bb6af.pth`, 104 MB), seed 42, GPU B300
- Driver: `vbench_eval.py` with `VBENCH_EVALUATE=/models/yiliu7/VBench/evaluate.py`

## Per-prompt comparison (93/93 matched by filename)

- XPU worse on 60 prompts, better on 33
- Largest regressions (bf16 − xpu):
  | Δ | bf16 | xpu | prompt |
  |---|---|---|---|
  | +42.59 | 70.21 | 27.62 | time lapse of sunrise on mars. |
  | +33.93 | 63.06 | 29.12 | Origami dancers in white paper, 3D render, ... |
  | +23.31 | 67.78 | 44.47 | An animated painting of fluffy white clouds moving in sky. |
  | +22.63 | 59.38 | 36.75 | Splash of turquoise water in extreme slow motion, ... |
  | +19.33 | 67.38 | 48.05 | An epic tornado attacking above a glowing city at night, ... |
- Largest improvements (xpu − bf16):
  | Δ | bf16 | xpu | prompt |
  |---|---|---|---|
  | +14.34 | 60.82 | 75.16 | a shark is swimming in the ocean. |
  | +11.88 | 63.93 | 75.81 | Turtle swimming in ocean. |
  | +9.33 | 63.31 | 72.64 | An astronaut is riding a horse in the space ... |
  | +8.54 | 64.77 | 73.31 | Balloon full of water exploding in extreme slow motion. |
  | +8.39 | 68.76 | 77.14 | A 3D model of a 1800s victorian house. |

## Artifacts

- BF16 videos + logs: `/models/yiliu7/wan-res/bf16_cachedit_dense93/`
  (uploaded: HF `Yi30/vbench_dense93`, videos under `bf16/`)
- BF16 result: `/models/yiliu7/wan-res/bf16_cachedit_dense93/vbench_eval/results_2026-09-13-14:11:52_eval_results.json`
- XPU videos: `/models/yiliu7/wan-res/vbench_dense93_xpu/`
  (source: HF dataset `Yi30/vbench-wan22-t2v-dense93-xpu`)
- XPU result: `/models/yiliu7/wan-res/vbench_dense93_xpu/vbench_eval/results_2026-09-14-02:31:08_eval_results.json`
- Full per-video scores are in each `*_eval_results.json` (`video_results` lists).

## Scope notes

- Only `imaging_quality` (no-reference) was scored for both sets, per the
  campaign goal. Other dimensions (aesthetic, consistency, motion, semantic)
  are not yet scored for either set.
- Black-frame sweep (`scripts/scan_black.py`, 9 sampled frames per video,
  BLACK = min<12 and std<8): **93/93 ok, 0 black on both sets** (re-swept
  2026-09-14 after the XPU download; the XPU set had also been swept on D93
  at campaign end). The XPU set was generated across three kernel builds.
