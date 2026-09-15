# VBench dense93 — bf16 + cache-dit baseline (NVIDIA B300)

Run 2026-09-13, node `llm-server` (4x NVIDIA B300 SXM6, 275 GB; GPUs 2 and 3 used,
TP=1 per GPU, prompts split 47/46). Baseline for the XPU quantized-recipe
comparisons: **bf16 weights (no `--quantization`) + cache-dit**, same dense93
generation spec as the 3-XPU campaign (see `vbench_dense93_handoff.md`).

## Recipe (pinned)

Wan2.2-T2V-A14B-Diffusers served by vLLM-Omni:
- `--cache-backend cache_dit --enable-cache-dit-summary --boundary-ratio 0.875 --flow-shift 5.0`
- Cache-DiT DBCache defaults (residual_diff_threshold=0.24, Fn_compute_blocks=1,
  dynamic policy, separate CFG)
- 1280x720, 81 frames @ 16 fps, 40 steps, seed 42, guidance 4.0/3.0
  (low/high-noise), 137-char Chinese negative prompt (identical to xpu envs)
- 93 prompts = the VBench `imaging_quality` set, extracted from
  `VBench/vbench/VBench_full_info.json` (unique, order preserved)

## Provenance

| Item | Value |
|---|---|
| vllm-omni | worktree `/models/yiliu7/vllm-omni-baseline` @ `da4a08b65` (last `origin/main` commit on the vllm 0.26.0 base; `311e714d3`+ needs vllm 0.28) |
| generation env | `/models/yiliu7/envs/omni-bf16` (uv, py3.12; vllm 0.26.0, torch 2.11.0+cu130) — repro via `/models/yiliu7/scripts/setup_omni_bf16.sh` |
| scoring env | `/models/yiliu7/envs/vbench` (uv, py3.12; torch 2.11.0+cu130, pyiqa, decord, opencv-headless, av) |
| VBench | clone `/models/yiliu7/VBench` @ `fd18b3d05` (master) |
| env files | `bf16_cachedit_dense93_gpu{2,3}.env` (+ `_smoke` variants) in repo root |
| model | `/models/Wan2.2-T2V-A14B-Diffusers` (118 GB) |
| weights cache | `/models/yiliu7/cache/vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth` (104 MB; `VBENCH_CACHE_DIR`) |

## Generation

- 93/93 videos, `{prompt}-0.mp4` in `/models/yiliu7/wan-res/bf16_cachedit_dense93/`
- Wall time: gpu2 47 videos / 7336 s, gpu3 46 videos / 7155 s (parallel; ~2.0 h total)
- Per video: ~156 s wall (driver), ~154 s server e2e (40 steps, 81 frames) — cache-dit enabled
- Verified: server logs show `Cache-dit enabled successfully on Wan22Pipeline`,
  no quantization config, no tracebacks; all 93 = 1280x720 / 81f / 16 fps;
  `scripts/scan_black.py` over all 93 → 93 ok, 0 black
- Smoke (4 steps / 5 frames) passed on both GPUs before the full run
  (artifacts in `smoke_gpu{2,3}/`)

## Result — VBench `imaging_quality` (MUSIQ-SPAQ, no-reference)

| Metric | Value |
|---|---|
| **overall** | **0.706581 (70.66/100)** |
| per-video min / max | 49.9 / 80.1 |
| n | 93 |

Raw artifacts (scores + per-video detail + prompt mapping):
`/models/yiliu7/wan-res/bf16_cachedit_dense93/vbench_eval/`
(`results_*_eval_results.json`, `prompts.json`).

## Re-running / scoring notes

- Generation (resume-safe):
  `no_proxy=localhost,127.0.0.1 python generate.py bf16_cachedit_dense93_gpu{2,3}.env \
   --prompt-file .../prompts_gpu{2,3}.txt --server-log ... --timeout 2700`
- **Proxy gotcha**: the node's global `http_proxy` (proxy.ims.intel.com:911) has
  `no_proxy` without localhost — driver `urllib` calls to 127.0.0.1 get a 403
  unless `no_proxy=localhost,127.0.0.1` is exported. Use proxy.ims.intel.com:911
  for external downloads.
- Scoring: `VBENCH_EVALUATE=/models/yiliu7/VBench/evaluate.py VBENCH_CACHE_DIR=/models/yiliu7/cache/vbench
  PYTHONPATH=/models/yiliu7/VBench CUDA_VISIBLE_DEVICES=<gpu> envs/vbench/bin/python
  vbench_eval.py --seed 42 -- --videos_path <dir> --dimension imaging_quality
  --mode custom_input --prompt_file <{video:prompt} json> --output_path <out>`
  (`vbench_eval.py` gained a `VBENCH_EVALUATE` env override for the path;
  `PYTHONPATH` is required because `runpy` does not add the VBench dir to `sys.path`.)

## Open

- The XPU dense93 set (sagev3 mxfp8 cache-dit, 93/93 on D93) is generated but
  not yet scored — comparison against this baseline is pending that.
