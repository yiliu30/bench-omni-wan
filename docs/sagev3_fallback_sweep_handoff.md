# Sage V3 Hybrid fallback-block sweeps (2 & 3 pinned blocks) — handoff

2026-09-13, node D93-C, container `yi-wan`.

Follow-up to the dense93 campaign (`docs/vbench_dense93_handoff.md`), which
ran with 4 transformer blocks (33,34,38,39) pinned to plain SDPA
(`SAGE_ATTN_FORCE_SDPA_BLOCKS`) while the other 36 used the deepklox Sage V3
Hybrid quantized kernel. These sweeps asked: how many pins are actually
needed?

- **2-block sweep**: all C(4,2)=6 pairs -> **6/6 PASS**
- **3-block sweep**: all C(4,3)=4 triples -> **4/4 PASS**

## Fixed recipe (identical across all 10 runs)
- Prompt: "Two anthropomorphic cats in comfy boxing gear and bright gloves
  fight intensely on a spotlighted stage." (+ standard negative prompt)
- 720x1280 @ 16fps, 81 frames, 40 steps, seed 42, guidance 4.0/3.0,
  boundary 0.875, flow shift 5.0
- MXFP8 linear quant + cache-dit (DBCache) + cache-dit summary
- DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3, layout auto (zero-copy NHD),
  fused prep, SAGE_ATTN_REPORT_FALLBACKS=1
- Kernel: /workspace/deepklox-sage3, .so md5 f200d765 (5-arg binding),
  used via the Python compat shim
  `/workspace/deepklox-sage3-compat/sitecustomize.py` on PYTHONPATH
  (auto no-ops once a 6-arg lse build is installed)
- Runner: `/workspace/vllm-omni-inner/wan/run_wan17_sagev3_fb_sweep_newbuild.sh`
  (= the fb33_34_38_39 newbuild runner + compat PYTHONPATH prefix, the only
  diff); drivers: `sweep_fb2_driver2.sh` / `sweep_fb3_driver.sh`

## Reference counters (40 steps; cache-dit skips 9 steps of block compute)
- 4-block baseline: `forced_sdpa=124, sage=1166`
- 2-block runs:    `forced_sdpa=62,  sage=1228` (1166+62)
- 3-block runs:    `forced_sdpa=93,  sage=1197` (1166+31)

## Results — 2-block (xpu0/xpu1, 03:27-04:41 UTC)
| fb set | xpu | status | forced_sdpa | sdpa_fallback | nonfinite | layout_fallback | sage | nhd_zero_copy |
|--------|-----|--------|-------------|---------------|-----------|-----------------|------|---------------|
| 33,34  | 0   | PASS   | 62          | 0             | 0         | 0               | 1228 | 1228          |
| 33,38  | 0   | PASS   | 62          | 0             | 0         | 0               | 1228 | 1228          |
| 38,39  | 0   | PASS   | 62          | 0             | 0         | 0               | 1228 | 1228          |
| 33,39  | 1   | PASS   | 62          | 0             | 0         | 0               | 1228 | 1228          |
| 34,38  | 1   | PASS   | 62          | 0             | 0         | 0               | 1228 | 1228          |
| 34,39  | 1   | PASS   | 62          | 0             | 0         | 0               | 1228 | 1228          |

## Results — 3-block (xpu0/xpu1, 04:57-05:35 UTC)
| fb set    | xpu | status | forced_sdpa | sdpa_fallback | nonfinite | layout_fallback | sage | nhd_zero_copy |
|-----------|-----|--------|-------------|---------------|-----------|-----------------|------|---------------|
| 33,34,38  | 0   | PASS   | 93          | 0             | 0         | 0               | 1197 | 1197          |
| 33,38,39  | 0   | PASS   | 93          | 0             | 0         | 0               | 1197 | 1197          |
| 33,34,39  | 1   | PASS   | 93          | 0             | 0         | 0               | 1197 | 1197          |
| 34,38,39  | 1   | PASS   | 93          | 0             | 0         | 0               | 1197 | 1197          |

All 10 videos pass the luminance/finite scan (scan_black.py -> "ok").

## Files (working area)
- `/workspace/tmp_yi_yiwan/sweep_fb2_runs/` — 6 mp4 + 6 logs + HANDOFF.md
- `/workspace/tmp_yi_yiwan/sweep_fb3_runs/` — 4 mp4 + 4 logs + HANDOFF.md
- TSVs/driver logs: `/workspace/tmp_yi_yiwan/sweep_fb{2,3}/`
- Scripts: `/workspace/vllm-omni-inner/wan/{run_wan17_sagev3_fb_sweep_newbuild.sh,
  sweep_fb2_driver2.sh, sweep_fb3_driver.sh}`

## Reproduce (one config / full sweep)
```bash
WAN_SAGE_FALLBACK_BLOCKS="33,34" bash \
  /workspace/vllm-omni-inner/wan/run_wan17_sagev3_fb_sweep_newbuild.sh 40 81 0
# exit report: grep "Sage attention V3 call counts" <log>

docker exec -d yi-wan bash -c 'nohup bash /workspace/vllm-omni-inner/wan/sweep_fb3_driver.sh 0 "33,34,38" "33,38,39" > /workspace/tmp_yi_yiwan/sweep_fb3/driver_xpu0.out 2>&1 &'
```

## Pitfalls (learned)
- mp4 name != log name: runner writes `wan22_output_<stem>.mp4` vs
  `<stem>.log`; deriving the video from the log needs the `wan22_output_`
  prefix (driver v1 bug -> false FAILs + wasted retries)
- killing a sweep driver orphans the in-flight text_to_video.py + renamed
  `vLLM-Omni::DiffusionWorker` children; kill via /proc/PID/environ
  ZE_AFFINITY_MASK, not cmdline grep
- node D93-C rebooted 5x on the evening of 09-12; after any reboot:
  `docker start yi-wan`, then relaunch drivers (resume logic reuses
  finished artifacts)

## Conclusion / next steps
On f200d765, every pinned subset of size 2 or 3 left the freed blocks on
the quantized kernel with zero runtime fallbacks / non-finite rescues /
layout fallbacks. The 4-block pin is not required for numerical stability
on this build. Next probe: 0-block (full sage) run; if it passes, drop the
selective fallback from the production recipe (re-verify after the lse
rebuild, which is still mid-rebase in deepklox-sage3).
