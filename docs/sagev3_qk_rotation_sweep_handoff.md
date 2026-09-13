# Sage V3 QK Hadamard rotation — 3-block fallback sweep (rotation ON) — handoff

2026-09-13, node D93-C, container `yi-wan`.

Tests the new optional Q/K Hadamard rotation in the Sage V3 Hybrid XPU
attention path (`SAGE_ATTN_QK_ROTATION`), applied to all four 3-block
fallback sets from {33,34,38,39}, and compared against the existing
rotation-OFF 3-block baselines (this folder's parent: `sweep_fb3_runs/`,
same kernel, same recipe).

## Purpose
The deepklox Sage V3 Hybrid kernel quantizes Q/K to int8/FP4 with per-token
scales. A few outlier channels in the head dim inflate those scales and
coarsen the rest of the block. A fixed orthogonal head-dim rotation
P (P P^T = I) leaves QK^T invariant in exact arithmetic but redistributes
head-dim energy (QuaRot/SpinQuant-style outlier mitigation). V and the
residual stream are untouched.

## What was added (uncommitted, env-gated, default OFF)
- `vllm_omni/diffusion/attention/backends/sage_attn3.py`:
  - env: `SAGE_ATTN_QK_ROTATION` (0/1, default 0),
    `SAGE_ATTN_QK_ROTATION_SEED` (default 42),
    `SAGE_ATTN_QK_ROTATION_DEBUG` (0/1, one-shot Q/K magnitude print).
  - P = Sylvester 128x128 Hadamard x seeded +/-1 diagonal, /sqrt(128), bf16,
    cached per (head_dim, dtype, device).
  - Applied in `forward_xpu` on the sage-kernel path only (after all
    fallback checks, before the deepklox call): `q = q @ P; k = k @ P`.
    Cross-attention, forced-SDPA blocks and SDPA fallbacks are untouched.
  - atexit report line now ends with `qk_rotation=<0|1>(seed=<n>)`.
- `wan/run_wan17_sagev3_fb_sweep_newbuild.sh`: passes through
  `WAN_SAGE_QK_ROTATION` / `WAN_SAGE_QK_ROTATION_SEED` /
  `WAN_SAGE_QK_ROTATION_DEBUG` and appends `_qkrot` to the artifact tag
  when enabled.

Math sanity (bf16 P, f32 checks): orthonormality err 2.1e-4; QK^T vs
unrotated f32 reference max abs diff 0.15 at score rms 11.4.

## Fixed recipe (identical to the fb3 baselines)
- Prompt: "Two anthropomorphic cats in comfy boxing gear and bright gloves
  fight intensely on a spotlighted stage." (+ standard negative prompt)
- 720x1280 @ 16fps, 81 frames, 40 steps, seed 42, guidance 4.0/3.0,
  boundary 0.875, flow shift 5.0
- MXFP8 linear quant + cache-dit, SAGE_ATTN_XPU_LAYOUT=auto (zero-copy NHD),
  fused prep, SAGE_ATTN_REPORT_FALLBACKS=1, SAGE_ATTN_QK_ROTATION=1 (seed 42)
- Kernel: /workspace/deepklox-sage3, .so md5 f200d765 (5-arg binding) via the
  compat shim on PYTHONPATH
- Runner: `WAN_SAGE_FALLBACK_BLOCKS="<set>" WAN_SAGE_QK_ROTATION=1
  WAN_SAGE_QK_ROTATION_DEBUG=1 bash run_wan17_sagev3_fb_sweep_newbuild.sh
  40 81 <xpu>`

## Runs (all 4/4 ok on the black scan; counters exact: forced_sdpa=93, sage=1197, cross_sdpa=1290, all other fallbacks 0, qk_rotation=1(seed=42))
| fb set   | xpu | run ts       | q max pre->post | k max pre->post |
|----------|-----|--------------|-----------------|-----------------|
| 33,34,38 | 0   | 134022       | 12.63 -> 6.50   | 9.50 -> 8.00    |
| 33,34,39 | 1   | 134022       | 12.88 -> 6.66   | 9.69 -> 8.00    |
| 33,38,39 | 0   | 141022       | 12.69 -> 6.16   | 9.63 -> 8.13    |
| 34,38,39 | 1   | 141022       | 12.69 -> 6.56   | 9.63 -> 8.06    |

RMS unchanged (q ~1.057, k ~1.059 pre/post, all runs). Q outliers roughly
halved; K outliers reduced ~15-17%. Smoke run (2 steps, 17f) preceded the
sweep: `sweep_qkrot_smoke.driver.log`.

## Comparison vs rotation-OFF baselines (compare_videos.py, 81 frames each)
| fb set   | ON artifact (this folder) ts | OFF baseline (`../sweep_fb3_runs/`) ts | PSNR dB | mean_abs_diff |
|----------|------------------------------|------------------------------------------|---------|---------------|
| 33,34,38 | 134022 (xpu0)                | 045746 (xpu0)                            | 16.03   | 21.72         |
| 33,34,39 | 134022 (xpu1)                | 045745 (xpu1)                            | 16.63   | 21.24         |
| 33,38,39 | 141022 (xpu0)                | 051527 (xpu0)                            | 16.22   | 21.51         |
| 34,38,39 | 141022 (xpu1)                | 051521 (xpu1)                            | 15.99   | 22.34         |

## Read
The rotation changes the generated video substantially (~16 dB PSNR vs the
rotation-OFF runs, consistently across all four fallback sets) — the Q/K
scale reconditioning moves the quantized-kernel output far more than
device/seed noise. Whether that is an improvement (vs the bf16 reference)
is the open question: PSNR vs the OFF runs measures *difference*, not
*quality*. Next step: judge against a bf16 (or dense93 bf16) reference, or
VMAF/visual inspection.

## Rerun
    cd /workspace/vllm-omni-inner/wan
    WAN_SAGE_FALLBACK_BLOCKS="33,34,38" WAN_SAGE_QK_ROTATION=1 \
    WAN_SAGE_QK_ROTATION_DEBUG=1 nohup bash \
      run_wan17_sagev3_fb_sweep_newbuild.sh 40 81 0 > /tmp/driver.log 2>&1 &

Black scan:
    /opt/gfx-deps/venv/bin/python3 /workspace/tmp_yi_yiwan/vbench_dense93/monitor/scan_black.py <mp4>

## Monitoring (standing)
`/workspace/tmp_yi_yiwan/monitor_wan/monitor_wan_black.sh` (container pid
tracked in `monitor_wan.pid`) black-scans every new/changed top-level mp4
in /workspace/tmp_yi_yiwan every 120s (stability gate 90s so in-flight
encodes are skipped). Log: `monitor_wan/monitor_wan_black.log`. BLACK
videos are logged (once per file version) but NOT moved. First full pass
(2026-09-13): 72 ok, 4 BLACK — old svdquant mxfp4 experimental artifacts
(`wan22_output_svdquant_mxfp4_r32_i1_b0_39_*`, `*_embedout_*`,
`_r32_i2_nofallback_*`, `_r128_i1_nofallback_*`), kept in place. Note: the
older dense93 monitor's BLACK case-pattern (`*"| BLACK")`) was also too
strict (scan line ends `| BLACK <<< BLACK`) — it never fired because
dense93 had zero blacks.

## Monitoring update (2026-09-13 14:5x)
`monitor_wan_black.sh` is now cron-resurrected: a root crontab entry
(`* * * * * /usr/bin/docker exec -d yi-wan bash /workspace/tmp_yi_yiwan/monitor_wan/monitor_wan_black.sh`)
starts it within 1 min of any crash or container restart. The script's
single-instance guard was hardened (pid must be alive AND its cmdline must
contain the script name, so a stale pidfile after a restart cannot block a
fresh start). Verified end-to-end twice (kill -> auto-restart at the next
minute tick). Residual gap CLOSED: yi-wan now has restart policy
`unless-stopped` (set 2026-09-13), so after a full node reboot the
container auto-starts and the cron line resurrects the monitor within
1 min. (Reversible: `docker update --restart no yi-wan`.)

## Monitoring alert-path verification (2026-09-13 15:0x)
End-to-end test of the BLACK alert path: a synthetic 5-frame black 720p
mp4 (`_test_black.mp4`, generator kept at
`/workspace/tmp_yi_yiwan/_make_test_black.py`) was dropped into the
watched dir; the monitor logged
`BLACK VIDEO: _test_black.mp4 (...)` on its next cycle and recorded it in
state (one alarm per file version, no re-alarm). Test file removed
afterwards. This test also caught a second pattern bug: the first "fix"
had only added state recording but kept the end-anchored case pattern
`*"| BLACK")`, which still can't match `... | BLACK <<< BLACK` (an
intermediate run was logged as UNKNOWN). Final case patterns:
`*ERROR*`, `*| BLACK*`, `*| ok)`, `*NO_FRAMES*` (one-shot, state-recorded),
else UNKNOWN (retried next cycle).
