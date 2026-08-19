# MXAttention Mode Comparison

This note compares the MXAttention modes used in the Wan sweep and adds
Sage3 `mxfp4_hw` as the closest non-MXAttention code path. The video metrics
compare the available MXAttention outputs against the BF16 baseline video
`wan_bf16_t2v_out.mp4`.

## What Each Mode Does

| Mode | Q/K/V quantization | P quantization | Hadamard | Main intent |
|---|---|---|---|---|
| `ocp_mxfp4_direct` | OCP MXFP4, `qmax=6.0` | Direct MXFP4 probability quantization | No | Raw direct MXFP4 hardware baseline |
| `uos_only` | OCP MXFP4 with UOS, default `qmax=7.25` | UOS-scaled MXFP4 probability quantization | No | Isolate UOS scaling without PNQ or Hadamard |
| `pnq_only` | OCP MXFP4, `qmax=6.0` | PNQ probability normalization without UOS | No | Isolate PNQ without UOS or Hadamard |
| `hadamard_only` | OCP MXFP4, `qmax=6.0` | Direct MXFP4 probability quantization | Yes | Isolate Hadamard without UOS or PNQ |
| `uos_pnq` | OCP MXFP4 with UOS, default `qmax=7.25` | UOS plus PNQ probability normalization | No | Isolate PNQ on top of UOS |
| `uos_hadamard` | OCP MXFP4 with UOS, default `qmax=7.25` | UOS-scaled MXFP4 probability quantization | Yes | Isolate Hadamard on top of UOS |
| `mxattention_full` | OCP MXFP4 with UOS, default `qmax=7.25` | UOS plus PNQ probability normalization | Yes | Full MXAttention path: Hadamard, UOS, and PNQ |
| Sage3 `mxfp4_hw` | MXFP4, fixed E2M1 max `6.0` | Direct MXFP4 probability quantization | No | Sage3 hardware MXFP4 path with QK smoothing |

All MXAttention modes use the same hardware MXFP4 attention wrapper. The
mode switch selects the effective quantization max, whether PNQ is enabled, and
whether Q/K receive the normalized FWHT Hadamard transform before quantization.
Sage3 `mxfp4_hw` uses the same underlying direct MXFP4 kernel family, but enters
through the Sage3 attention stack and applies QK smoothing by default.

## Code Path Summary

The mode selection is implemented in `vllm_qdq_plugin/mxattention/api.py`:

| Mode | Effective settings |
|---|---|
| `ocp_mxfp4_direct` | `effective_qmax=6.0`, `pnq=False`, `hadamard=False` |
| `uos_only` | `effective_qmax=MXATTENTION_QMAX`, `pnq=False`, `hadamard=False` |
| `pnq_only` | `effective_qmax=6.0`, `pnq=True`, `hadamard=False` |
| `hadamard_only` | `effective_qmax=6.0`, `pnq=False`, `hadamard=True` |
| `uos_pnq` | `effective_qmax=MXATTENTION_QMAX`, `pnq=True`, `hadamard=False` |
| `uos_hadamard` | `effective_qmax=MXATTENTION_QMAX`, `pnq=False`, `hadamard=True` |
| `mxattention_full` | `effective_qmax=MXATTENTION_QMAX`, `pnq=True`, `hadamard=MXATTENTION_USE_HADAMARD` |
| Sage3 `mxfp4_hw` | fixed E2M1 max `6.0`, `pnq=False`, no Hadamard, QK smoothing enabled unless `SAGE3_DISABLE_PER_BLOCK_MEAN=1` |

Inputs are quantized by `quantize_mxfp4_uos()`, padded to the kernel tile
requirements, then dispatched to:

| PNQ | Kernel call |
|---|---|
| Disabled | `mxfp4_flash_attention()` |
| Enabled | `mxfp4_pnq_flash_attention()` |

Compared with Sage3 `mxfp4_hw`, `ocp_mxfp4_direct` is the closest MXAttention
mode because both use direct MXFP4 and no PNQ. Sage3 `mxfp4_hw` additionally
applies QK smoothing and passes `delta_s` correction by default. For the
closest apples-to-apples kernel comparison, run Sage3 `mxfp4_hw` with
`SAGE3_DISABLE_PER_BLOCK_MEAN=1`.

## MXAttention vs Sage3 `mxfp4_hw`

| Aspect | MXAttention `ocp_mxfp4_direct` | Sage3 `mxfp4_hw` |
|---|---|---|
| Backend gate | `VLLM_MXATTENTION=1` | `VLLM_SAGE3_TRITON=1` |
| Mode/config | `MXATTENTION_MODE=ocp_mxfp4_direct` | `SAGE3_QUANT_FORMAT=mxfp4_hw` |
| Q/K/V quantizer | `quantize_mxfp4_uos(..., qmax=6.0)` | `quantize_to_mxfp4()` |
| Attention kernel | `mxfp4_flash_attention()` | `mxfp4_flash_attention()` |
| Probability path | direct MXFP4 P quantization | direct MXFP4 P quantization |
| QK smoothing | No | Yes by default |
| `delta_s` correction | No | Yes when smoothing is active |
| Fallback behavior | catches hardware failure and falls back to SDPA | expects Sage3 kernel path to work |

## BF16 Video Diff Results

Baseline:

`wan_bf16_t2v_out.mp4`

Candidate videos:

| Mode | Video |
|---|---|
| `ocp_mxfp4_direct` | `wan_t2v_a14b_ocp_mxfp4_direct_after_review_fix_1280x720_81f_40steps.mp4` |
| `uos_only` | `wan_t2v_a14b_uos_only_after_review_fix_1280x720_81f_40steps.mp4` |
| `pnq_only` | `wan_t2v_a14b_pnq_only_1280x720_81f_40steps.mp4` |
| `hadamard_only` | `wan_t2v_a14b_hadamard_only_1280x720_81f_40steps.mp4` |
| `uos_pnq` | `wan_t2v_a14b_uos_pnq_1280x720_81f_40steps.mp4` |
| `uos_hadamard` | `wan_t2v_a14b_uos_hadamard_1280x720_81f_40steps.mp4` |
| `mxattention_full` | `wan_t2v_a14b_mxattention_full_after_review_fix_1280x720_81f_40steps.mp4` |
| Sage3 `mxfp4_hw` | `wan_t2v_a14b_mxfp4_hw_attn_1280x720_81f_40steps.mp4` |

Metrics are decoded-video metrics over 81 frames at 1280x720 and 16 fps.
Higher PSNR and SSIM are better. Lower LPIPS, MAE, and MSE are better.

| Mode | PSNR dB | SSIM | LPIPS | MAE | MSE |
|---|---:|---:|---:|---:|---:|
| `ocp_mxfp4_direct` | 12.1748 | 0.522198 | 0.423660 | 0.149470 | 0.061137 |
| `uos_only` | 12.2266 | 0.543362 | 0.420311 | 0.146941 | 0.060420 |
| `pnq_only` | 13.1408 | 0.647583 | 0.389023 | 0.126260 | 0.048797 |
| `hadamard_only` | 11.8759 | 0.337470 | 0.422294 | 0.160352 | 0.065412 |
| `uos_pnq` | 16.9244 | 0.733163 | 0.272667 | 0.070590 | 0.021243 |
| `uos_hadamard` | 11.8983 | 0.332207 | 0.420684 | 0.160471 | 0.065209 |
| `mxattention_full` | 17.4825 | 0.757471 | 0.222936 | 0.063539 | 0.018635 |
| Sage3 `mxfp4_hw` | 11.8395 | 0.315158 | 0.420188 | 0.162548 | 0.065968 |

## Takeaways

### Ranking

The ranking is consistent across the distortion metrics:

| Rank | PSNR / SSIM / MAE / MSE order | LPIPS order |
|---:|---|---|
| 1 | `mxattention_full` | `mxattention_full` |
| 2 | `uos_pnq` | `uos_pnq` |
| 3 | `pnq_only` | `pnq_only` |
| 4 | `uos_only` | Sage3 `mxfp4_hw` |
| 5 | `ocp_mxfp4_direct` | `uos_only` |
| 6 | `uos_hadamard` or `hadamard_only` | `uos_hadamard` |
| 7 | `hadamard_only` or `uos_hadamard` | `hadamard_only` |
| 8 | Sage3 `mxfp4_hw` | `ocp_mxfp4_direct` |

`mxattention_full` is closest to BF16 by every measured metric. It improves
PSNR by 5.31 dB over the direct MXAttention path, raises SSIM by 0.235, cuts
LPIPS by 47.4%, cuts MAE by 57.5%, and cuts MSE by 69.5%.

`uos_only` is only slightly better than `ocp_mxfp4_direct` in the video-level
metrics. In this Wan run, UOS alone helps a little. `pnq_only` is materially
better than both direct modes, but adding UOS to PNQ recovers much more quality
(`uos_pnq` reaches 16.92 dB PSNR and 0.733 SSIM).

Hadamard without PNQ does not help this Wan sample. Both `hadamard_only` and
`uos_hadamard` are worse than their non-Hadamard counterparts on PSNR and SSIM
and roughly tied on LPIPS. Hadamard still helps in the full path when combined
with PNQ, where `mxattention_full` improves over `uos_pnq` on all reported
metrics.

Sage3 `mxfp4_hw` should be treated as a related direct MXFP4 hardware baseline,
not as one of the MXAttention modes. Its default QK smoothing means it is not
identical to `ocp_mxfp4_direct` unless smoothing is disabled. In this generated
Wan sample it has lower PSNR and SSIM than all MXAttention modes, while its
LPIPS is close to the direct and Hadamard-only variants.

### Component Effects

The following deltas use `ocp_mxfp4_direct` or the relevant lower-order mode as
the local baseline. Positive PSNR/SSIM is better; negative LPIPS/MAE/MSE is
better.

| Change | PSNR dB | SSIM | LPIPS | MAE | MSE |
|---|---:|---:|---:|---:|---:|
| UOS alone vs direct | +0.0518 | +0.021164 | -0.003349 | -0.002529 | -0.000717 |
| PNQ alone vs direct | +0.9660 | +0.125385 | -0.034637 | -0.023210 | -0.012340 |
| Hadamard alone vs direct | -0.2988 | -0.184728 | -0.001366 | +0.010882 | +0.004276 |
| PNQ on top of UOS | +4.6978 | +0.189801 | -0.147644 | -0.076351 | -0.039177 |
| Hadamard on top of UOS | -0.3282 | -0.211155 | +0.000374 | +0.013530 | +0.004789 |
| UOS on top of PNQ | +3.7836 | +0.085580 | -0.116356 | -0.055670 | -0.027554 |
| UOS on top of Hadamard | +0.0224 | -0.005263 | -0.001610 | +0.000119 | -0.000203 |
| Hadamard on top of UOS+PNQ | +0.5581 | +0.024308 | -0.049731 | -0.007050 | -0.002608 |
| Sage3 `mxfp4_hw` vs direct | -0.3353 | -0.207040 | -0.003472 | +0.013078 | +0.004832 |

The main signal is that PNQ and UOS are not additive in a simple linear way.
PNQ alone is useful, UOS alone is weak, but UOS plus PNQ is much stronger than
either component alone. `uos_pnq` reduces LPIPS by 29.9%, MAE by 44.1%, and MSE
by 56.5% relative to `pnq_only`.

Hadamard has a conditional effect. It is harmful without PNQ in this sample:
`hadamard_only` and `uos_hadamard` both lose about 0.3 dB PSNR against their
non-Hadamard counterparts and lose more than 0.18 SSIM against direct. With
PNQ active, Hadamard becomes beneficial: `mxattention_full` improves over
`uos_pnq` by 0.56 dB PSNR, 0.024 SSIM, 18.2% LPIPS, 10.0% MAE, and 12.3% MSE.

### Frame-Level Stability

The aggregate metric ordering is not coming from a few isolated frames. Against
`ocp_mxfp4_direct`, frame-level wins over 81 frames were:

| Mode | PSNR wins | SSIM wins | LPIPS wins |
|---|---:|---:|---:|
| `uos_only` | 40 / 81 | 77 / 81 | 40 / 81 |
| `pnq_only` | 77 / 81 | 67 / 81 | 71 / 81 |
| `hadamard_only` | 11 / 81 | 0 / 81 | 40 / 81 |
| `uos_pnq` | 81 / 81 | 80 / 81 | 81 / 81 |
| `uos_hadamard` | 11 / 81 | 0 / 81 | 36 / 81 |
| `mxattention_full` | 81 / 81 | 81 / 81 | 81 / 81 |
| Sage3 `mxfp4_hw` | 10 / 81 | 0 / 81 | 42 / 81 |

`uos_pnq` beats `pnq_only` on 81 / 81 frames for PSNR and LPIPS and 80 / 81
frames for SSIM. `mxattention_full` beats `uos_pnq` on 63 / 81 frames for PSNR,
74 / 81 frames for SSIM, and 81 / 81 frames for LPIPS. That makes the final
Hadamard gain smaller than the PNQ+UOS gain, but still broad rather than a
single-frame artifact.

### Interpretation

For this Wan prompt and seed, the attention degradation seems dominated by the
probability path and scale calibration rather than by Q/K rotation alone.
Direct MXFP4 and UOS-only are close, which suggests changing the Q/K/V dynamic
range without changing the probability quantization path is not enough.

PNQ is the first large improvement. It likely reduces error in the softmax
probability representation, which is where small attention-score perturbations
can become visible video-level changes. UOS then makes PNQ much more effective,
which suggests PNQ benefits from the wider or better-calibrated input scale
rather than acting independently.

Hadamard alone is not a quality fix here. It changes the Q/K distribution before
MXFP4 quantization, but without PNQ the probability path still uses the direct
MXFP4 kernel behavior and quality drops sharply by SSIM. In the full mode,
Hadamard appears to complement PNQ by improving the already-stabilized path.
The practical conclusion is to treat Hadamard as a full-path enhancer, not as a
standalone replacement for PNQ or UOS.

### Remaining Gaps

Two runs would make the ablation matrix cleaner:

| Missing control | Why it matters |
|---|---|
| `pnq_hadamard_only` with `qmax=6.0`, `pnq=True`, `hadamard=True` | Separates the Hadamard+PNQ interaction from UOS. The current sweep proves Hadamard helps on top of UOS+PNQ, but not whether it also helps on top of PNQ without UOS. |
| Sage3 `mxfp4_hw` with `SAGE3_DISABLE_PER_BLOCK_MEAN=1` | Removes Sage3 QK smoothing so the hardware baseline is closer to `ocp_mxfp4_direct`. The current Sage3 row is useful, but it is not an exact direct-kernel control. |

The current results are also single prompt / single seed. They are enough to
validate that the MXAttention modes are wired and to expose the strongest
component effects, but not enough for a final quality claim across prompts,
motion types, or seeds.

Detailed reports:

- `wan_mxattention_diff_report.csv`
- `wan_mxattention_diff_report.json`
