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

`mxattention_full` is closest to BF16 by every measured metric. It improves
PSNR by about 5.3 dB over the direct path, raises SSIM from roughly 0.52 to
0.76, and cuts LPIPS from roughly 0.42 to 0.22.

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
Wan sample it has lower PSNR and SSIM than all three MXAttention modes, while
its LPIPS is close to the two direct MXAttention variants.

Detailed reports:

- `wan_mxattention_diff_report.csv`
- `wan_mxattention_diff_report.json`
