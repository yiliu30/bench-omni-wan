# Wan2.2 Lightning I2V: XPU imaging-quality results

**Result:** BF16 Flash has the highest mean MUSIQ-SPAQ score on this
50-image-prompt subset, followed by MXFP4 Sage Hybrid with four forced Flash
fallback blocks, then MXFP4 Sage Hybrid without forced fallback. These are
exploratory subset measurements, **not official full-suite VBench-I2V scores**.

| Recipe | Valid videos | Mean raw MUSIQ-SPAQ | VBench-normalized `/100` | Difference from BF16 |
|---|---:|---:|---:|---:|
| BF16 Flash | 50 | 72.8411 | 0.728411 | baseline |
| Sage V3, Flash blocks `0,33,34,38`, MXFP4 Linear | 50 | 72.3147 | 0.723147 | -0.005264 |
| Sage V3, no forced fallback, MXFP4 Linear | 50 | 71.2530 | 0.712530 | -0.015881 |

Here **MXFP4 refers to online quantization of model linear-layer weights
and activations for linear GEMM** (`--quantization mxfp4`), not MXFP4
attention. Sage V3 Hybrid handles self-attention separately; the two
MXFP4 recipes differ only in whether blocks `0,33,34,38` are forced to
use Flash fallback. Cross-attention uses SDPA in both.

The four-block recipe scored above BF16 on 21 of 50 individual pairs;
the no-forced-fallback recipe did so on 18 of 50. These pair counts and
mean differences are descriptive, not a significance test.

## What was measured

The source is the first 50 `imaging_quality` entries in upstream VBench-I2V
metadata order: 8 abstract, 40 architecture, and 2 indoor pairs, each with
its official 1:1 input crop and the same prompt and seed across recipes.
All three recipes used Wan2.2-I2V-A14B-Lightning with the upstream VAE
decoder, 4 Euler steps, guidance 1.0, 81 frames at 640x640 and 16 FPS.
Generation ran on physical XPUs 0-3, with one video per pair per recipe.
See [campaign setup and generation results](I2V_LIGHTNING_50.md) for
the remaining settings, failure history, routing checks, and source
provenance. The incomplete fast-VAE output in `results/i2v_lightning_50/`
was excluded.

Scoring applies VBench's `imaging_quality` MUSIQ-SPAQ checkpoint and default
`longer` preprocessing (512x512) to **every decoded frame**. It averages the
81 frame scores for each video, averages the 50 video means per recipe,
then divides by 100 for the normalized score. All 150 validated videos
and 12,150 frames were scored on XPU. The upstream evaluation CLI hardcodes
CUDA; the XPU scorer follows its method but does not claim to have run that
CLI unchanged. The checkpoint SHA-256 is
`358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50`.

This metric judges technical visual appearance; it **does not compare** a
video to its input image or prompt. It cannot establish image fidelity,
prompt adherence, or which video people prefer. Nor do the separate
BF16-versus-four-block decoded-pixel MAE/SSIM numbers in the campaign
index measure perceptual quality. The ordered subset is heavily weighted
toward architecture and uses **one** sample per pair, rather than the full
355-pair/five-video protocol. Do not generalize these small score differences
to the full benchmark.

## Data and reproducibility

- [`summary.json`](results/i2v_imaging_quality_xpu/summary.json) holds full-precision
  means; [`per_video.csv`](results/i2v_imaging_quality_xpu/per_video.csv)
  contains all 150 pair/recipe scores and source-video SHA-256 hashes.
  Recipe-specific `000.json`-`049.json` records under
  [`results/i2v_imaging_quality_xpu/`](results/i2v_imaging_quality_xpu/)
  contain all per-frame scores.
- [`provenance.json`](results/i2v_imaging_quality_xpu/provenance.json) pins
  the scorer, checkpoint, manifests, package versions, and XPU identities.
  The scoring implementation is
  [`score_i2v_imaging_quality_xpu.py`](score_i2v_imaging_quality_xpu.py).
  Detailed execution instructions are in [README.md](README.md#xpu-imaging-quality-scoring).
- XPU 2 reset twice during scoring. The remaining ten scoring jobs moved
  to XPU 3, as recorded in
  [`recovery.json`](results/i2v_imaging_quality_xpu/recovery.json).
  The source videos, their generation-device assignments, model weights,
  and preprocessing were unchanged; no video was regenerated.

The `results/` artifacts are local and gitignored. The table above records
the aggregate values in this document so the conclusion remains visible
without access to those files.

## Visual comparison

[Three-column contact sheet](results/i2v_three_way_frames/overview.png):
five matched pairs (`000`, `010`, `020`, `030`, `049`), with frames 20 and
60 from each video. Each of the ten rows is **BF16 Flash | MXFP4 Sage with
four fallback blocks | MXFP4 Sage without forced fallback**, at full
640x640 frame resolution per column. Individual rows are saved as
`pair_NNN_frame_FFF.png` under
[`results/i2v_three_way_frames/`](results/i2v_three_way_frames/).
The first selected pair is abstract, the middle three are architecture,
and the last is indoor; these five examples are illustrative, not
representative of the full subset. Recreate the images in `yi-wan` with
`/opt/gfx-deps/venv/bin/python compare_i2v_three_way_frames.py`.
