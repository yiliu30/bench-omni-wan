# Lightning VBench-I2V: 50-pair exploratory comparison

This campaign selects the **first 50 `imaging_quality` entries in upstream
JSON order**, not a balanced or random sample. The entries are 8 abstract,
40 architecture, and 2 indoor images. Each pair receives one BF16/Flash
video and one MXFP4-weight/Sage V3 Hybrid video with forced Flash fallback
on self-attention blocks `0,33,34,38`. Cross-attention uses SDPA in both.
VBench's recommended five videos per image–prompt pair are **not** generated.

Upstream metadata: [VBench-I2V](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v),
commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`,
SHA-256 `8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8`.
The original input JPEGs are the official `crop/1-1` files in the
[VBench Drive folder](https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx).
Their individual Drive IDs and digests are in
`/home/yiliu7/vbench_i2v/data/official_crop/source.json`; inputs and model
weights are not redistributed here.

Both recipes use Wan2.2-I2V-A14B-Lightning, square 640x640 output,
81 frames/16 FPS, 4 Euler steps, guidance 1.0, boundary ratio 0.9,
flow shift 1.0, seed 0, empty negative prompt, no cache, and the unmodified
upstream Wan VAE decoder (`VLLM_OMNI_WAN_VAE_FAST_DECODE=0`). The opt-in
fast decoder repeatedly reset physical XPU 2 during pair 10's BF16 decode;
the upstream decoder successfully decoded that same pair on XPU 2. Its
partial run and failure logs remain under `results/i2v_lightning_50/`
and are excluded from this consistently configured campaign. Deterministic
assignment is `pair index % 4` to physical XPUs 0–3, with both recipes of
each pair on the same XPU. MXFP4 applies online to **model weights and
activations**, not to the Sage attention variant; the candidate uses
Hybrid attention and `--enforce-eager`.

Local output directory: [`results/i2v_lightning_50_upstream_vae/`](results/i2v_lightning_50_upstream_vae/).
`manifest.json` freezes prompts, crop digests and device assignments;
`runtime.json` freezes code revisions, portable MXFP4 code digest, XPU UUIDs,
and common settings. The two separate env files
[`i2v_lightning_bf16.env`](i2v_lightning_bf16.env) and
[`i2v_lightning_sage.env`](i2v_lightning_sage.env) declare both complete
recipes; `recipe_envs.json` pins their settings and SHA-256 digests for
subsequent resumes. Each recipe has its own video, per-pair record and
per-device server log. `smoke.json` attests the first validated pair and
backend routing. `comparison/` contains matched frame sheets at
0/20/40/60/80, decoded-pixel per-frame and per-pair MAE/SSIM CSVs, and
`summary.json`. These similarity measurements are **not** perceptual image
quality scores and do not rank the models' quality.

**Completed:** 50/50 validated BF16/Flash videos and 50/50 validated
MXFP4/Sage+Flash videos, each 81 decoded 640x640 frames at 16 FPS.
The full comparison contains 250 contact sheets and 4,050 per-frame
rows. Against the paired BF16 videos, MXFP4 has **14.47/255 mean
decoded-pixel MAE** and **0.685 mean SSIM** over all 4,050 matched
frames. These are descriptive similarities, **not** VBench image-quality
scores or evidence that one recipe looks better. See
[pair 000, frame 40](results/i2v_lightning_50_upstream_vae/comparison/pair_000_frame_040.png),
[pair 010, frame 40](results/i2v_lightning_50_upstream_vae/comparison/pair_010_frame_040.png),
and [pair 049, frame 40](results/i2v_lightning_50_upstream_vae/comparison/pair_049_frame_040.png);
per-pair values are in
[`comparison/per_pair.csv`](results/i2v_lightning_50_upstream_vae/comparison/per_pair.csv).

All four physical XPUs contributed to both recipes. Each final server
log verifies BF16's forced Flash self-attention or MXFP4's Sage Hybrid
plus four forced Flash blocks, SDPA cross-attention, zero nonfinite
fallbacks, and upstream VAE decode. XPU 2 reset during pair 42's
MXFP4 VAE encode; its recorded failure was retried on **the same XPU**
after an engine restart, and both remaining videos completed.
The final videos/records are complete; failed-attempt records and logs
are retained for diagnosis. The initial generation used runner SHA-256
`d27b36fcc8f7a0b2062bb709bc406cdd459dd4fd89e649ae2ba740c76b069350`,
while the recipe-env-aware runner added afterward has SHA-256
`5bc1ed3c2945e8a7b3ebdaf0a2b96a520e92c24f006f87895ac7778042c41451`.
Resuming with the two new env files reused all 100 validated videos
without model reload or regeneration. The comparison source SHA-256 is
`9c79fccf5486239ea40522d8407349970e1a553405896cb7873bc972fef766d7`,
and the official-crop downloader SHA-256 is
`57c735f5719d9fe6e69e26622bfe92fd359807836be64c22babce736b783d0a7`.

The separate [MXFP4/Sage no-forced-block-fallback arm](results/i2v_lightning_50_no_fallback/)
generated another **50 validated 81-frame videos** on the same 50
inputs and device assignments. Its final logs record 0 forced
self-attention calls, 0 Flash fallback calls, positive Sage Hybrid
self-attention calls, and SDPA cross-attention on all four XPUs.
One XPU-2 attempt at pair 10 had near-black frames; its failed video
and record are retained separately, and an identical same-device retry
passed validation. All 4,050 frames in the final outputs were checked
for black/flat frames. The no-fallback arm is **not included** in the
two-recipe MAE/SSIM figures above; its XPU MUSIQ-SPAQ score appears below. Its
recipe is in [`i2v_lightning_sage_no_fallback.env`](i2v_lightning_sage_no_fallback.env);
the generating runner SHA-256 is
`256a80893c3e546da8cb22ed613f4f48946a04871d1fd6d1ad774a8fbaeab3af`.

Reproduce or resume with the commands in [README.md](README.md). The
four-device exposure and read-only model bind mount in `yi-wan` must be
rechecked after any container restart. The campaign is complete only after
exactly **50 validated videos per recipe**, matched manifests, backend
routing counters, and all comparison outputs have been confirmed.

For an official full-suite benchmark submission, a later evaluation would
need the full prompt suite and five generated videos per pair; these
50-pair/one-sample results are explicitly exploratory.
Unlike the I2V consistency dimensions, `imaging_quality` does **not** read
or compare the input image or prompt: it scores decoded frames with MUSIQ-SPAQ,
averages all frames per video, then averages videos and divides by 100.
The upstream `--mode custom_input` accepts our `000.mp4` through `049.mp4`
filenames directly without a `custom_image_folder` for this dimension.
On a separate CUDA-capable VBench environment with the required
dependencies, the upstream entry point is:

```bash
cd /home/yiliu7/vbench_i2v/VBench
python evaluate_i2v.py --dimension imaging_quality --mode custom_input \
  --ratio 1-1 \
  --videos_path /home/yiliu7/bench-omni-wan/results/i2v_lightning_50_upstream_vae/bf16_flash \
  --output_path /path/to/bf16_scoring
# Repeat with mxfp4_sage_flash4 and a different output_path.
```

The upstream entry point hardcodes CUDA, so this command is **not runnable
in the current XPU-only `yi-wan` environment** without adapting the
evaluation dependencies. The three completed arms were instead scored
on XPU with VBench's MUSIQ-SPAQ model, the upstream checkpoint SHA-256
`358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50`,
and its exact default `longer` preprocessing:

See [the dedicated imaging-quality results](I2V_IMAGING_QUALITY_RESULTS.md)
for the aggregate comparison, interpretation, limitations and data files.

| Recipe | Mean raw MUSIQ-SPAQ | VBench-normalized `/100` |
|---|---:|---:|
| BF16 Flash | 72.8411 | 0.728411 |
| MXFP4 Sage + four Flash fallback blocks | 72.3147 | 0.723147 |
| MXFP4 Sage without forced block fallback | 71.2530 | 0.712530 |

These are **50-pair subset scores with one sample per pair**, not official
355-pair/five-sample VBench benchmark submissions. Every frame of every
video was scored (12,150 frames total); each video averages 81 MUSIQ-SPAQ
frame scores, and the 50 video means are averaged and divided by 100.
The BF16-to-four-block difference is 0.005264 normalized, and the
BF16-to-no-forced-block difference is 0.015881; these are descriptive
scores on this ordered subset. The model measures technical appearance
without checking image fidelity or prompt adherence. The separate
pixel MAE/SSIM is not MUSIQ-SPAQ.

Per-video scores are in
[`per_video.csv`](results/i2v_imaging_quality_xpu/per_video.csv), all
per-frame scores in recipe-specific `000.json`–`049.json` files under
[`results/i2v_imaging_quality_xpu/`](results/i2v_imaging_quality_xpu/),
and aggregate scores in
[`summary.json`](results/i2v_imaging_quality_xpu/summary.json).
`provenance.json` pins the scorer digest, weight digest, model package
versions, manifest digests and XPU identities. `recovery.json` records
that repeated XPU-2 device resets required moving only the final ten
**scoring jobs** to XPU 3; no videos were regenerated.
