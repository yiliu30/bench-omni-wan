# I2V Lightning 50-prompt subset with Sage V3 pure-MXFP4 (sageattn_v3)

Fourth accuracy arm: same 50 official crops / prompts / MXFP4 weights / upstream VAE /
seed / sampling as the paired campaign and the no-fallback arm, but the self-attention
kernel is the **pure-MXFP4** DeepKloX kernel (`deepklox.sageattn_v3`, Try77 port: E2M1
QK + online-E2M1 PV + native XMX denominator) instead of the hybrid variant. Blocks
0,33,34,38 keep the recipe's forced flash fallback, exactly like `mxfp4_sage_flash4`.

## Kernel side (deepklox-wan)

- Branch `port-sage-v3-pure-mxfp4` in `/home/yiliu7/deepklox-wan` (do NOT repoint the
  venv editable install; it stays on this checkout by project rule).
  - `ea885742` cherry-pick of deepklox-next `944d3600` (Try77 kernel as `csrc/sageattn/v3_pure/`,
    `build_v3_pure.sh`, setup.py flag `DEEPKLOX_SAGEATTN_V3`)
  - `75bb43f5` cherry-pick of deepklox-next `56644819` (`_C.sageattn_v3` binding,
    `deepklox.sageattn_v3` wrapper, tests, bench)
  - Conflicts: only setup.py (wan tree lacks the XMX-denominator commits) — resolved by
    keeping just the pure additions; `DEEPKLOX_SAGEATTN_V3_HYBRID_XMX_DENOM` stays
    absent from wan's tree (it is not a wan knob).
- Submodule `third-party/internal-sycl-tla` @ `4e53a3fd` — identical pointer in both
  checkouts; no submodule work needed.

### Build flags (superset of the previous wan build)

Old (confirmed from the pre-port `_C`: only `is_xe3p` + `sageattn_v3_hybrid` registered):

    DEEPKLOX_ENABLED_KERNELS=sageattn XPU_BUILD_WITH_XE3P=1
    DEEPKLOX_SAGEATTN_V3_HYBRID=1 DEEPKLOX_SAGEATTN_V1=0 DEEPKLOX_SAGEATTN_V2=0
    GPU_ARCHS=native (default)

New adds only `DEEPKLOX_SAGEATTN_V3=1`. Rebuild command (inside container `yi-wan`,
under `flock /home/yiliu7/scratch/build.lock`): `/home/yiliu7/scratch/build_wan_v3_port.sh`
(`setup.py build_ext --inplace`, same recipe as `build_v3_pure.sh`). Build log:
`/home/yiliu7/scratch/wan_v3_pure_build.log` (BUILD_EXIT=0, 2026-09-27 00:23 CST).

### Sanity gates (all green, 2026-09-27, ZE_AFFINITY_MASK=0, log `/home/yiliu7/scratch/gate_cd.log`)

- a) `_C.sageattn_v3_hybrid` True AND `_C.sageattn_v3` True.
- b) pure H1/S128 forward finite (absmax 1.4844, bf16 out HND).
- c) hybrid regression `tests/test_sageattn_v3_hybrid.py -q`: 12 passed, 30 skipped
  (skips = long/stress tiers + upstream FIXME — green short tier, 0 failures).
- d) pure suite `tests/test_sageattn_v3.py -q`: 27 passed, 9 skipped
  (long/stress tiers by design; matches the port commit's 27-pass short-tier count).

## Host reboot gotchas (2026-09-26 22:20 CST)

The host rebooted mid-setup. Two things that used to be true are NOT after a reboot:

1. `yi-wan` (privileged) now enumerates **all 8 XPUs**; before the reboot it saw only
   renderD128-131. `gpu_preflight` demands exactly 4, so run the driver with
   `ZE_AFFINITY_MASK=0,1,2,3` + `ZE_FLAT_DEVICE_HIERARCHY=FLAT`. Indices 0-3 are the
   same physical GPUs as the previous arms (UUIDs verified against
   `results/i2v_lightning_50_no_fallback/runtime.json`).
2. The model at `/mnt/disk1/hf_models/Wan2.2-I2V-A14B-Lightning-Diffusers` is NOT in the
   container's bind list (only `/data`, `/home/yiliu7`); the pre-reboot mount was manual
   and ephemeral. Restore (inside the privileged container, device node exists):

       mkdir -p /mnt/hostroot && mount /dev/nvme0n1p2 /mnt/hostroot
       mount --bind /mnt/hostroot/mnt/disk1/hf_models/Wan2.2-I2V-A14B-Lightning-Diffusers \
         /mnt/disk1/hf_models/Wan2.2-I2V-A14B-Lightning-Diffusers

   (shares the host root superblock; model load is read-only.)

## Harness wiring (bench-omni-wan)

`i2v_lightning_subset.py` previously could only run `hybrid` (recipe validation pinned
`SAGE_ATTN3_XPU_VARIANT=hybrid`, the server env was rebuilt from the canonical recipe
file, and routing verification matched "kernel variant: hybrid"). Added a fourth
standalone recipe `mxfp4_sage_v3_pure` (same pattern as `mxfp4_sage_no_fallback`):

- `i2v_lightning_sage_pure.env` = `i2v_lightning_sage.env` + `RECIPE=mxfp4_sage_v3_pure`
  + `FALLBACK_BLOCKS=0,33,34,38`, `SAGE_ATTN3_XPU_VARIANT=pure_mxfp4`,
  `OUTPUT=results/i2v_lightning_50_sage_v3_pure_mxfp4`. Forced blocks identical to the
  hybrid flash4 arm, so the arms are pixel-comparable except for the attention kernel.
- `recipe_config`/`config`/`main`/`serve_and_run`/`verify_routing` made recipe-aware
  (variant string + forced-block check); the paired and no-fallback arms validate
  unchanged (13/13 harness tests pass).
- Expected routing: `sage` counter > 0 on the 36 unforced self-attn blocks at
  seq=33600 (=64*525, 64-aligned, no alignment fallback); blocks 0,33,34,38 ->
  `forced_sdpa`->`fallback_flash`; cross-attn -> `cross_sdpa`; `sdpa_fallback` must be 0.

## Run / monitor

    sudo -n docker exec yi-wan bash /home/yiliu7/scratch/run_pure_preflight_smoke.sh   # smoke (1 video, XPU0)
    sudo -n docker exec -d yi-wan bash /home/yiliu7/scratch/run_pure_campaign.sh ...   # full 50

- No-build GPU note: subset uses XPUs 0-3; do not hold `/home/yiliu7/scratch/gpu.lock`.
- Campaign log: `results/i2v_lightning_50_sage_v3_pure_mxfp4.nohup.log`;
  per-server logs `.../mxfp4_sage_v3_pure/server_xpu{0..3}.log`.
- Progress: `ls results/i2v_lightning_50_sage_v3_pure_mxfp4/mxfp4_sage_v3_pure/[0-9][0-9][0-9].mp4 | wc -l`
- ETA: no_fallback arm median 51.5 s/video; 50 videos / 4 devices ≈ 12-14 min compute
  plus per-server model init (~15-25 min, overlapped) → expect the whole run well under
  ~1.5 h.
