#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=${BENCH_ROOT:-/dev/shm/.tmp_yi/workspace/bench-omni-wan}
OMNI_ROOT=${OMNI_ROOT:-/dev/shm/.tmp_yi/workspace/omni-wan}
PLUGIN_ROOT=${PLUGIN_ROOT:-/dev/shm/.tmp_yi/workspace/vllm-qdq-plugin}
MODEL=${MODEL:-/dev/shm/.tmp_yi/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers}
CUDA_DEVICES=${CUDA_DEVICES:-1,2}
OUTPUT=${1:-"$BENCH_ROOT/wan_t2v_a14b_mxfp4_hw_attn_1280x720_81f_40steps.mp4"}
LOG=${LOG:-"${OUTPUT%.mp4}.log"}

if [[ ! -d "$MODEL" ]]; then
  echo "Missing Wan model directory: $MODEL" >&2
  echo "Set MODEL=/path/to/Wan2.2-T2V-A14B-Diffusers and rerun." >&2
  exit 2
fi

if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists; refusing to overwrite: $OUTPUT" >&2
  exit 2
fi

cd "$OMNI_ROOT"

(
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  PYTHONPATH="$BENCH_ROOT/wan_flash_only_shim:$PLUGIN_ROOT/src:$OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  WAN_OMNI_INIT_TIMEOUT="${WAN_OMNI_INIT_TIMEOUT:-3600}" \
  WAN_OMNI_STAGE_INIT_TIMEOUT="${WAN_OMNI_STAGE_INIT_TIMEOUT:-3600}" \
  VLLM_SAGE3_TRITON=1 \
  DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN \
  SAGE3_QUANT_FORMAT=mxfp4_hw \
  "$OMNI_ROOT/.venv/bin/python" \
    "$OMNI_ROOT/examples/offline_inference/text_to_video/text_to_video.py" \
    --model "$MODEL" \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --num-inference-steps 40 \
    --guidance-scale 4.0 \
    --guidance-scale-high 3.0 \
    --flow-shift 5.0 \
    --fps 16 \
    --seed 42 \
    --tensor-parallel-size 2 \
    --output "$OUTPUT"
) 2>&1 | tee "$LOG"

test -s "$OUTPUT"
echo "Saved: $OUTPUT"
