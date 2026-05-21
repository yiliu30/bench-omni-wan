#!/usr/bin/env bash

set -euo pipefail

vbench_dir=/home/yiliu7/workspace/VBench
python_bin=/home/yiliu7/workspace/venvs/vbench/bin/python
seeded_eval_py=/home/yiliu7/workspace/vllm-omni/bench-wan/vbench_eval.py
video_dir=/home/yiliu7/workspace/vllm-omni/bench-wan/output/default_fp8_linear
video_dir=/home/yiliu7/workspace/vllm-omni/bench-wan/output/default_bf16
eval_seed=${EVAL_SEED:-42}
cuda_devices=${CUDA_VISIBLE_DEVICES:-5,6}

# video_dir=/mnt/ctrl/disk2/yiliu7/wan-res/default_real_sage3_attn_only
# video_dir=/home/yiliu7/workspace/vllm-omni/bench-wan/output/default_mxfp4_linear_only

# Custom-input dimensions supported by VBench for this filename-based prompt set.
# Exclude human_action here: it expects filenames that map cleanly to human action
# labels, which these mixed subject/vehicle/animal prompts do not provide.
dims=(
  subject_consistency
  background_consistency
  temporal_flickering
  motion_smoothness
  aesthetic_quality
  imaging_quality
  overall_consistency
  temporal_style
)

test_basename=$(basename "$video_dir")
test_name="${test_basename}_custom_input_supported"
test_out_dir=./final-score/"$test_name"

mkdir -p "$test_out_dir"

PYTHONHASHSEED="$eval_seed" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES="$cuda_devices" \
"$python_bin" "$seeded_eval_py" \
    --seed "$eval_seed" \
    -- \
    --videos_path "$video_dir" \
    --dimension "${dims[@]}" \
    --mode=custom_input \
    --output_path "$test_out_dir"
