#!/usr/bin/env bash
# Run VBench no-reference evaluation on a directory of mp4 videos.
#
# Usage:
#   ./vbench_eval.sh <videos_dir> <prompt> [output_dir] [cuda_device]
#
#   videos_dir   Path to a directory containing .mp4 files (flat, no subdirs).
#   prompt        The text prompt shared by all videos, OR path to a prompts.json file.
#   output_dir    Where to write results (default: ./vbench_output/<dirname>).
#   cuda_device   GPU index (default: 0).
#
# Examples:
#   # Single prompt for all videos:
#   ./vbench_eval.sh /home/yiliu7/workspace/wan-res/route_sweep \
#     "Two anthropomorphic cats in comfy boxing gear fight on a stage."
#
#   # Different prompt per video — pass a prompts.json path:
#   ./vbench_eval.sh /home/yiliu7/workspace/wan-res/route_sweep \
#     /path/to/prompts.json
#
#   # Custom output dir and GPU:
#   ./vbench_eval.sh /path/to/videos "a prompt" /path/to/output 2
#
# Output:
#   <output_dir>/results_<timestamp>_eval_results.json   per-dimension scores
#   <output_dir>/results_<timestamp>_full_info.json      video→prompt mapping
#
# Dependencies (already installed):
#   venv:  /home/yiliu7/workspace/venvs/omni/
#   VBench:/home/yiliu7/workspace/VBench/
#   cached model weights: /home/yiliu7/.cache/vbench/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/yiliu7/workspace/venvs/omni/bin/python}"
VBENCH_EVAL="$SCRIPT_DIR/vbench_eval.py"

# ---- args ----
if [ $# -lt 2 ]; then
    echo "Usage: $0 <videos_dir> <prompt_or_prompts_json> [output_dir] [cuda_device]"
    echo ""
    echo "  videos_dir   Directory of .mp4 files"
    echo "  prompt        Shared prompt text for all videos, or path to a prompts.json"
    echo "  output_dir    Default: ./vbench_output/<dirname>"
    echo "  cuda_device   Default: 0"
    exit 1
fi

VIDEOS_DIR="$(realpath "$1")"
PROMPT_ARG="$2"
OUTPUT_DIR="${3:-$SCRIPT_DIR/vbench_output/$(basename "$VIDEOS_DIR")}"
CUDA_DEVICE="${4:-0}"

# ---- dims ----
DIMS=(
    subject_consistency
    background_consistency
    temporal_flickering
    motion_smoothness
    dynamic_degree
    aesthetic_quality
    imaging_quality
    overall_consistency
    temporal_style
)

# ---- validate ----
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "ERROR: videos_dir not found: $VIDEOS_DIR"
    exit 1
fi

N_MP4=$(find "$VIDEOS_DIR" -maxdepth 1 -name "*.mp4" | wc -l)
if [ "$N_MP4" -eq 0 ]; then
    echo "ERROR: no .mp4 files found in $VIDEOS_DIR"
    exit 1
fi
echo "Found $N_MP4 mp4 files in $VIDEOS_DIR"

# ---- prompt resolution ----
PROMPT_FILE=""
if [ -f "$PROMPT_ARG" ]; then
    # Already a JSON file — use directly
    PROMPT_FILE="$(realpath "$PROMPT_ARG")"
    echo "Using prompt file: $PROMPT_FILE"
else
    # Text prompt — generate a temp prompt file mapping every mp4 to the same prompt
    PROMPT_FILE="$(mktemp /tmp/vbench_prompts_XXXXXX.json)"
    echo "Generating prompt file for shared prompt: ${PROMPT_ARG:0:80}..."
    PROMPT_TEXT="$PROMPT_ARG" VIDEOS_DIR="$VIDEOS_DIR" PROMPT_FILE="$PROMPT_FILE" \
      "$PYTHON" -c "
import json, os
prompt = os.environ['PROMPT_TEXT']
vdir = os.environ['VIDEOS_DIR']
files = [f for f in os.listdir(vdir) if f.endswith('.mp4')]
json.dump({f: prompt for f in files}, open(os.environ['PROMPT_FILE'],'w'), indent=2)
print(f'Wrote {len(files)} entries')
"
fi

# ---- run ----
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "VBench evaluation"
echo "  videos:   $VIDEOS_DIR ($N_MP4 clips)"
echo "  prompt:   $PROMPT_FILE"
echo "  output:   $OUTPUT_DIR"
echo "  dims:     ${#DIMS[@]} (${DIMS[*]})"
echo "  cuda:     GPU $CUDA_DEVICE"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  "$PYTHON" -u "$VBENCH_EVAL" --seed 42 -- \
    --videos_path "$VIDEOS_DIR" \
    --dimension "${DIMS[@]}" \
    --mode custom_input \
    --prompt_file "$PROMPT_FILE" \
    --output_path "$OUTPUT_DIR"

# ---- results ----
echo ""
echo "Done. Results:"
ls -lh "$OUTPUT_DIR"/*_eval_results.json 2>/dev/null || echo "  (no results file — check output above for errors)"

# Clean up temp prompt file (only if we generated it)
if [ -z "${KEEP_PROMPT_FILE:-}" ]; then
    [ -f "$PROMPT_FILE" ] && rm -f "$PROMPT_FILE"
fi
