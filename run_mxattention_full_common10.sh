#!/usr/bin/env bash
# Generate the common 10 VBench subject-consistency clips with MXAttention full,
# then run the custom-input VBench dimensions after the TP server exits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/mxattention_full_common10.env}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/subject_consistency.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/mxattention_full_common10}"
VIDEOS_DIR="$OUTPUT_ROOT/videos"
VBENCH_DIR="${VBENCH_DIR:-$SCRIPT_DIR/../VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-$SCRIPT_DIR/../venvs/vbench/bin/python}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-1}"
PLUGIN_ROOT="${PLUGIN_ROOT:-/dev/shm/.tmp_yi/workspace/vllm-qdq-plugin}"
OMNI_ROOT="${OMNI_ROOT:-/dev/shm/.tmp_yi/workspace/omni-wan}"

for file in "$ENV_FILE" "$PROMPT_FILE" "$PLUGIN_ROOT/src/vllm_qdq_plugin" "$OMNI_ROOT/.venv/bin/python"; do
  [[ -e "$file" ]] || { echo "Missing prerequisite: $file" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT"
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    PYTHONPATH="$SCRIPT_DIR/wan_flash_only_shim:$PLUGIN_ROOT/src:$OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$OMNI_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate.py" "$ENV_FILE" \
        --prompt-file "$PROMPT_FILE" \
        --num-prompts 10 \
        --output-dir "$VIDEOS_DIR" \
        --server-log "$OUTPUT_ROOT/server.log" \
        "$@"
    exit 0
  fi
done

if [[ "${SKIP_GENERATION:-0}" != "1" ]]; then
  PYTHONPATH="$SCRIPT_DIR/wan_flash_only_shim:$PLUGIN_ROOT/src:$OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$OMNI_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate.py" "$ENV_FILE" \
      --prompt-file "$PROMPT_FILE" \
      --num-prompts 10 \
      --output-dir "$VIDEOS_DIR" \
      --server-log "$OUTPUT_ROOT/server.log" \
      "$@"
fi

"$SCRIPT_DIR/verify_mxattention_full_common10.py" \
  --video-dir "$VIDEOS_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --server-log "$OUTPUT_ROOT/server.log"

if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  VBENCH_DIR="$VBENCH_DIR" VBENCH_PYTHON="$VBENCH_PYTHON" \
    "$SCRIPT_DIR/vbench_eval_folder.py" "$VIDEOS_DIR" \
      --cuda "$EVAL_CUDA_DEVICE" \
      --prompt-file "$OUTPUT_ROOT/prompts.json" \
      --output-dir "$OUTPUT_ROOT/vbench"
fi
