#!/usr/bin/env bash
# Generate all 72 subject-consistency clips using four TP1 MXAttention servers.
# Existing clips in the common-10 output directory are reused.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/mxattention_full72.env}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/subject_consistency.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/mxattention_full_common10}"
VIDEOS_DIR="$OUTPUT_ROOT/videos"
FULL72_ROOT="$OUTPUT_ROOT/full72"
PLUGIN_ROOT="${PLUGIN_ROOT:-/dev/shm/.tmp_yi/workspace/vllm-qdq-plugin}"
OMNI_ROOT="${OMNI_ROOT:-/dev/shm/.tmp_yi/workspace/omni-wan}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-1}"
PIDS=()

for file in "$ENV_FILE" "$PROMPT_FILE" "$PLUGIN_ROOT/src/vllm_qdq_plugin" "$OMNI_ROOT/.venv/bin/python"; do
  [[ -e "$file" ]] || { echo "Missing prerequisite: $file" >&2; exit 2; }
done
mkdir -p "$VIDEOS_DIR" "$FULL72_ROOT/logs"

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM

mapfile -t ALL_PROMPTS < <(awk 'NF {print}' "$PROMPT_FILE")
[[ "${#ALL_PROMPTS[@]}" -eq 72 ]] || { echo "Expected 72 prompts, found ${#ALL_PROMPTS[@]}" >&2; exit 2; }

# Create four deterministic temporary prompt shards containing only missing clips.
for shard in 0 1 2 3; do
  : > "$FULL72_ROOT/logs/shard${shard}.prompts"
done
for i in "${!ALL_PROMPTS[@]}"; do
  prompt="${ALL_PROMPTS[$i]}"
  output="$VIDEOS_DIR/$prompt-0.mp4"
  if [[ ! -s "$output" ]]; then
    shard=$((i / 18))
    printf '%s\n' "$prompt" >> "$FULL72_ROOT/logs/shard${shard}.prompts"
  fi
done

gpus=(1 2 3 6)
for shard in 0 1 2 3; do
  gpu="${gpus[$shard]}"
  port=$((8102 + shard))
  prompt_count=$(wc -l < "$FULL72_ROOT/logs/shard${shard}.prompts")
  echo "Shard $shard: GPU $gpu, port $port, missing prompts $prompt_count"
  if [[ "$prompt_count" -eq 0 ]]; then
    continue
  fi
  PYTHONPATH="$SCRIPT_DIR/wan_flash_only_shim:$PLUGIN_ROOT/src:$OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$OMNI_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate.py" "$ENV_FILE" \
      --prompt-file "$FULL72_ROOT/logs/shard${shard}.prompts" \
      --num-prompts "$prompt_count" \
      --output-dir "$VIDEOS_DIR" \
      --server-log "$FULL72_ROOT/logs/shard${shard}.server.log" \
      --cuda-devices "$gpu" --port "$port" --tp 1 \
      > "$FULL72_ROOT/logs/shard${shard}.generate.log" 2>&1 &
  PIDS+=("$!")
done
wait
trap - EXIT INT TERM

"$SCRIPT_DIR/verify_mxattention_full72.py" \
  --video-dir "$VIDEOS_DIR" --prompt-file "$PROMPT_FILE" \
  --log-dir "$FULL72_ROOT/logs" --output-root "$FULL72_ROOT"

if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  VBENCH_DIR="${VBENCH_DIR:-$SCRIPT_DIR/VBench}"
  VBENCH_PYTHON="${VBENCH_PYTHON:-$OMNI_ROOT/.venv/bin/python}"
  PYTHONPATH="$SCRIPT_DIR/VBench${PYTHONPATH:+:$PYTHONPATH}" \
    "$OMNI_ROOT/.venv/bin/python" "$SCRIPT_DIR/vbench_eval_folder.py" \
      "$VIDEOS_DIR" --cuda "$EVAL_CUDA_DEVICE" \
      --vbench-dir "$VBENCH_DIR" --vbench-python "$VBENCH_PYTHON" \
      --prompt-file "$FULL72_ROOT/prompts.json" \
      --output-dir "$FULL72_ROOT/vbench"
fi
