#!/usr/bin/env bash
# Continue the shared 72-prompt run on GPUs 0,1,4,5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/mxattention_full72.env}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/subject_consistency.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/mxattention_full_common10}"
VIDEOS_DIR="$OUTPUT_ROOT/videos"
RUN_ROOT="$OUTPUT_ROOT/full72/gpu0145"
LOG_DIR="$RUN_ROOT/logs"
PLUGIN_ROOT="${PLUGIN_ROOT:-/dev/shm/.tmp_yi/workspace/vllm-qdq-plugin}"
OMNI_ROOT="${OMNI_ROOT:-/dev/shm/.tmp_yi/workspace/omni-wan}"
PIDS=()

for file in "$ENV_FILE" "$PROMPT_FILE" "$PLUGIN_ROOT/src/vllm_qdq_plugin" "$OMNI_ROOT/.venv/bin/python"; do
  [[ -e "$file" ]] || { echo "Missing prerequisite: $file" >&2; exit 2; }
done
mkdir -p "$VIDEOS_DIR" "$LOG_DIR"

cleanup() {
  for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM

mapfile -t prompts < <(awk 'NF {print}' "$PROMPT_FILE")
[[ "${#prompts[@]}" -eq 72 ]] || { echo "Expected 72 prompts, found ${#prompts[@]}" >&2; exit 2; }
: > "$LOG_DIR/missing.prompts"
for prompt in "${prompts[@]}"; do
  [[ -s "$VIDEOS_DIR/$prompt-0.mp4" ]] || printf '%s\n' "$prompt" >> "$LOG_DIR/missing.prompts"
done
mapfile -t missing < "$LOG_DIR/missing.prompts"
echo "GPU 0/1/4/5 completion run: ${#missing[@]} missing prompts"
[[ "${#missing[@]}" -gt 0 ]] || exit 0

gpus=(0 1 4 5)
for shard in 0 1 2 3; do
  gpu="${gpus[$shard]}"
  port=$((8110 + shard))
  shard_file="$LOG_DIR/gpu${gpu}.prompts"
  : > "$shard_file"
  for i in "${!missing[@]}"; do
    [[ $((i % 4)) -eq "$shard" ]] && printf '%s\n' "${missing[$i]}" >> "$shard_file"
  done
  count=$(wc -l < "$shard_file")
  echo "GPU $gpu: port $port, prompts $count"
  [[ "$count" -gt 0 ]] || continue
  PYTHONPATH="$SCRIPT_DIR/wan_flash_only_shim:$PLUGIN_ROOT/src:$OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$OMNI_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate.py" "$ENV_FILE" \
      --prompt-file "$shard_file" --num-prompts "$count" \
      --output-dir "$VIDEOS_DIR" \
      --server-log "$LOG_DIR/gpu${gpu}.server.log" \
      --cuda-devices "$gpu" --port "$port" --tp 1 \
      > "$LOG_DIR/gpu${gpu}.generate.log" 2>&1 &
  PIDS+=("$!")
done
wait
trap - EXIT INT TERM
echo "GPU 0/1/4/5 completion workers finished."
