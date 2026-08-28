#!/usr/bin/env bash
# Complete the shared 72-prompt run on GPUs 4 and 5.
# The existing GPU 2/3/6 run is intentionally left untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/mxattention_full72.env}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/subject_consistency.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/mxattention_full_common10}"
VIDEOS_DIR="$OUTPUT_ROOT/videos"
RUN_ROOT="$OUTPUT_ROOT/full72/gpu45"
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

# Re-scan at launch so this run only claims currently missing outputs.
: > "$LOG_DIR/missing.prompts"
for prompt in "${prompts[@]}"; do
  [[ -s "$VIDEOS_DIR/$prompt-0.mp4" ]] || printf '%s\n' "$prompt" >> "$LOG_DIR/missing.prompts"
done

mapfile -t missing < "$LOG_DIR/missing.prompts"
total="${#missing[@]}"
echo "GPU 4/5 completion run: $total missing prompts"
if [[ "$total" -eq 0 ]]; then
  echo "Nothing to generate."
  exit 0
fi

for shard in 0 1; do
  gpus=(4 5)
  gpu="${gpus[$shard]}"
  port=$((8106 + shard))
  shard_file="$LOG_DIR/gpu${gpu}.prompts"
  : > "$shard_file"
  for i in "${!missing[@]}"; do
    [[ $((i % 2)) -eq "$shard" ]] && printf '%s\n' "${missing[$i]}" >> "$shard_file"
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
echo "GPU 4/5 completion workers finished."
