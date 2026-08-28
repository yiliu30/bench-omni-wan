#!/usr/bin/env bash
# Resume the MXAttention-full 72-prompt run on physical GPUs 1-6.
# Valid existing clips are never overwritten; only missing prompts are assigned.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/mxattention_full72.env}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/subject_consistency.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output/mxattention_full_common10}"
VIDEOS_DIR="$OUTPUT_ROOT/videos"
RUN_ROOT="$OUTPUT_ROOT/full72/gpu1to6"
LOG_DIR="$RUN_ROOT/logs"
PLUGIN_ROOT="${PLUGIN_ROOT:-/dev/shm/.tmp_yi/workspace/vllm-qdq-plugin}"
OMNI_ROOT="${OMNI_ROOT:-/dev/shm/.tmp_yi/workspace/omni-wan}"
DRY_RUN=0
PIDS=()
GPUS=(1 2 3 4 5 6)

usage() {
  echo "Usage: $0 [--dry-run]" >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    *) usage; exit 2 ;;
  esac
  shift
done

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

# Re-scan immediately before scheduling. Non-empty outputs are immutable inputs.
missing_file="$LOG_DIR/missing.prompts"
: > "$missing_file"
for prompt in "${prompts[@]}"; do
  [[ -s "$VIDEOS_DIR/$prompt-0.mp4" ]] || printf '%s\n' "$prompt" >> "$missing_file"
done
mapfile -t missing < "$missing_file"
echo "GPU 1-6 completion run: ${#missing[@]} missing prompts"

for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  shard_file="$LOG_DIR/gpu${gpu}.prompts"
  : > "$shard_file"
  for i in "${!missing[@]}"; do
    if [[ $((i % ${#GPUS[@]})) -eq "$shard" ]]; then
      printf '%s\n' "${missing[$i]}" >> "$shard_file"
    fi
  done
  echo "GPU $gpu: port $((8120 + gpu)), prompts $(wc -l < "$shard_file")"
done

if [[ "$DRY_RUN" -eq 1 || "${#missing[@]}" -eq 0 ]]; then
  [[ "$DRY_RUN" -eq 1 ]] && echo "Dry run: no processes were changed."
  [[ "${#missing[@]}" -eq 0 ]] && echo "Nothing to generate."
  exit 0
fi

# A stale vLLM server is safe to stop only when it has no active generate.py
# client. Never touch non-vLLM compute work on a requested GPU.
for gpu in "${GPUS[@]}"; do
  mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | awk -F, '{gsub(/ /, "", $1); if ($1 != "") print $1}')
  for pid in "${gpu_pids[@]}"; do
    [[ -n "$pid" ]] || continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" != *"vllm_omni.entrypoints.cli.main serve"* ]]; then
      echo "Refusing GPU $gpu: non-vLLM compute process $pid is active: $cmd" >&2
      exit 1
    fi
  done
done
if pgrep -af 'generate.py' >/dev/null; then
  echo "Refusing to clean stale servers while a generation client is active:" >&2
  pgrep -af 'generate.py' >&2
  exit 1
fi
for pid in $(pgrep -f 'vllm_omni.entrypoints.cli.main serve' || true); do
  [[ -r "/proc/$pid/environ" ]] || continue
  visible="$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
  for gpu in "${GPUS[@]}"; do
    if [[ "$visible" == "$gpu" ]]; then
      echo "Stopping stale vLLM server pid=$pid on GPU $gpu"
      kill -TERM "$pid"
      for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && { echo "Stale server $pid did not stop" >&2; exit 1; }
      break
    fi
  done
done

for gpu in "${GPUS[@]}"; do
  port=$((8120 + gpu))
  if ss -ltn "( sport = :$port )" | awk 'NR > 1 {found=1} END {exit !found}'; then
    echo "Port $port is already in use" >&2
    exit 1
  fi
done

config_sha256="$(sha256sum "$ENV_FILE" | awk '{print $1}')"
"$OMNI_ROOT/.venv/bin/python" - "$RUN_ROOT/manifest.json" "$ENV_FILE" "$config_sha256" "$missing_file" "$LOG_DIR" <<'PY'
import json, sys
from pathlib import Path
manifest, env_file, digest, missing_file, log_dir = map(Path, sys.argv[1:])
gpus = [1, 2, 3, 4, 5, 6]
missing = [x.strip() for x in Path(missing_file).read_text().splitlines() if x.strip()]
assignments = {str(gpu): [p for i, p in enumerate(missing) if i % len(gpus) == n] for n, gpu in enumerate(gpus)}
manifest.write_text(json.dumps({
    "config": "mxattention_full", "environment": {"VLLM_MXATTENTION": "1", "MXATTENTION_MODE": "mxattention_full", "MXATTENTION_QMAX": "7.25", "MXATTENTION_USE_HADAMARD": "1", "DIFFUSION_ATTENTION_BACKEND": "SAGE_ATTN"},
    "config_file": str(env_file.resolve()), "config_sha256": str(digest), "gpus": gpus,
    "ports": {str(gpu): 8120 + gpu for gpu in gpus}, "assignments": assignments,
    "log_dir": str(log_dir.resolve()),
}, indent=2) + "\n")
PY

for gpu in "${GPUS[@]}"; do
  shard_file="$LOG_DIR/gpu${gpu}.prompts"
  count=$(wc -l < "$shard_file")
  [[ "$count" -gt 0 ]] || continue
  PYTHONPATH="$SCRIPT_DIR/wan_flash_only_shim:$PLUGIN_ROOT/src:$OMNI_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$OMNI_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate.py" "$ENV_FILE" \
      --prompt-file "$shard_file" --num-prompts "$count" --output-dir "$VIDEOS_DIR" \
      --server-log "$LOG_DIR/gpu${gpu}.server.log" --cuda-devices "$gpu" \
      --port "$((8120 + gpu))" --tp 1 > "$LOG_DIR/gpu${gpu}.generate.log" 2>&1 &
  PIDS+=("$!")
done
wait
trap - EXIT INT TERM
echo "GPU 1-6 completion workers finished. Manifest: $RUN_ROOT/manifest.json"
