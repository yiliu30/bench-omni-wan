#!/usr/bin/env bash
# VBench dense93 campaign via bench-omni-wan's generate.py, one XPU.
# generate.py owns its server for the whole list (start -> all jobs -> kill);
# it is resume-safe, so a re-run only generates missing videos. This watchdog
# retries the whole generate.py invocation while videos are still missing
# (e.g. server or XPU died mid-run), probing the device between attempts.
# Usage: vbench_benchomni_watchdog.sh <xpu>
set -euo pipefail
XPU="${1:?usage: vbench_benchomni_watchdog.sh <xpu>}"
BENCH="/workspace/bench-omni-wan"
BASE="/workspace/tmp_yi_yiwan/vbench_dense93"
LOGS="${BASE}/logs"
PY="/opt/gfx-deps/venv/bin/python3"
mkdir -p "${LOGS}"

# The container env sets a corporate http_proxy; localhost traffic must never
# go through it (urllib would 403). urllib honors no_proxy.
export no_proxy="localhost,127.0.0.1${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Toolchain env (LD_LIBRARY_PATH for libmpicxx etc. that torch's import
# dlopens); the server subprocess inherits this via generate.py's os.environ.
source /opt/gfx-deps/env.sh >/dev/null 2>&1
source /home/ubuntu/main_cri_toolchain/env.sh >/dev/null 2>&1

case "${XPU}" in
  0) ENV_FILE="xpu_sagev3_mxfp8_cachedit_fb33343839_xpu0.env"; PROMPTS="${BASE}/prompts_xpu0_3way.txt" ;;
  1) ENV_FILE="xpu_sagev3_mxfp8_cachedit_fb33343839_xpu1.env"; PROMPTS="${BASE}/prompts_xpu1_3way.txt" ;;
  2) ENV_FILE="xpu_sagev3_mxfp8_cachedit_fb33343839_xpu2.env"; PROMPTS="${BASE}/prompts_xpu2_3way.txt" ;;
  *) echo "xpu must be 0, 1 or 2" >&2; exit 2 ;;
esac
STATUS="${LOGS}/xpu${XPU}_watchdog3.log"
OUTDIR="${BASE}"

say() { echo "$@" | tee -a "${STATUS}"; }

xpu_probe() {
  # Uses the env the script sourced at top level (the SAME env the server
  # subprocess inherits — proven working). Do NOT re-source here: re-sourcing
  # the two env files on top of an already-sourced shell reorders
  # LD_LIBRARY_PATH and breaks torch (sycl undefined symbol
  # _ZN4sycl3_V17handler... on D93 2026-09-12; both the original inline
  # version and the heredoc version failed this way).
  ZE_AFFINITY_MASK="${XPU}" timeout 600 /opt/gfx-deps/venv/bin/python -c '
import torch
assert torch.xpu.is_available() and torch.xpu.device_count() == 1, torch.xpu.device_count()
t = torch.ones(4096, device="xpu")
torch.xpu.synchronize()
print("probe: xpu OK, sum", int(t.sum()))
'
}

count_missing() {
  local missing=0 p
  while IFS= read -r p; do
    [[ -z "${p//[[:space:]]/}" ]] && continue
    [[ -f "${OUTDIR}/${p}-0.mp4" ]] || missing=$((missing + 1))
  done < "${PROMPTS}"
  echo "${missing}"
}

for attempt in 1 2 3 4 5; do
  say "=== xpu${XPU} attempt ${attempt} $(date -u +%FT%TZ); missing before: $(count_missing)"
  "${PY}" "${BENCH}/generate.py" "${BENCH}/${ENV_FILE}" \
    --prompt-file "${PROMPTS}" --timeout 2700 \
    --server-log "${LOGS}/xpu${XPU}_full_server_n2.log" >> "${STATUS}" 2>&1 || true
  missing="$(count_missing)"
  say "=== xpu${XPU} attempt ${attempt} finished; missing now: ${missing}"
  if (( missing == 0 )); then
    say "xpu${XPU} campaign complete"
    exit 0
  fi
  say "  probing XPU ${XPU} before retry..."
  xpu_probe >> "${STATUS}" 2>&1 || say "  probe FAILED"
  sleep 60
done
say "xpu${XPU} giving up after 5 attempts; missing: $(count_missing)"
exit 1
