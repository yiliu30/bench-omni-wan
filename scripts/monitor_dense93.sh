#!/bin/bash
# Persistent guard for the dense93 baseline output dir (NVIDIA B300 node).
# Every 5 min: completeness check (93/93 named files, none <1MB, count drift).
# Every 60 min: full black-frame re-sweep (scripts/scan_black.py).
# Alerts are appended as ALERT lines to <base>/logs/monitor.log.
# Usage: monitor_dense93.sh <base_dir> [python_with_av]
set -u
BASE="${1:-/models/yiliu7/wan-res/bf16_cachedit_dense93}"
PY="${2:-/models/yiliu7/envs/vbench/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPTS="$BASE/prompts.txt"
EXPECTED=$(grep -c . "$PROMPTS")
INTERVAL=300
SWEEP_EVERY=12
i=0
log(){ echo "$(date -u '+%F %T') $*" >> "$BASE/logs/monitor.log"; }
log "monitor start: expected=$EXPECTED interval=${INTERVAL}s black_sweep_every=$((SWEEP_EVERY * INTERVAL / 60))min"
while true; do
  i=$((i+1))
  N=$(ls "$BASE"/*.mp4 2>/dev/null | wc -l)
  BAD=$("$PY" - "$BASE" "$PROMPTS" <<'PYEOF'
import os, sys
B, P = sys.argv[1], sys.argv[2]
ps = [l.strip() for l in open(P) if l.strip()]
miss = [p for p in ps if not os.path.isfile(os.path.join(B, p + '-0.mp4'))]
small = [p for p in ps if os.path.isfile(os.path.join(B, p + '-0.mp4'))
         and os.path.getsize(os.path.join(B, p + '-0.mp4')) < 1_000_000]
print(len(miss) + len(small), '; '.join((miss + small)[:3]) or '-')
PYEOF
)
  NCOUNT=$(echo "$BAD" | awk '{print $1}')
  log "tick=$i videos=$N/$EXPECTED bad=$NCOUNT"
  if [ "$N" != "$EXPECTED" ]; then
    log "ALERT: video count $N != expected $EXPECTED (drift or stray files)"
  fi
  if [ "$NCOUNT" != "0" ]; then
    log "ALERT: missing/small files: $BAD"
  fi
  if [ $((i % SWEEP_EVERY)) -eq 0 ]; then
    "$PY" "$REPO/scripts/scan_black.py" "$BASE/*.mp4" >> "$BASE/logs/monitor_black.log" 2>&1
    rc=$?
    log "black sweep rc=$rc $(tail -1 "$BASE/logs/monitor_black.log" | cut -c1-100)"
    if [ "$rc" != "0" ]; then
      log "ALERT: BLACK video detected (see monitor_black.log)"
    fi
  fi
  sleep "$INTERVAL"
done
