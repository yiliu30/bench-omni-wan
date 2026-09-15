#!/bin/bash
# monitor_black.sh — continuous guard for the dense93 campaign.
#
# Every 120s:
#   1. log progress (total done, per-list missing, video-stall warning)
#   2. luminance-scan every new/changed video at the base dir root
#      - BLACK: quarantine the video, then restart that XPU's server
#   3. step-stall check per active XPU: if the denoise step counter in the
#      server log does not advance for 25 min, the device is hung — restart
#      that XPU's server (much faster than waiting for the 45-min client
#      timeout). Observed on D93 2026-09-12 14:06: job hung ~45 min.
#   4. restart = kill all vllm_omni procs with ZE_AFFINITY_MASK=<xpu>, then
#      SIGTERM that XPU's generate.py if needed. The campaign watchdog
#      (scripts/vbench_watchdog_3xpu.sh) probes the XPU, retries generate.py
#      with a FRESH server, and resume-skip continues the campaign.
#
# State: scanned_state.txt (name|mtime), step_state_xpu<x> (step|epoch)
# Log:   logs/monitor_black.log
BASE=/workspace/tmp_yi_yiwan/vbench_dense93
MON=$BASE/monitor
LOGS=$BASE/logs
PY=/opt/gfx-deps/venv/bin/python3
QUAR=$BASE/black_quarantine
STATUS=$LOGS/monitor_black.log
STATE=$MON/scanned_state.txt
PIDFILE=$MON/monitor.pid
STALL_STEP_SECS=1500   # 25 min without step progress => hung device
mkdir -p "$QUAR" "$LOGS"; touch "$STATE"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "monitor already running (pid $(cat "$PIDFILE"))"; exit 0
fi
echo $$ > "$PIDFILE"

say() { echo "[$(date -u +%FT%TZ)] $*" >> "$STATUS"; }
say "monitor_black started (pid $$, v2 with step-stall detection)"

count_missing() {
  local missing=0 p
  while IFS= read -r p; do
    [[ -z "${p//[[:space:]]/}" ]] && continue
    [[ -f "$BASE/$p-0.mp4" ]] || missing=$((missing+1))
  done < "$1"
  echo "$missing"
}

xpu_job_pid() {  # generate.py pid for an xpu (empty if not running)
  ps aux | grep -F "xpu_sagev3_mxfp8_cachedit_fb33343839_xpu${1}.env" | grep -v grep | grep generate.py | awk '{print $2}' | head -1
}

restart_xpu_server() {  # $1 = xpu
  local xpu=$1 killed=0 pid gpid spid i
  # Note: the server renames itself to "vLLM-Omni::DiffusionWorker" (case
  # differs from cmdline "vllm_omni..."), so match case-insensitively.
  for pid in $(ps aux | grep -i "vllm.omni\|vllm_omni" | grep -v grep | awk '{print $2}'); do
    if tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -q "^ZE_AFFINITY_MASK=${xpu}$"; then
      kill "$pid" 2>/dev/null && killed=$((killed+1))
    fi
  done
  say "restart_xpu_server: killed ${killed} vllm_omni proc(s) for xpu${xpu}"
  for gpid in $(ps aux | grep -F "xpu_sagev3_mxfp8_cachedit_fb33343839_xpu${xpu}.env" | grep -v grep | grep generate.py | awk '{print $2}'); do
    for spid in $(cat /proc/$gpid/task/$gpid/children 2>/dev/null); do
      kill -- -"$spid" 2>/dev/null || kill "$spid" 2>/dev/null
    done
    kill -TERM "$gpid" 2>/dev/null && say "restart_xpu_server: SIGTERM generate.py pid ${gpid} (xpu${xpu})"
  done
  for i in $(seq 1 12); do
    [ -z "$(xpu_job_pid "$xpu")" ] && break
    sleep 10
  done
  say "restart_xpu_server: xpu${xpu} handed back to watchdog (probe + fresh server + resume)"
}

step_of() {  # $1 = xpu; last denoise step number from that xpu's server log
  tail -c 200000 "$LOGS/xpu${1}_full_server_n2.log" 2>/dev/null \
    | grep -aoE "[0-9]+/40 \[" | tail -1 | cut -d/ -f1
}

check_step_stall() {  # $1 = xpu
  local xpu=$1
  local sfile="$MON/step_state_xpu${xpu}"
  local now step prev pstep epoch pjob job
  now=$(date +%s)
  job=$(xpu_job_pid "$xpu")
  [ -z "$job" ] && return
  step=$(step_of "$xpu")
  prev=$(cat "$sfile" 2>/dev/null)
  pstep=${prev%%|*}; rest=${prev#*|}; epoch=${rest%%|*}; pjob=${rest##*|}
  # new generate.py process (fresh watchdog attempt) => fresh boot window
  if [ "$job" != "$pjob" ]; then
    echo "0|$now|$job" > "$sfile"
    return
  fi
  if [ -z "$step" ] || [ "$step" = "0" ]; then
    # booting, or only tqdm's "0/40" bar-init line: boot window
    if [ $((now - epoch)) -gt $STALL_STEP_SECS ]; then
      say "BOOT STALL: xpu${xpu} no step progress since server start ($(( (now - epoch) / 60 )) min) — restarting server"
      echo "0|0|$job" > "$sfile"
      restart_xpu_server "$xpu"
    fi
    return
  fi
  if [ "$step" != "$pstep" ]; then
    echo "${step}|${now}|$job" > "$sfile"; return
  fi
  if [ $((now - epoch)) -gt $STALL_STEP_SECS ]; then
    say "STEP STALL: xpu${xpu} stuck at step ${step}/40 for $(( (now - epoch) / 60 )) min — restarting server (device issue)"
    echo "0|0|$job" > "$sfile"
    restart_xpu_server "$xpu"
  fi
}

prev_total=0; stall=0
while true; do
  total=$(ls "$BASE"/*.mp4 2>/dev/null | wc -l)
  if [ "$total" -eq "$prev_total" ]; then stall=$((stall+1)); else stall=0; fi
  m2=$(count_missing "$BASE/prompts_xpu2_3way.txt")
  say "progress: ${total}/93 | xpu2 missing=${m2}"
  [ $stall -ge 23 ] && say "WARNING: no new video for ~$((stall*2)) min (check server logs)"
  prev_total=$total

  # step-stall checks for whichever xpu jobs are alive
  for x in 0 1 2; do
    [ -n "$(xpu_job_pid "$x")" ] && check_step_stall "$x"
  done

  for f in "$BASE"/*.mp4; do
    [ -e "$f" ] || continue
    name=$(basename "$f"); mtime=$(stat -c %Y "$f")
    prev=$(grep -F "$name|" "$STATE" 2>/dev/null | head -1 | cut -d'|' -f2-)
    [ "$prev" = "$mtime" ] && continue
    res=$("$PY" "$MON/scan_black.py" "$f" 2>/dev/null | head -1)
    case "$res" in
      *ERROR*)
        say "scan ERROR: $name ($res)"
        ;;
      *"| BLACK")
        say "BLACK VIDEO: $name ($res) — quarantining + restarting its XPU server"
        mv "$f" "$QUAR/${name%.mp4}_$(date -u +%Y%m%d_%H%M%S).mp4"
        prompt="${name%-0.mp4}"; xpu=""
        grep -qxF "$prompt" "$BASE/prompts_xpu2_3way.txt" && xpu=2
        [ -n "$xpu" ] && restart_xpu_server "$xpu" || say "note: $name not in any active list; quarantined only"
        ;;
      *"| ok")
        grep -vF "$name|" "$STATE" > "$STATE.tmp" 2>/dev/null
        echo "$name|$mtime" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
        say "scan ok: $name"
        ;;
      *)
        say "scan UNKNOWN result: $name ($res)"
        ;;
    esac
  done
  sleep 120
done
