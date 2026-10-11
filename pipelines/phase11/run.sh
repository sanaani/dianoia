#!/bin/bash
# Start, stop and check the phase 11 pipeline (10/07/2026), as pipelines/phase10/run.sh does for phase 10.
#
#   pipelines/phase11/run.sh [target...]   preview, then run in the background (default target: everything)
#   pipelines/phase11/run.sh stop          stop snakemake and every process under it
#   pipelines/phase11/run.sh status        finished steps, running steps, failures, memory
#
# The dry runs (smoke_gpu, smoke_verdicts) are steps of the pipeline itself, on a small copy of the real files, and
# the GPU step depends on them, so no separate smoke command is needed. Refuses to start while another snakemake
# (phase 10's included) is alive: both use the one recorded AWS instance.
set -u
cd "$(dirname "$0")" || exit 1
PY=C:/Users/shaun/dianoia-data/venv/Scripts
LOG=.snakemake/run.log
alive() { tasklist 2>/dev/null | grep -iE '^snakemake\.exe' ; }

case "${1:-run}" in
stop)
  for pid in $(tasklist 2>/dev/null | grep -iE '^snakemake\.exe' | awk '{print $2}'); do taskkill //F //T //PID "$pid" > /dev/null 2>&1; done
  sleep 2; alive && echo "STILL RUNNING" || echo "stopped" ;;
status)
  grep -E "steps \(|Error in rule|MEMCAP|Exiting|Complete log|Nothing to be done" "$LOG" 2>/dev/null | tail -6
  grep -oE "rule [a-z_0-9]+:" "$LOG" 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
  alive > /dev/null && echo "running" || echo "not running" ;;
*)
  [ "${1:-}" = run ] && shift
  if alive; then echo "REFUSED: a snakemake run is alive (this phase's or another's)"; exit 1; fi
  FREE=$(powershell -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)')
  CAP=$(sed -n 's/^mem_mb_total: //p' config.yaml); MEM=$(( FREE - 1536 < CAP ? FREE - 1536 : CAP ))
  [ "$MEM" -lt 1500 ] && { echo "REFUSED: only $FREE MB free"; exit 1; }
  ARGS=(--cores 4 --resources mem_mb=$MEM gpu=1 --retries 1 --keep-going --rerun-incomplete "$@")
  "$PY/snakemake.exe" --unlock > /dev/null 2>&1
  "$PY/snakemake.exe" -n --quiet rules "${ARGS[@]}" || exit 1
  echo "memory for this run: $MEM MB"
  nohup "$PY/snakemake.exe" "${ARGS[@]}" > "$LOG" 2>&1 &
  sleep 15
  if grep -qE "LockException|WorkflowError|SyntaxError|Exiting because" "$LOG" || ! alive > /dev/null; then
    echo "NOT RUNNING:"; tail -5 "$LOG"; exit 1
  fi
  echo "running; log: pipelines/phase11/$LOG" ;;
esac
