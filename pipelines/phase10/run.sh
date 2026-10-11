#!/bin/bash
# Start, stop and check the phase 10 pipeline (10/05/2026). The only way it is run, so its flags are never retyped.
#
#   pipelines/phase10/run.sh [target...]   preview, then run in the background (default target: everything)
#   pipelines/phase10/run.sh stop          stop snakemake and every process under it
#   pipelines/phase10/run.sh status        finished steps, running steps, failures, memory
#
# Guards: refuses to start while a run (or a stray download from one) is alive, since two curls writing one file
# corrupt it; stopping a run with TaskStop left both curls running (10/05/2026). Memory for the run is the smaller of
# config.yaml's mem_mb_total and what Windows has free minus 1.5 GB.
set -u
cd "$(dirname "$0")" || exit 1
PY=C:/Users/shaun/dianoia-data/venv/Scripts
LOG=.snakemake/run.log
alive() { tasklist 2>/dev/null | grep -iE '^(snakemake|curl)\.exe' ; }

case "${1:-run}" in
smoke)
  # The laptop steps on make_fixture.py's synthetic PatentsView, in the fixture's own working directory, so it can
  # run beside a real run. Foreground; exit 1 if any step fails.
  FX=C:/Users/shaun/dianoia-data/fixture-p10; W=$FX/work/p10
  [ -d "$FX/raw" ] || "$PY/python.exe" make_fixture.py "$FX" || exit 1
  rm -rf "$W"; mkdir -p "$W/review" && touch "$W/review/APPROVED"  # synthetic data needs no human review
  shift; T=("$@"); [ ${#T[@]} -eq 0 ] && T=("$W/smoke/report/REPORT.md")
  "$PY/snakemake.exe" -s "$PWD/Snakefile" -d "$W" --configfile "$PWD/config.yaml" "$PWD/smoke.yaml" --config data=$FX \
    --cores 6 --resources mem_mb=4000 gpu=1 --retries 0 "${T[@]}" > "$W/smoke.log" 2>&1
  rc=$?; grep -E "steps \(|Error in rule" "$W/smoke.log" | tail -3
  [ $rc -eq 0 ] && echo "SMOKE PASSED" || { echo "SMOKE FAILED ($W/smoke.log)"; for f in "$W"/logs/*.log; do grep -l -E "Traceback|MEMCAP|Error" "$f"; done; }
  exit $rc ;;
stop)
  for pid in $(tasklist 2>/dev/null | grep -iE '^(snakemake|curl)\.exe' | awk '{print $2}'); do taskkill //F //T //PID "$pid" > /dev/null 2>&1; done
  sleep 2; alive && echo "STILL RUNNING" || echo "stopped" ;;
status)
  grep -E "steps \(|Error in rule|MEMCAP|Exiting|Complete log|Nothing to be done" "$LOG" 2>/dev/null | tail -6
  grep -oE "rule [a-z_0-9]+:" "$LOG" 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
  grep -l MEMCAP "$(sed -n 's/^data: //p' config.yaml)"/work/p10/logs/*.log 2>/dev/null | sed 's/^/memory-capped: /'
  alive > /dev/null && echo "running" || echo "not running"
  powershell -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; "{0:N1} GB free of {1:N1} GB" -f ($o.FreePhysicalMemory/1MB),($o.TotalVisibleMemorySize/1MB)' ;;
*)
  [ "${1:-}" = run ] && shift
  if alive; then echo "REFUSED: a run or download is alive (run.sh stop first)"; exit 1; fi
  # Every real run is preceded by the fixture smoke: a syntax error reached the real run because only the real run
  # had been restarted after an edit (10/05/2026). SKIP_SMOKE=1 only to resume a run with no code changed since.
  if [ "${SKIP_SMOKE:-0}" != 1 ]; then ./run.sh smoke || { echo "REFUSED: smoke failed"; exit 1; }; fi
  FREE=$(powershell -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)')
  CAP=$(sed -n 's/^mem_mb_total: //p' config.yaml); MEM=$(( FREE - 1536 < CAP ? FREE - 1536 : CAP ))
  [ "$MEM" -lt 1500 ] && { echo "REFUSED: only $FREE MB free"; exit 1; }
  ARGS=(--cores 8 --resources mem_mb=$MEM net=4 gpu=1 --retries 2 --keep-going --rerun-incomplete "$@")
  # Nothing is alive (checked above), so any lock is stale: "run.sh stop" kills snakemake before it can remove its
  # lock, and the next start died on it (10/05/2026).
  "$PY/snakemake.exe" --unlock > /dev/null 2>&1
  "$PY/snakemake.exe" -n --quiet rules "${ARGS[@]}" || exit 1
  echo "memory for this run: $MEM MB"
  nohup "$PY/snakemake.exe" "${ARGS[@]}" > "$LOG" 2>&1 &
  # Confirm it is running: the failed start above printed "started" and exited at once.
  sleep 15
  if grep -qE "LockException|WorkflowError|SyntaxError|Exiting because" "$LOG" || ! alive > /dev/null; then
    echo "NOT RUNNING:"; tail -5 "$LOG"; exit 1
  fi
  echo "running; log: pipelines/phase10/$LOG" ;;
esac
