#!/bin/bash
# The phase 10 GPU job end to end, for pipelines/phase10 (10/05/2026): budget gate, start, run (smoke lines first),
# wait, download with checksums, terminate. Never leaves an instance running: on any failure the job's log is copied
# here and the instance is terminated (its self-terminate cap is the backstop).
#
#   scripts/p10-gpu.sh <work/p10 dir> <config.used.yaml>
set -u
P10=$1; CFG=$2
S=$(cd "$(dirname "$0")" && pwd)
WORK=$(dirname "$P10")
# A verified result already downloaded is placed, never paid for again: a pipeline retry after a failed finish
# launched a second instance to redo a 2.5-hour job whose result was already here (10/07/2026).
if [ -f "$P10/out/gpu.json" ] && (cd "$WORK" && sha256sum -c --quiet logs/sums-p10.txt 2>/dev/null); then
  mkdir -p "$P10/metrics" && cp "$P10/out/gpu.json" "$P10/metrics/gpu.json" && echo "verified result already downloaded; placed, no instance started"
  exit 0
fi
val() { sed -n "/^aws:/,/^[a-z]/s/^  $1: //p" "$CFG"; }
HOURS=$(val hours); MARKET=$(val market); LEFT=$(val budget_left_usd); MAX=$(val max_cost_usd)
# g5.xlarge us-east-1 prices: on-demand $1.006/h; spot is charged below it, so on-demand is the ceiling.
COST=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h * 1.006}')
echo "budget gate: cap ${HOURS}h at most \$$COST; allowed \$$MAX; budget left \$$LEFT"
awk -v c="$COST" -v m="$MAX" -v l="$LEFT" 'BEGIN{exit !(c <= m && c <= l)}' || { echo "REFUSED: over budget"; exit 1; }
# A record whose instance EC2 reports as gone is stale (an instance terminated by hand left one, 10/07/2026).
if [ -f "$WORK/aws-instance" ]; then
  read -r OLD _ < "$WORK/aws-instance"
  st=$(aws ec2 describe-instances --instance-ids "$OLD" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  case "$st" in terminated|shutting-down) echo "stale record of $OLD ($st) removed"; rm -f "$WORK/aws-instance" ;;
    *) echo "REFUSED: instance $OLD is recorded in $WORK/aws-instance and is ${st:-unknown}"; exit 1 ;; esac
fi

# Inputs live under work/p10, which aws.sh uploads at the same relative path.
cp "$CFG" "$P10/config.used.yaml" 2>/dev/null || true
abort() {
  echo "ABORT: $1"
  "$S/aws.sh" ssh 'tail -40 logs/P10.log; tail -20 logs/run-P10.sh.smoke 2>/dev/null; cat logs/done-p10 2>/dev/null'
  read -r ID _ < "$WORK/aws-instance" && aws ec2 terminate-instances --instance-ids "$ID" --output text > /dev/null && rm -f "$WORK/aws-instance" && echo "terminated $ID"
  exit 1
}
"$S/aws.sh" start "$HOURS" "$MARKET" || { [ -f "$WORK/aws-instance" ] && abort "start"; exit 1; }
"$S/aws.sh" run run-P10.sh || abort "run (smoke or start)"
r=$("$S/aws.sh" wait done-p10 run-P10.sh | tail -1); echo "wait: $r"
[ "$r" = DONE ] || abort "job $r"
# A failed finish leaves the instance running (its self-terminate cap is the backstop): finish refuses to terminate
# when outputs are not downloaded, and terminating here anyway destroyed what that guard protected (10/07/2026).
"$S/aws.sh" finish logs/sums-p10.txt || { echo "DOWNLOAD NOT FINISHED: instance left running; see aws.sh finish output above"; exit 1; }
mkdir -p "$P10/metrics" && cp "$P10/out/gpu.json" "$P10/metrics/gpu.json" && echo "gpu.json in place"
