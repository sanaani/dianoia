#!/bin/bash
# The phase 11 GPU job end to end, for pipelines/phase11 (10/07/2026), as scripts/p10-gpu.sh does for phase 10:
# budget gate, start, run (smoke lines first), wait, download with checksums, terminate. Never leaves an instance
# running: on any failure the job's log is shown and the instance is terminated (its self-terminate cap is the backstop).
#
#   scripts/p11-gpu.sh <work/p11 dir> <config.used.yaml>
set -u
P11=$1; CFG=$2
S=$(cd "$(dirname "$0")" && pwd)
WORK=$(dirname "$P11")
# A verified result already downloaded is placed, never paid for again (phase 10's retry launched a second
# instance to redo a finished job, 10/07/2026).
if [ -f "$P11/out/p11.json" ] && (cd "$WORK" && sha256sum -c --quiet logs/sums-p11.txt 2>/dev/null); then
  mkdir -p "$P11/metrics" && cp "$P11/out/p11.json" "$P11/metrics/p11.json" && echo "verified result already downloaded; placed, no instance started"
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

cp "$CFG" "$P11/config.used.yaml" 2>/dev/null || true
abort() {
  echo "ABORT: $1"
  "$S/aws.sh" ssh 'tail -40 logs/P11.log; tail -20 logs/run-P11.sh.smoke 2>/dev/null; cat logs/done-p11 2>/dev/null'
  read -r ID _ < "$WORK/aws-instance" && aws ec2 terminate-instances --instance-ids "$ID" --output text > /dev/null && rm -f "$WORK/aws-instance" && echo "terminated $ID"
  exit 1
}
"$S/aws.sh" start "$HOURS" "$MARKET" || { [ -f "$WORK/aws-instance" ] && abort "start"; exit 1; }
"$S/aws.sh" run run-P11.sh || abort "run (smoke or start)"
r=$("$S/aws.sh" wait done-p11 run-P11.sh | tail -1); echo "wait: $r"
[ "$r" = DONE ] || abort "job $r"
# A failed finish leaves the instance running (its self-terminate cap is the backstop): finish refuses to terminate
# when outputs are not downloaded, and terminating here anyway destroyed what that guard protected (10/07/2026).
"$S/aws.sh" finish logs/sums-p11.txt || { echo "DOWNLOAD NOT FINISHED: instance left running; see aws.sh finish output above"; exit 1; }
mkdir -p "$P11/metrics" && cp "$P11/out/p11.json" "$P11/metrics/p11.json" && echo "p11.json in place"
