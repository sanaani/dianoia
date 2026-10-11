#!/bin/bash
# Phase 14 step 1's cosine pass on one AWS GPU, for pipelines/phase14 rule far_cosine (10/08/2026): copy the laptop
# run's checkpoint into work/p14g, budget gate, start, run (smoke lines first), wait, download with checksums,
# terminate, then fold the GPU's rows into work/p14/far-tmp (p13-data.py far-merge). Never leaves an instance
# running: on any failure the job's log is shown and the instance is terminated (its self-terminate cap is the backstop).
#
#   scripts/p14-far-gpu.sh <work/p14 dir> <config.used.yaml> <python>
set -u
P14=$1; CFG=$2; PY=$3
S=$(cd "$(dirname "$0")" && pwd)
WORK=$(dirname "$P14")
G=$WORK/p14g
merge() { "$PY" "$S/p13-data.py" far-merge --data "$(dirname "$WORK")" --config "$CFG" --out "$P14" --work "$G"; }
# A verified result already downloaded is merged, never paid for again.
if [ -f "$G/far-tmp/far-part.jsonl" ] && (cd "$WORK" && sha256sum -c --quiet logs/sums-p14f.txt 2>/dev/null); then
  merge && echo "verified result already downloaded; merged, no instance started"; exit $?
fi
val() { sed -n "/^aws:/,/^[a-z]/s/^  $1: //p" "$CFG"; }
HOURS=$(val hours); MARKET=$(val market); LEFT=$(val budget_left_usd); MAX=$(val max_cost_usd)
# g5.xlarge us-east-1 prices: on-demand $1.006/h; spot is charged below it, so on-demand is the ceiling.
COST=$(awk -v h="$HOURS" 'BEGIN{printf "%.2f", h * 1.006}')
echo "budget gate: cap ${HOURS}h at most \$$COST; allowed \$$MAX; budget left \$$LEFT"
awk -v c="$COST" -v m="$MAX" -v l="$LEFT" 'BEGIN{exit !(c <= m && c <= l)}' || { echo "REFUSED: over budget"; exit 1; }
if [ -f "$WORK/aws-instance" ]; then
  read -r OLD _ < "$WORK/aws-instance"
  st=$(aws ec2 describe-instances --instance-ids "$OLD" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  case "$st" in terminated|shutting-down) echo "stale record of $OLD ($st) removed"; rm -f "$WORK/aws-instance" ;;
    *) echo "REFUSED: instance $OLD is recorded in $WORK/aws-instance and is ${st:-unknown}"; exit 1 ;; esac
fi
# The GPU starts from the laptop's checkpoint; its own far rows go to p14g so the download cannot overwrite p14's.
rm -rf "$G"; mkdir -p "$G/far-tmp"
cp "$P14/far-tmp/low.jsonl" "$P14/far-tmp/low.done.json" "$G/far-tmp/" || exit 1
[ -f "$P14/far-tmp/cosine.progress.json" ] && cp "$P14/far-tmp/cosine.progress.json" "$G/far-tmp/" || echo '{"done": 0}' > "$G/far-tmp/cosine.progress.json"
cp "$G/far-tmp/cosine.progress.json" "$G/far-tmp/cosine.start.json"
cp "$CFG" "$G/config.used.yaml"
abort() {
  echo "ABORT: $1"
  "$S/aws.sh" ssh 'tail -40 logs/P14F.log; tail -20 logs/run-P14F.sh.smoke 2>/dev/null; cat logs/done-p14f 2>/dev/null'
  read -r ID _ < "$WORK/aws-instance" && aws ec2 terminate-instances --instance-ids "$ID" --output text > /dev/null && rm -f "$WORK/aws-instance" && echo "terminated $ID"
  exit 1
}
"$S/aws.sh" start "$HOURS" "$MARKET" || { [ -f "$WORK/aws-instance" ] && abort "start"; exit 1; }
"$S/aws.sh" run run-P14F.sh || abort "run (smoke or start)"
r=$("$S/aws.sh" wait done-p14f run-P14F.sh | tail -1); echo "wait: $r"
[ "$r" = DONE ] || abort "job $r"
"$S/aws.sh" finish logs/sums-p14f.txt || { echo "DOWNLOAD NOT FINISHED: instance left running; see aws.sh finish output above"; exit 1; }
merge
