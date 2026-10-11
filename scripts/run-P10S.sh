#!/bin/bash
# P10S (10/06/2026): run the exclusion cascade over the scale-up corpus (p10/corpus/corpus-<year>.jsonl, 207,959 pairs:
# every borrowing in the population the cascade was calibrated on, beyond the 75,053-pair pool). Per year shard:
#   stage 1 (unrelated)  scores every pair
#   stage 2 (same-tech)  scores stage 1's survivors
#   (stage 3 dropped 10/06/2026: P10D7's screen model scored AUC 0.52 on held-out pairs and removed Sonnet keeps at
#   the same rate as everything else at every cut, so Sonnet labels all stage 1-2 survivors)
# Checkpointed: a step whose decide.json exists is skipped, so an interrupted (spot) run resumes where it stopped.
# Survivors of all three go to Sonnet (p10-label.py subset --scores, then sonnet --prompt v4).
# Job contract: logs/done-p10s holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p10-decide.py --target unrelated --seed p10/label/stage3-pairs.jsonl --labels p10/label/stage3-labels.jsonl --gold p10/label/gold-all.jsonl --pool p10/corpus/corpus-2024.jsonl --out p10/scale-smoke/s1 --score-only fastino/GLiNER2.5-Decide --smoke 20
# needs: remote-p10-decide.py p10/label/stage3-pairs.jsonl p10/label/stage3-labels.jsonl p10/label/gold-all.jsonl
# (not listed, uploaded once by hand with "aws.sh upload", 3.6 GB: p10/stage1/model-final.tgz p10/stage2/model-final.tgz
# p10/corpus; "run" re-uploads everything on the needs line. The job fails at untar if stage 1's model is missing, and
# waits up to 2 hours for stage 2's.)
cd ~/work
PY=/opt/pytorch/bin/python
L=p10/label
fail() { echo "FAILED: $1" > logs/done-p10s; exit 1; }
$PY -c "import gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q gliner2 || fail install
untar() { [ -d p10/stage$1/model/final ] || { mkdir -p p10/stage$1/model && tar -xzf p10/stage$1/model-final.tgz -C p10/stage$1/model; } || fail untar$1; }
ARGS="--seed $L/stage3-pairs.jsonl --labels $L/stage3-labels.jsonl --gold $L/gold-all.jsonl"
step() {  # step <year> <n> <target> <model> [--pool-keep ...]
  Y=$1 N=$2 T=$3 M=$4; shift 4
  O=p10/scale/$Y/s$N
  [ -f $O/decide.json ] && return 0
  $PY remote-p10-decide.py --target $T $ARGS --pool p10/corpus/corpus-$Y.jsonl --out $O --score-only $M "$@" \
    >> logs/P10S-$Y.log 2>&1 || fail "$Y stage $N"
  rm -f $O/train.jsonl $O/val.jsonl
}
# Stage 1 over every year first, then stage 2: the stage 2 model can still be uploading while stage 1 runs (it arrives
# in parts and is moved into place only once the whole file verifies, so its presence means it is complete).
YEARS=$(ls p10/corpus/corpus-*.jsonl | sed 's/.*corpus-\([0-9]*\)[.]jsonl/\1/')
untar 1
for Y in $YEARS; do step $Y 1 unrelated p10/stage1/model/final; echo "s1 $Y $(date -u +%H:%M)" >> logs/P10S-progress.txt; done
for i in $(seq 120); do [ -f p10/stage2/model-final.tgz ] && break; sleep 60; done
untar 2
for Y in $YEARS; do
  step $Y 2 same-tech p10/stage2/model/final --pool-keep p10/scale/$Y/s1/pool-scores.jsonl
  echo "s2 $Y $(date -u +%H:%M)" >> logs/P10S-progress.txt
done
sha256sum p10/scale/*/s*/pool-scores.jsonl p10/scale/*/s*/decide.json > logs/sums-p10s.txt || fail sums
printf 'p10/stage1/model/\np10/stage2/model/\n' > logs/discard-p10s.txt
echo ALLDONE > logs/done-p10s
