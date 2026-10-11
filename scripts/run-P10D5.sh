#!/bin/bash
# P10D5: the first two exclusion stages (Shaun, 10/06/2026: prove exclusion through a series of steps, not inclusion).
#   stage 1 "unrelated":  Sonnet unrelated vs everything with a link; scores the whole val/test pool (75,053 pairs)
#   stage 2 "same-tech":  Sonnet same technology vs Sonnet real with borrow >= 4; scores only stage 1's survivors
# Both also score kept-pairs.jsonl (Sonnet-kept pairs + the 50 hand-checked pool pairs), so each stage is measured on
# every hand-checked pair on its own. A pair survives if every stage keeps it.
# Job contract: logs/done-p10d5 holds ALLDONE only if every step succeeded.
# smoke: sudo /opt/pytorch/bin/pip install -q gliner2 && /opt/pytorch/bin/python remote-p10-decide.py --target unrelated --seed p10/label/seed.jsonl,p10/label/seed2.jsonl --labels p10/label/seed-sonnet.jsonl,p10/label/seed2-sonnet.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/kept-pairs.jsonl --out p10/stage1-smoke --smoke 10 && /opt/pytorch/bin/python remote-p10-decide.py --target same-tech --seed p10/label/seed.jsonl,p10/label/seed2.jsonl --labels p10/label/seed-sonnet.jsonl,p10/label/seed2-sonnet.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/pool.jsonl,p10/label/kept-pairs.jsonl --pool-keep p10/stage1-smoke/pool-scores.jsonl --out p10/stage2-smoke --smoke 10
# needs: remote-p10-decide.py p10/label/seed.jsonl p10/label/seed2.jsonl p10/label/seed-sonnet.jsonl p10/label/seed2-sonnet.jsonl p10/label/gold-all.jsonl p10/label/pool.jsonl p10/label/kept-pairs.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
L=p10/label
fail() { echo "FAILED: $1" > logs/done-p10d5; exit 1; }
$PY -c "import gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q gliner2 || fail install
stage() {  # stage <n> <target> <pool files> [--pool-keep ...]
  N=$1 T=$2 P=$3; shift 3
  $PY remote-p10-decide.py --target $T --max-fit 6000 --seed $L/seed.jsonl,$L/seed2.jsonl --labels $L/seed-sonnet.jsonl,$L/seed2-sonnet.jsonl \
    --gold $L/gold-all.jsonl --pool $P --out p10/stage$N "$@" > logs/P10D5-stage$N.log 2>&1 || fail stage$N
  grep -vE "it/s|Warning" logs/P10D5-stage$N.log > logs/P10D5-stage$N-clean.log
  tar -czf p10/stage$N/model-final.tgz -C p10/stage$N/model final || fail tar$N
  printf "p10/stage$N/train.jsonl\np10/stage$N/val.jsonl\np10/stage$N/model/\n" >> logs/discard-p10d5.txt
  sha256sum p10/stage$N/decide.json p10/stage$N/pool-scores.jsonl p10/stage$N/model-final.tgz logs/P10D5-stage$N-clean.log >> logs/sums-p10d5.txt || fail sums$N
}
rm -f logs/discard-p10d5.txt logs/sums-p10d5.txt
stage 1 unrelated $L/pool.jsonl,$L/kept-pairs.jsonl
stage 2 same-tech $L/pool.jsonl,$L/kept-pairs.jsonl --pool-keep p10/stage1/pool-scores.jsonl
echo ALLDONE > logs/done-p10d5
