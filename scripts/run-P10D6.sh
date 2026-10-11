#!/bin/bash
# P10D6: score Claude's 190 hand-labeled pairs (gold-all.jsonl) with each P10D5 stage model, so the cascade can be
# judged pair by pair on the 140 raw filtered borrowings (gold.jsonl, 2013-2017, excluded from training), not only on
# pairs Sonnet had already called real. Seconds per stage.
# Job contract: logs/done-p10d6 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p10-decide.py --target unrelated --seed p10/label/seed.jsonl --labels p10/label/seed-sonnet.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/gold-all.jsonl --out p10/gold1-smoke --score-only p10/stage1/model/final --smoke 5
# needs: remote-p10-decide.py p10/label/gold-all.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
L=p10/label
fail() { echo "FAILED: $1" > logs/done-p10d6; exit 1; }
rm -f logs/sums-p10d6.txt
for N in 1 2; do
  T=$([ $N = 1 ] && echo unrelated || echo same-tech)
  $PY remote-p10-decide.py --target $T --seed $L/seed.jsonl --labels $L/seed-sonnet.jsonl --gold $L/gold-all.jsonl \
    --pool $L/gold-all.jsonl --out p10/gold-stage$N --score-only p10/stage$N/model/final > logs/P10D6-$N.log 2>&1 || fail stage$N
  sha256sum p10/gold-stage$N/pool-scores.jsonl >> logs/sums-p10d6.txt || fail sums$N
done
printf 'p10/gold-stage1/\np10/gold-stage2/\n' > logs/discard-p10d6.txt
echo ALLDONE > logs/done-p10d6
