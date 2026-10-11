#!/bin/bash
# P10D7 (10/06/2026): (1) batched-scoring speed test: 3,000 pool pairs scored one at a time and in batches with the
# base Decide model, compared pair by pair; (2) stage 3 "screen" (Shaun: a Fastino pass emulating Sonnet's screen,
# tuned to remove): trained on the 9,164 stage 1-2 survivors Sonnet labeled (keep = v3 borrow >= 4 and v4 C/G, 22.8%),
# hand-checked pairs excluded; prints a removal curve on held-out citing patents and scores the hand-checked pairs.
# Job contract: logs/done-p10d7 holds ALLDONE only if every step succeeded.
# smoke: sudo /opt/pytorch/bin/pip install -q gliner2 && /opt/pytorch/bin/python remote-p10-decide.py --target screen --seed p10/label/stage3-pairs.jsonl --labels p10/label/stage3-labels.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/pool.jsonl --out p10/speed-smoke --score-only fastino/GLiNER2.5-Decide --batch-size 8 --speed-test 40 && /opt/pytorch/bin/python remote-p10-decide.py --target screen --seed p10/label/stage3-pairs.jsonl --labels p10/label/stage3-labels.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/rubric-hand-pairs.jsonl,p10/label/gate2-pairs.jsonl --out p10/stage3-smoke --smoke 10
# needs: remote-p10-decide.py p10/label/stage3-pairs.jsonl p10/label/stage3-labels.jsonl p10/label/gold-all.jsonl p10/label/pool.jsonl p10/label/rubric-hand-pairs.jsonl p10/label/gate2-pairs.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
L=p10/label
fail() { echo "FAILED: $1" > logs/done-p10d7; exit 1; }
$PY -c "import gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q gliner2 || fail install
rm -f logs/sums-p10d7.txt logs/discard-p10d7.txt
for BS in 32; do
  $PY remote-p10-decide.py --target screen --seed $L/stage3-pairs.jsonl --labels $L/stage3-labels.jsonl --gold $L/gold-all.jsonl \
    --pool $L/pool.jsonl --out p10/speed$BS --score-only fastino/GLiNER2.5-Decide --batch-size $BS --speed-test 3000 > logs/P10D7-speed$BS.log 2>&1 || fail speed$BS
  printf "p10/speed$BS/\n" >> logs/discard-p10d7.txt
done
$PY remote-p10-decide.py --target screen --max-fit 6000 --seed $L/stage3-pairs.jsonl --labels $L/stage3-labels.jsonl --gold $L/gold-all.jsonl \
  --pool $L/rubric-hand-pairs.jsonl,$L/gate2-pairs.jsonl --out p10/stage3 > logs/P10D7-stage3.log 2>&1 || fail stage3
grep -vE "it/s|Warning" logs/P10D7-stage3.log > logs/P10D7-stage3-clean.log
grep -h speed_test logs/P10D7-speed*.log > logs/P10D7-speed.txt
tar -czf p10/stage3/model-final.tgz -C p10/stage3/model final || fail tar
printf 'p10/stage3/train.jsonl\np10/stage3/val.jsonl\np10/stage3/model/\n' >> logs/discard-p10d7.txt
sha256sum p10/stage3/decide.json p10/stage3/pool-scores.jsonl p10/stage3/model-final.tgz logs/P10D7-stage3-clean.log logs/P10D7-speed.txt > logs/sums-p10d7.txt || fail sums
echo ALLDONE > logs/done-p10d7
