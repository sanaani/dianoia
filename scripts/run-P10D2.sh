#!/bin/bash
# P10D2: GLiNER2.5-Decide trained on Sonnet's borrow scores (positives: real with borrow >= 4), judged on 190 hand
# labels by ranked precision and AUC, then scoring the 12,000 Sonnet-labeled pool pairs for comparison
# (remote-p10-decide.py). Round 1 (P10D) trained on the plain label and reached 50% precision.
# Job contract: logs/done-p10d2 holds ALLDONE only if every step succeeded.
# smoke: sudo /opt/pytorch/bin/pip install -q gliner2 && /opt/pytorch/bin/python remote-p10-decide.py --seed p10/label/seed.jsonl,p10/label/seed2.jsonl --labels p10/label/seed-sonnet.jsonl,p10/label/seed2-sonnet.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/pool-sub.jsonl --out p10/decide2-smoke --smoke 10
# needs: remote-p10-decide.py p10/label/seed.jsonl p10/label/seed2.jsonl p10/label/seed-sonnet.jsonl p10/label/seed2-sonnet.jsonl p10/label/gold-all.jsonl p10/label/pool-sub.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p10d2; exit 1; }
$PY -c "import gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q gliner2 || fail install
$PY remote-p10-decide.py --seed p10/label/seed.jsonl,p10/label/seed2.jsonl --labels p10/label/seed-sonnet.jsonl,p10/label/seed2-sonnet.jsonl \
  --gold p10/label/gold-all.jsonl --pool p10/label/pool-sub.jsonl --out p10/decide2 > logs/P10D2.log 2>&1 || fail decide
grep -vE "it/s|Warning" logs/P10D2.log > logs/P10D2-clean.log
sha256sum p10/decide2/decide.json p10/decide2/pool-scores.jsonl logs/P10D2-clean.log > logs/sums-p10d2.txt || fail sums
echo ALLDONE > logs/done-p10d2
