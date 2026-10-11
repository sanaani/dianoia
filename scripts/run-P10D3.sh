#!/bin/bash
# P10D3: score the second Sonnet-labeled pool subset (18,000 pairs) with the P10D2 model, so every pair of the fresh
# hand check of 10/06/2026 has a Fastino score (P10D2 scored only the first subset, which held 22 of the 50).
# Job contract: logs/done-p10d3 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p10-decide.py --seed p10/label/seed.jsonl --labels p10/label/seed-sonnet.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/pool-sub2.jsonl --out p10/decide3-smoke --score-only p10/decide2/model/final --smoke 10
# needs: remote-p10-decide.py p10/label/pool-sub2.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p10d3; exit 1; }
$PY remote-p10-decide.py --seed p10/label/seed.jsonl --labels p10/label/seed-sonnet.jsonl --gold p10/label/gold-all.jsonl \
  --pool p10/label/pool-sub2.jsonl --out p10/decide3 --score-only p10/decide2/model/final > logs/P10D3.log 2>&1 || fail score
sha256sum p10/decide3/decide.json p10/decide3/pool-scores.jsonl > logs/sums-p10d3.txt || fail sums
echo ALLDONE > logs/done-p10d3
