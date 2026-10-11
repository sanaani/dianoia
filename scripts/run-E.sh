#!/bin/bash
# Equivalence checks for the fast remote-jepa.py (vector cache, GPU-resident batches, vectorized scoring), after V.
#  E1: J7's predictor scored by the new code on J7's data: must reproduce J7's numbers (old code).
#  E2: J4b's predictor scored on J7's test pool: the fair J4b vs J7 comparison.
#  E3: J4b retrained by the new code (same seed and arguments): must match J4b within run-to-run noise.
# Job contract: logs/done-e holds ALLDONE only if every step succeeded.
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-e; exit 1; }
until [ -f logs/done-v ]; do sleep 30; done
A="--adapter ablation/S1/lora.pt --layers 2 --dropout 0.3 --wd 0.1 --eval-every 1000 --vec-cache jepa/vec-cache.pt --require-audit"
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --out jepa/E1 --eval-only jepa/J7/predictor.pt > logs/E1.log 2>&1 || fail E1
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --out jepa/E2 --eval-only jepa/J4B/predictor.pt > logs/E2.log 2>&1 || fail E2
$PY remote-jepa.py --units jepa/units-j4b.jsonl $A --out jepa/E3 --steps 15000 > logs/E3.log 2>&1 || fail E3
sha256sum jepa/E1/jepa.json jepa/E2/jepa.json jepa/E3/jepa.json jepa/E3/predictor.pt jepa/J7/jepa.json jepa/J7/predictor.pt jepa/J4B/jepa.json logs/E1.log logs/E2.log logs/E3.log logs/J7.log > logs/sums-e.txt || fail sums
echo ALLDONE > logs/done-e
