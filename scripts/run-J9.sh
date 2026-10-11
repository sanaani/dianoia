#!/bin/bash
# J9 (10/04/2026), two configurations side by side on one GPU, after proving side-by-side runs do not interfere.
#  P1, P2: J8's predictor scored on J7's pool by two processes at once; each must reproduce the solo J8-on-J7 (365).
#  J9A: J8's data, source mix alpha 0.5 (chat and meeting got worse when wiki became 81% of training).
#  J9B: J8's data, 4 layers (J5 memorized at 1/10 the data; does capacity pay now?).
# Both then scored on J7's pool. Job contract: logs/done-j9 holds ALLDONE only if every step succeeded.
# needs: remote-jepa.py remote-encoder.py
cd ~/work
export SKIP_PROOF=${SKIP_PROOF:-1}  # proven 10/04/2026: [365, 365, 365]
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-j9; exit 1; }
A="--adapter ablation/S1/lora.pt --dropout 0.3 --wd 0.1 --vec-cache jepa/vec-cache.pt --require-audit"
if [ "${SKIP_PROOF:-0}" != 1 ]; then
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --layers 2 --out jepa/P1 --eval-only jepa/J8/predictor.pt > logs/P1.log 2>&1 &
p1=$!
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --layers 2 --out jepa/P2 --eval-only jepa/J8/predictor.pt > logs/P2.log 2>&1 &
p2=$!
wait $p1 || fail P1; wait $p2 || fail P2
$PY - <<'PYEOF' || fail parallel-proof
import json
r = [json.load(open(f"jepa/{d}/jepa.json"))["predictor"] for d in ("J8-on-J7", "P1", "P2")]
print("parallel proof", [x["median_rank"] for x in r], [x["top10"] for x in r])
assert r[0] == r[1] == r[2], "side-by-side scoring differs from solo"
PYEOF
fi
# One at a time: two J8-sized runs (each about 7 GB of host memory while loading) ran the 16 GB instance out of
# memory (10/04/2026). Side by side is proven equivalent (P1, P2) and fits datasets up to about J7's size.
$PY remote-jepa.py --units jepa/units-j8.jsonl $A --layers 2 --mix-alpha 0.5 --out jepa/J9A --steps 40000 --eval-every 2000 > logs/J9A.log 2>&1 || fail J9A
$PY remote-jepa.py --units jepa/units-j8.jsonl $A --layers 4 --out jepa/J9B --steps 40000 --eval-every 2000 > logs/J9B.log 2>&1 || fail J9B
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --layers 2 --out jepa/J9A-on-J7 --eval-only jepa/J9A/predictor.pt > logs/J9A-on-J7.log 2>&1 || fail J9A-on-J7
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --layers 4 --out jepa/J9B-on-J7 --eval-only jepa/J9B/predictor.pt > logs/J9B-on-J7.log 2>&1 || fail J9B-on-J7
sha256sum jepa/J9A/jepa.json jepa/J9A/predictor.pt jepa/J9B/jepa.json jepa/J9B/predictor.pt jepa/J9A-on-J7/jepa.json \
  jepa/J9B-on-J7/jepa.json jepa/P1/jepa.json jepa/P2/jepa.json logs/J9A.log logs/J9B.log logs/P1.log > logs/sums-j9.txt || fail sums
echo ALLDONE > logs/done-j9
