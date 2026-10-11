#!/bin/bash
# J4b alone: J4 data on conversation-level splits (the new baseline). J7 waits for the wiki duplicate fix.
# Both: J4's recipe (2 layers, dropout 0.3, wd 0.1), 15,000 steps, checkpoint chosen by validation pool rank,
# and --require-audit (no training without a passing audit of the exact file).
# Job contract (aws.sh wait): logs/done-j4b holds ALLDONE only if every step succeeded, FAILED otherwise.
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-j4b; exit 1; }
for R in j4b; do
  $PY remote-jepa.py --units jepa/units-$R.jsonl --require-audit --adapter ablation/S1/lora.pt --out jepa/${R^^} \
    --steps 15000 --layers 2 --dropout 0.3 --wd 0.1 --eval-every 1000 > logs/${R^^}.log 2>&1 || fail "$R"
done
sha256sum jepa/J4B/jepa.json jepa/J4B/predictor.pt logs/J4B.log > logs/sums-j4b.txt || fail sums
echo ALLDONE > logs/done-j4b
