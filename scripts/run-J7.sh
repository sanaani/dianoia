#!/bin/bash
# J4b: J4's data on conversation-level splits (the new baseline). J7: + Wikipedia talk-page conversations.
# Both: J4's recipe (2 layers, dropout 0.3, wd 0.1), 15,000 steps, checkpoint chosen by validation pool rank,
# and --require-audit (no training without a passing audit of the exact file).
# Job contract (aws.sh wait): logs/done-j7 holds ALLDONE only if every step succeeded, FAILED otherwise.
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-j7; exit 1; }
for R in j4b j7; do
  $PY remote-jepa.py --units jepa/units-$R.jsonl --require-audit --adapter ablation/S1/lora.pt --out jepa/${R^^} \
    --steps 15000 --layers 2 --dropout 0.3 --wd 0.1 --eval-every 1000 > logs/${R^^}.log 2>&1 || fail "$R"
done
sha256sum jepa/J4B/jepa.json jepa/J4B/predictor.pt jepa/J7/jepa.json jepa/J7/predictor.pt logs/J4B.log logs/J7.log > logs/sums-j7.txt || fail sums
echo ALLDONE > logs/done-j7
