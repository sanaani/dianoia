#!/bin/bash
# J8: J7's data + WikiConv 2018 (vLLM-split, about 10x the transitions), J4's recipe, more steps for more data;
# checkpoint by validation pool rank, --require-audit. Then J8's predictor scored on J7's exact test pool (J8-on-J7),
# so the comparison with J7 (416 overall, wiki 590) differs only in training data.
# Job contract: logs/done-j8 holds ALLDONE only if every step succeeded.
# needs: remote-jepa.py remote-encoder.py ablation/S1/lora.pt
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-j8; exit 1; }
A="--adapter ablation/S1/lora.pt --layers 2 --dropout 0.3 --wd 0.1 --vec-cache jepa/vec-cache.pt --require-audit"
$PY remote-jepa.py --units jepa/units-j8.jsonl $A --out jepa/J8 --steps 40000 --eval-every 2000 > logs/J8.log 2>&1 || fail J8
$PY remote-jepa.py --units jepa/units-j7.jsonl $A --out jepa/J8-on-J7 --eval-only jepa/J8/predictor.pt > logs/J8-on-J7.log 2>&1 || fail J8-on-J7
sha256sum jepa/J8/jepa.json jepa/J8/predictor.pt jepa/J8-on-J7/jepa.json logs/J8.log logs/J8-on-J7.log > logs/sums-j8.txt || fail sums
echo ALLDONE > logs/done-j8
