#!/bin/bash
# JEPA J4: J3 recipe on 3x data (consecutive windows of whole training meetings and threads, split by q3).
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-jepa.py --units jepa/units-w.jsonl --adapter ablation/S1/lora.pt --out jepa/J4 --steps 15000 --layers 2 --dropout 0.3 --wd 0.1 > logs/J4.log 2>&1
sha256sum jepa/J4/jepa.json jepa/J4/predictor.pt > logs/sums-j4.txt
echo ALLDONE > logs/done-j4
