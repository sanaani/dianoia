#!/bin/bash
# JEPA J5: J4 data, 4 layers, 40,000 steps (J4 validation loss was still falling).
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-jepa.py --units jepa/units-w.jsonl --adapter ablation/S1/lora.pt --out jepa/J5 --steps 40000 --layers 4 --dropout 0.3 --wd 0.1 --eval-every 1000 > logs/J5.log 2>&1
sha256sum jepa/J5/jepa.json jepa/J5/predictor.pt > logs/sums-j5.txt
echo ALLDONE > logs/done-j5
