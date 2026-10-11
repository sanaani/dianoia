#!/bin/bash
# JEPA J6: J5 (4 layers) with checkpoints selected by validation pool rank, 20,000 steps.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-jepa.py --units jepa/units-w.jsonl --adapter ablation/S1/lora.pt --out jepa/J6 --steps 20000 --layers 4 --dropout 0.3 --wd 0.1 --eval-every 1000 > logs/J6.log 2>&1
sha256sum jepa/J6/jepa.json jepa/J6/predictor.pt > logs/sums-j6.txt
echo ALLDONE > logs/done-j6
