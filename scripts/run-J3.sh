#!/bin/bash
# JEPA J3: 2 layers, dropout 0.3, wd 0.1, paraphrased training units added, early stopping on validation.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-jepa.py --adapter ablation/S1/lora.pt --out jepa/J3 --steps 8000 --layers 2 --dropout 0.3 --wd 0.1 > logs/J3.log 2>&1
sha256sum jepa/J3/* > logs/sums-j3.txt
echo ALLDONE > logs/done-j3
