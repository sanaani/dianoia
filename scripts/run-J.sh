#!/bin/bash
# First JEPA test: encode the splitter training units' ideas with S1, train the next-idea predictor, score it.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-jepa.py --adapter ablation/S1/lora.pt --out jepa/J1 > logs/J1.log 2>&1
sha256sum jepa/J1/* > logs/sums-j.txt
echo ALLDONE > logs/done-j
