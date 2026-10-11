#!/bin/bash
# JEPA J2: residual predictor with a cosine term, 15,000 steps; reuses J1 instance.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-jepa.py --adapter ablation/S1/lora.pt --out jepa/J2 --steps 15000 > logs/J2.log 2>&1
sha256sum jepa/J1/* jepa/J2/* > logs/sums-j.txt
echo ALLDONE > logs/done-j2
