#!/bin/bash
# FOMC windows from ConvoKit's fomc-corpus (CPU; the GPU job splits them after the laptop's pre-clean).
# Job contract: logs/done-f holds ALLDONE only if every step succeeded.
# needs: remote-fomc-windows.py
cd ~/work
mkdir -p jepa
/opt/pytorch/bin/python remote-fomc-windows.py --out jepa/fomc-windows.jsonl > logs/F.log 2>&1 || { echo FAILED > logs/done-f; exit 1; }
sha256sum jepa/fomc-windows.jsonl logs/F.log > logs/sums-f.txt && echo ALLDONE > logs/done-f || echo FAILED > logs/done-f
