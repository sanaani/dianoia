#!/bin/bash
# Arm M: continue S1 for a short run on arm S's data plus merged-idea pairs. The targeted merge test runs first
# (M1 and S1), then end to end, then the lpara and spara tests as regression checks.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-encoder.py train --data pairs/arm-M.jsonl --init ablation/S1/lora.pt --out out/M1 --steps 400 --lr 1e-4 --batch 24 --natural-frac 0.3 --contrast-n 2 --lpara-n 3 --spara-n 3 --merge-n 8 --seed 3 > logs/M1-train.log 2>&1 || exit 1
$PY remote-encoder.py eval --adapter out/M1/lora.pt --test pairs/test-merge.jsonl --out out/M1-merge > logs/M1-merge.log 2>&1
$PY remote-encoder.py eval --adapter ablation/S1/lora.pt --test pairs/test-merge.jsonl --out out/S1-merge > logs/S1-merge.log 2>&1
echo MERGEDONE > logs/done-merge
$PY remote-e2e.py --adapter out/M1/lora.pt --out e2e/M1 > logs/M1-e2e.log 2>&1
$PY remote-encoder.py eval --adapter out/M1/lora.pt --test pairs/test-lpara.jsonl --out out/M1-lpara > logs/M1-lpara.log 2>&1
$PY remote-encoder.py eval --adapter out/M1/lora.pt --test pairs/test-spara.jsonl --out out/M1-spara > logs/M1-spara.log 2>&1
find out e2e -name '*.json*' -newer pairs/arm-M.jsonl | sort | xargs sha256sum > logs/sums-m.txt
echo ALLDONE > logs/done-m
