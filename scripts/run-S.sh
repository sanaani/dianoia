#!/bin/bash
# Arm S: continue arm P on arm P's data plus the splitter's own paraphrase pairs; then every test.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-encoder.py train --data pairs/arm-S.jsonl --init ablation/P1/lora.pt --out out/S1 --steps 1500 --lr 1e-4 --batch 20 --natural-frac 0.3 --contrast-n 3 --lpara-n 4 --spara-n 5 --seed 2 > logs/S1-train.log 2>&1 || exit 1
$PY remote-e2e.py --adapter out/S1/lora.pt --out e2e/S1 > logs/S1-e2e.log 2>&1
$PY remote-encoder.py eval --adapter out/S1/lora.pt --out out/S1 > logs/S1-eval.log 2>&1
$PY remote-encoder.py eval --adapter out/S1/lora.pt --test pairs/test-spara.jsonl --out out/S1-spara > logs/S1-spara.log 2>&1
$PY remote-encoder.py eval --adapter ablation/P1/lora.pt --test pairs/test-spara.jsonl --out out/P1-spara > logs/P1-spara.log 2>&1
$PY remote-encoder.py eval --adapter out/S1/lora.pt --test pairs/test-lpara.jsonl --out out/S1-lpara > logs/S1-lpara.log 2>&1
$PY remote-encoder.py eval --adapter out/S1/lora.pt --test pairs/test-contrast.jsonl --out out/S1-contrast > logs/S1-contrast.log 2>&1
find out e2e -name '*.json*' -newer pairs/arm-S.jsonl | sort | xargs sha256sum > logs/sums.txt
echo ALLDONE > logs/done
