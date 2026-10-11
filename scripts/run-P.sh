#!/bin/bash
# Arm P: train, then the standard, contrast and literal-paraphrase tests, then end to end; arm R on the new test.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-encoder.py train --data pairs/arm-P.jsonl --out out/P1 --steps 3000 --batch 20 --natural-frac 0.4 --contrast-n 4 --lpara-n 5 --seed 1 > logs/P1-train.log 2>&1 || exit 1
$PY remote-encoder.py eval --adapter out/P1/lora.pt --out out/P1 > logs/P1-eval.log 2>&1
$PY remote-encoder.py eval --adapter out/P1/lora.pt --test pairs/test-contrast.jsonl --out out/P1-contrast > logs/P1-contrast.log 2>&1
$PY remote-encoder.py eval --adapter out/P1/lora.pt --test pairs/test-lpara.jsonl --out out/P1-lpara > logs/P1-lpara.log 2>&1
$PY remote-e2e.py --adapter out/P1/lora.pt --out e2e/P1 > logs/P1-e2e.log 2>&1
$PY remote-encoder.py eval --adapter ablation/R1/lora.pt --test pairs/test-lpara.jsonl --out out/R1-lpara > logs/R1-lpara.log 2>&1
echo ALLDONE > logs/done
