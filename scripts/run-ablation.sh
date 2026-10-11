#!/bin/bash
# GPU driver for the lossy-positive ablation (PREREGISTRATION 10/02/2026): baseline, then L and B at seed 1, then seed 2.
set -x
cd ~/w
P=/opt/pytorch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f out/base/eval.jsonl ] || $P remote-encoder.py eval --out out/base > logs/base.log 2>&1
for seed in ${SEEDS:-1 2}; do
  for arm in L B; do
    run=$arm$seed
    [ -f out/$run/lora.pt ] || $P remote-encoder.py train --data pairs/arm-$arm.jsonl --out out/$run --seed $seed ${STEPS:+--steps $STEPS} > logs/$run-train.log 2>&1
    [ -f out/$run/eval.jsonl ] || $P remote-encoder.py eval --adapter out/$run/lora.pt --out out/$run > logs/$run-eval.log 2>&1
  done
done
touch ALL_DONE
