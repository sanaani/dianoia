#!/bin/bash
# G1b: G1 without the text models (10/05/2026). Shaun: a fine-tuned text model is not a system we would run, so it is
# not the comparison that matters; idea model, surprise and the size baseline only. Reuses G1's split.
# Job contract: logs/done-g1 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-cmv.py --pred gh/g1/pred.jsonl --pairs gh/g1/pairs.jsonl --texts gh/g1/texts.jsonl --adapter ablation/S1/lora.pt --predictor jepa/J10/predictor.pt --out gh/G1-smoke --skip-text --smoke 20
# needs: remote-cmv.py remote-jepa.py remote-encoder.py ablation/S1/lora.pt jepa/J10/predictor.pt gh/g1/pairs.jsonl gh/g1/texts.jsonl
# (gh/g1/pred.jsonl is not listed: G1 wrote it on the instance, so it is not here to upload)
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-g1; exit 1; }
$PY remote-cmv.py --pred gh/g1/pred.jsonl --pairs gh/g1/pairs.jsonl --texts gh/g1/texts.jsonl --adapter ablation/S1/lora.pt \
  --predictor jepa/J10/predictor.pt --out gh/G1 --skip-text > logs/G1.log 2>&1 || fail model
sha256sum gh/g1/pred.jsonl gh/G1/cmv.json gh/G1/features.jsonl logs/G1-split.log logs/G1.log > logs/sums-g1.txt || fail sums
echo ALLDONE > logs/done-g1
