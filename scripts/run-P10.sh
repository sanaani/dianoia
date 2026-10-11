#!/bin/bash
# P10: the phase 10 GPU stage (remote-p10.py): Qwen3 role replacement, Qwen3-Embedding recovery and leakage, H3.
# Started only by pipelines/phase10 (scripts/p10-gpu.sh), after the laptop dry run (rule smoke_gpu) passed.
# Job contract: logs/done-p10 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p10.py --work p10/gpu --config p10/config.used.yaml --out p10/smoke --smoke 10
# needs: remote-p10.py p10/gpu/cands.jsonl p10/gpu/queries.jsonl p10/gpu/h3_words.jsonl p10/config.used.yaml
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p10; exit 1; }
$PY remote-p10.py --work p10/gpu --config p10/config.used.yaml --out p10/out > logs/P10.log 2>&1 || fail gpu
sha256sum p10/out/gpu.json logs/P10.log > logs/sums-p10.txt || fail sums
# Intermediates (LM scores, embedding caches, the smoke run, the trimmed lexicon) are not downloaded. Unnamed, they
# made aws.sh finish refuse to terminate on 10/07/2026.
printf 'p10/out/scores.pt\np10/out/emb/\np10/out/cands-trimmed.jsonl\np10/smoke/\n' > logs/discard-p10.txt
echo ALLDONE > logs/done-p10
