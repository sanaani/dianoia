#!/bin/bash
# P12: the phase 12 GPU stage (remote-p12.py): contrastive LoRA training of Qwen3-Embedding-0.6B on far and near
# patent borrowings, then retrieval over phase 10's candidates.
# Started only by pipelines/phase12 (scripts/p12-gpu.sh), after the laptop dry run (rule smoke_gpu) passed.
# Resumable: remote-p12.py skips every stage whose output exists.
# Job contract: logs/done-p12 holds ALLDONE only if every step succeeded.
# smoke: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python remote-p12.py --work p12 --config p12/config.used.yaml --out p12/smoke --smoke
# needs: remote-p12.py p12/train-far.jsonl p12/train-near.jsonl p12/train-docs.jsonl p12/eval.jsonl p12/docs.jsonl p12/audit.json p12/config.used.yaml
cd ~/work
PY=/opt/pytorch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fail() { echo "FAILED: $1" > logs/done-p12; exit 1; }
$PY -c "import peft" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q peft || fail install
$PY remote-p12.py --work p12 --config p12/config.used.yaml --out p12/out > logs/P12.log 2>&1 || fail gpu
sha256sum p12/out/p12.json p12/out/scores/*.json logs/P12.log > logs/sums-p12.txt || fail sums
# Adapters (about 40 MB each) are kept: if far training helps, the next step builds on the trained encoder.
sha256sum p12/out/adapters/*/* >> logs/sums-p12.txt || fail sums
printf 'p12/smoke/\n' > logs/discard-p12.txt
echo ALLDONE > logs/done-p12
