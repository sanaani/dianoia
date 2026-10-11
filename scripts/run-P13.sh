#!/bin/bash
# P13: phase 13 step 1 (remote-p13.py): phase 12's three encoders scored on the far test questions asked as abstract,
# Background opening and title. Started only by pipelines/phase13 (scripts/p13-gpu.sh), after the laptop dry run.
# Resumable: remote-p13.py skips every encoder whose scores exist.
# Job contract: logs/done-p13 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p13.py --work p13 --config p13/config.used.yaml --out p13/smoke --smoke
# needs: remote-p13.py remote-p12.py p13/eval.jsonl p13/docs.jsonl p13/short.jsonl p13/config.used.yaml p13/adapters
cd ~/work
PY=/opt/pytorch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fail() { echo "FAILED: $1" > logs/done-p13; exit 1; }
$PY -c "import peft" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q peft || fail install
$PY remote-p13.py --work p13 --config p13/config.used.yaml --out p13/out > logs/P13.log 2>&1 || fail gpu
sha256sum p13/out/p13-short.json p13/out/scores/*.json logs/P13.log > logs/sums-p13.txt || fail sums
printf 'p13/smoke/\n' > logs/discard-p13.txt
echo ALLDONE > logs/done-p13
