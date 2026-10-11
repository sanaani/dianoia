#!/bin/bash
# P14: phase 14 step 3 (remote-p14.py): phase 12's three encoders on 300 strict OpenAlex pairs, each against 1,000
# candidates. Started only by pipelines/phase14 (scripts/p14-gpu.sh), after the laptop dry run.
# Resumable: remote-p14.py skips every encoder whose scores exist.
# Job contract: logs/done-p14 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p14.py --work p14/step3 --config p14/step3/config.used.yaml --out p14/step3/smoke --smoke
# needs: remote-p14.py remote-p12.py p14/step3/eval.jsonl p14/step3/docs.jsonl p14/step3/config.used.yaml p14/step3/adapters
cd ~/work
PY=/opt/pytorch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fail() { echo "FAILED: $1" > logs/done-p14; exit 1; }
$PY -c "import peft" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q peft || fail install
$PY remote-p14.py --work p14/step3 --config p14/step3/config.used.yaml --out p14/step3/out > logs/P14.log 2>&1 || fail gpu
sha256sum p14/step3/out/p14-test.json p14/step3/out/scores/*.json logs/P14.log > logs/sums-p14.txt || fail sums
printf 'p14/step3/smoke/\n' > logs/discard-p14.txt
echo ALLDONE > logs/done-p14
