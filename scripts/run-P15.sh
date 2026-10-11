#!/bin/bash
# P15: phase 15 (remote-p15.py): step 1's data curve (phase 12's far training on 419 and 838 pairs), then step 2's
# Fastino screen on phase 14's Sonnet-labeled far pairs. Started only by pipelines/phase15 (scripts/p15-gpu.sh),
# after the laptop dry run of the curve. The Fastino smoke runs here: it is too heavy for the laptop.
# Resumable: remote-p15.py skips every stage whose output exists.
# Job contract: logs/done-p15 holds ALLDONE only if every step succeeded.
# Smoke lines run before this script, so the first installs what the job needs (gliner2 is not on the image; pinned
# to the laptop's version, whose trainer remote-p15.py was written against).
# smoke: /opt/pytorch/bin/python -c "import peft, gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q peft gliner2==2.0.0
# smoke: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python remote-p15.py --work . --config p15/config.used.yaml --out p15/smoke --smoke
# needs: remote-p15.py remote-p12.py remote-p14.py p12/audit.json p12/train-far.jsonl p12/train-near.jsonl p12/train-docs.jsonl p12/eval.jsonl p12/docs.jsonl
# needs: p14/step3/eval.jsonl p14/step3/docs.jsonl p15/curve/train-419.jsonl p15/curve/train-838.jsonl p15/screen/pairs.jsonl p15/config.used.yaml
cd ~/work
PY=/opt/pytorch/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fail() { echo "FAILED: $1" > logs/done-p15; exit 1; }
$PY -c "import peft, gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q peft gliner2==2.0.0 || fail install
$PY remote-p15.py --work . --config p15/config.used.yaml --out p15/out > logs/P15.log 2>&1 || fail gpu
sha256sum p15/out/p15.json p15/out/scores/*.json p15/out/screen/scores.jsonl p15/out/screen/screen.json logs/P15.log > logs/sums-p15.txt || fail sums
printf 'p15/smoke/\np15/out/adapters/\np15/out/screen/train.jsonl\np15/out/screen/val.jsonl\n' > logs/discard-p15.txt
echo ALLDONE > logs/done-p15
