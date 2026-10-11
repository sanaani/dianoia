#!/bin/bash
# P11: the phase 11 GPU stage (remote-p11.py): LoRA training at three sizes, descriptions, retrieval.
# Started only by pipelines/phase11 (scripts/p11-gpu.sh), after the laptop dry run (rule smoke_gpu) passed.
# Resumable: remote-p11.py skips every stage whose output exists.
# Job contract: logs/done-p11 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p11.py --work p11 --config p11/config.used.yaml --out p11/smoke --smoke
# needs: remote-p11.py p11/train.jsonl p11/eval.jsonl p11/docs.jsonl p11/audit.json p11/config.used.yaml
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p11; exit 1; }
$PY -c "import peft" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q peft || fail install
$PY remote-p11.py --work p11 --config p11/config.used.yaml --out p11/out > logs/P11.log 2>&1 || fail gpu
sha256sum p11/out/p11.json p11/out/gen/*.jsonl logs/P11.log > logs/sums-p11.txt || fail sums
# Adapters (about 70 MB each) are kept: a later phase may reuse the trained model.
sha256sum p11/out/adapters/*/* >> logs/sums-p11.txt || fail sums
printf 'p11/smoke/\np11/out/emb_docs.pt\n' > logs/discard-p11.txt
echo ALLDONE > logs/done-p11
