#!/bin/bash
# P13S: phase 13 step 2d (remote-p13-filter.py): fine-tune the scoring filter, score the holdout and every far pair.
# Started only by scripts/p13-filter-gpu.sh. Resumable: epochs are checkpointed and score files are appended.
# Job contract: logs/done-p13s holds ALLDONE only if every step succeeded.
# smoke: (/opt/pytorch/bin/python -c "import yaml" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q pyyaml) && /opt/pytorch/bin/python remote-p13-filter.py --work p13/filter --config p13/config.used.yaml --out p13/filter/smoke --smoke
# needs: remote-p13-filter.py p13/config.used.yaml p13/filter/train.jsonl p13/filter/audit.json p13/far-text.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p13s; exit 1; }
$PY remote-p13-filter.py --work p13/filter --config p13/config.used.yaml --out p13/filter/out > logs/P13S.log 2>&1 || fail gpu
sha256sum p13/filter/out/holdout-scores.jsonl p13/filter/out/far-scores.jsonl p13/filter/out/train-log.json logs/P13S.log > logs/sums-p13s.txt || fail sums
printf 'p13/filter/smoke/\np13/filter/out/ckpt/\n' > logs/discard-p13s.txt
echo ALLDONE > logs/done-p13s
