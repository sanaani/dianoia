#!/bin/bash
# P14F: phase 14 step 1's cosine pass (p13-data.py far --until cosine) on one GPU, from the laptop run's checkpoint
# (10/08/2026). The laptop CPU scored about 22k pairs an hour; this is minutes. Started only by scripts/p14-far-gpu.sh.
# Resumable: the cosine pass records its progress per chunk and a restart continues from it.
# Job contract: logs/done-p14f holds ALLDONE only if every step succeeded.
# smoke: mkdir -p p14gs/far-tmp && head -300 p14g/far-tmp/low.jsonl > p14gs/far-tmp/low.jsonl && cp p14g/far-tmp/low.done.json p14gs/far-tmp/ && (/opt/pytorch/bin/python -c "import sentence_transformers, polars, yaml" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q sentence-transformers polars pyyaml) && /opt/pytorch/bin/python p13-data.py far --data ~ --config p14g/config.used.yaml --out p14gs --batch 256 --device cuda --until cosine
# needs: p13-data.py p10-label.py p10-data.py p12/medians.json p14g/config.used.yaml p14g/far-tmp/low.jsonl p14g/far-tmp/low.done.json p14g/far-tmp/cosine.progress.json p14g/far-tmp/cosine.start.json
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p14f; exit 1; }
$PY -c "import sentence_transformers, polars, yaml" 2>/dev/null || sudo $PY -m pip install -q sentence-transformers polars pyyaml || fail install
$PY p13-data.py far --data ~ --config p14g/config.used.yaml --out p14g --batch 256 --device cuda --until cosine > logs/P14F.log 2>&1 || fail gpu
sha256sum p14g/far-tmp/far-part.jsonl p14g/far-tmp/cosine.progress.json logs/P14F.log > logs/sums-p14f.txt || fail sums
printf 'p14gs/\n' > logs/discard-p14f.txt
echo ALLDONE > logs/done-p14f
