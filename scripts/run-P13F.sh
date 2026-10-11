#!/bin/bash
# P13F: phase 13 step 2's cosine pass (p13-data.py far --until cosine) on one GPU, from the laptop run's checkpoint
# (10/08/2026). The laptop CPU scored about 22k pairs an hour; this is minutes. Started only by scripts/p13-far-gpu.sh.
# Resumable: the cosine pass records its progress per chunk and a restart continues from it.
# Job contract: logs/done-p13f holds ALLDONE only if every step succeeded.
# smoke: mkdir -p p13gs/far-tmp && head -300 p13g/far-tmp/low.jsonl > p13gs/far-tmp/low.jsonl && cp p13g/far-tmp/low.done.json p13gs/far-tmp/ && (/opt/pytorch/bin/python -c "import sentence_transformers, polars, yaml" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q sentence-transformers polars pyyaml) && /opt/pytorch/bin/python p13-data.py far --data ~ --config p13g/config.used.yaml --out p13gs --batch 256 --device cuda --until cosine
# needs: p13-data.py p10-label.py p10-data.py p12/medians.json p13g/config.used.yaml p13g/far-tmp/low.jsonl p13g/far-tmp/low.done.json p13g/far-tmp/cosine.progress.json p13g/far-tmp/cosine.start.json
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p13f; exit 1; }
$PY -c "import sentence_transformers, polars, yaml" 2>/dev/null || sudo $PY -m pip install -q sentence-transformers polars pyyaml || fail install
$PY p13-data.py far --data ~ --config p13g/config.used.yaml --out p13g --batch 256 --device cuda --until cosine > logs/P13F.log 2>&1 || fail gpu
sha256sum p13g/far-tmp/far-part.jsonl p13g/far-tmp/cosine.progress.json logs/P13F.log > logs/sums-p13f.txt || fail sums
printf 'p13gs/\n' > logs/discard-p13f.txt
echo ALLDONE > logs/done-p13f
