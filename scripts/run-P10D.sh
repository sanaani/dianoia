#!/bin/bash
# P10D: GLiNER2.5-Decide learns Sonnet's borrowing labels (2,989 seed pairs, 2013-2017) and scores the 75,053-pair pool
# (val and test years) for pipelines/phase10 (remote-p10-decide.py). The laptop cannot train it (the optimizer state of
# a 340M model is about 4 GB), so the dry run is the smoke line here, on the instance, before the job.
# gliner2 is installed here: the image lacks it, and D1 (10/05/2026) installed it by hand.
# Job contract: logs/done-p10d holds ALLDONE only if every step succeeded.
# smoke: sudo /opt/pytorch/bin/pip install -q gliner2 && /opt/pytorch/bin/python remote-p10-decide.py --seed p10/label/seed.jsonl --labels p10/label/seed-sonnet.jsonl --gold p10/label/gold.jsonl --pool p10/label/pool.jsonl --out p10/decide-smoke --smoke 10
# needs: remote-p10-decide.py p10/label/seed.jsonl p10/label/seed-sonnet.jsonl p10/label/gold.jsonl p10/label/pool.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p10d; exit 1; }
$PY -c "import gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q gliner2 || fail install
$PY remote-p10-decide.py --seed p10/label/seed.jsonl --labels p10/label/seed-sonnet.jsonl --gold p10/label/gold.jsonl \
  --pool p10/label/pool.jsonl --out p10/decide > logs/P10D.log 2>&1 || fail decide
sha256sum p10/decide/decide.json p10/decide/pool-scores.jsonl logs/P10D.log > logs/sums-p10d.txt || fail sums
echo ALLDONE > logs/done-p10d
