#!/bin/bash
# Laptop side of J4b/J7, start to finish (10/04/2026): wait for the splitter jobs, download and verify, chain,
# build both datasets on conversation-level splits, audit them (stop if either fails), upload, and train.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-$HOME/dianoia-data/venv/Scripts/python.exe}
A=scripts/aws.sh
[ "$(bash $A wait done-k run-K.sh)" = DONE ] || { echo "wiki splitter died"; exit 1; }
[ "$(bash $A wait done-w2 run-W2.sh)" = DONE ] || { echo "window re-split died"; exit 1; }
bash $A download logs/sums-k.txt logs/sums-w2.txt
$PY scripts/jepa-chain.py --preds pred-windows2.jsonl --base "" --out chain-w2.jsonl
$PY scripts/jepa-chain.py --preds pred-wiki.jsonl --base "" --out chain-wiki.jsonl
$PY scripts/jepa-build.py units.jsonl chain-w2.jsonl --out units-j4b.jsonl
$PY scripts/jepa-build.py units.jsonl chain-w2.jsonl chain-wiki.jsonl --out units-j7.jsonl
$PY scripts/audit-selftest.py units-j4b.jsonl split/train.jsonl jepa/windows2.jsonl  # also writes the j4b audit
$PY scripts/audit-data.py units-j4b.jsonl --turns split/train.jsonl jepa/windows2.jsonl
$PY scripts/audit-data.py units-j7.jsonl --turns split/train.jsonl jepa/windows2.jsonl jepa/wiki-windows.jsonl
for R in j4b j7; do
  grep -q '"pass": true' "$HOME/dianoia-data/work/jepa/units-$R.jsonl.audit.json" || { echo "AUDIT FAILED: units-$R"; exit 1; }
done
bash $A upload jepa/units-j4b.jsonl jepa/units-j4b.jsonl.audit.json jepa/units-j7.jsonl jepa/units-j7.jsonl.audit.json remote-jepa.py
bash $A run run-J7.sh
echo "J4b and J7 training"
