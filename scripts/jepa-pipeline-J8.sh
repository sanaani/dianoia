#!/bin/bash
# Laptop side of J8 (10/04/2026): wait for the WikiConv split, download and verify, clean (near-copies, template ideas,
# held-out cross-posts), chain, build on conversation-level splits with J7's sources; then on the instance (the
# laptop ran out of memory on a 111 MB audit) self-test and audit, download the audit and check it names this file;
# then upload and train. Splitter path per source: J7 sources HF, WikiConv vLLM in every split (process.md).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-$HOME/dianoia-data/venv/Scripts/python.exe}
A=scripts/aws.sh
W=$(cygpath -m "$HOME/dianoia-data/work")  # Windows Python cannot open /c/... paths
if [ "${SKIP_SPLIT:-0}" != 1 ]; then
  [ "$(bash $A wait done-wc run-WC.sh)" = DONE ] || { echo "WikiConv split failed"; exit 1; }
  bash $A download logs/sums-wc.txt
  $PY scripts/clean-windows.py post pred-wc2018.jsonl pred-wc2018-post.jsonl wc2018-clean.jsonl
fi
[ "${SKIP_CHAIN:-0}" = 1 ] || $PY scripts/jepa-chain.py --preds pred-wc2018-post.jsonl --base "" --out chain-wc.jsonl --workers 3
$PY scripts/jepa-build.py units.jsonl chain-w2.jsonl chain-wiki.jsonl chain-wc.jsonl --out units-j8.jsonl
bash $A upload audit-data.py audit-selftest.py run-A8.sh jepa/units-j8.jsonl jepa/units-j8.jsonl.manifest.json \
  split/train.jsonl jepa/windows2.jsonl jepa/wiki-windows-dedup.jsonl jepa/wc2018-clean-post.jsonl
bash $A run run-A8.sh
[ "$(bash $A wait done-a8 run-A8.sh)" = DONE ] || { bash $A ssh 'tail -2 logs/A8-selftest.log'; echo "AUDIT FAILED: units-j8"; exit 1; }
bash $A download logs/sums-a8.txt
$PY -c "
import json, hashlib; p=r'$W/jepa/units-j8.jsonl'; a=json.load(open(p + '.audit.json'))
assert a['pass'] and a['units_sha256'] == hashlib.sha256(open(p, 'rb').read()).hexdigest(), 'audit does not name this file'
print('audit passes for this file')"
bash $A upload ablation/S1/lora.pt jepa/units-j7.jsonl jepa/units-j7.jsonl.manifest.json jepa/units-j7.jsonl.audit.json remote-jepa.py
bash $A run run-J8.sh
[ "$(bash $A wait done-j8 run-J8.sh)" = DONE ] || { echo "J8 failed"; exit 1; }
bash $A download logs/sums-j8.txt
echo "J8 DONE"
