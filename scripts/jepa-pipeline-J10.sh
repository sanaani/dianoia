#!/bin/bash
# Laptop side of J10 (10/05/2026), as J8's: wait for the split of the new sources (#ubuntu auto-disentangled, FOMC),
# download and verify, clean (near-copies, template ideas, held-out cross-posts), chain, build with every earlier
# source; self-test and audit on the instance, download the audit and check it names this file; upload and train.
# Splitter path per source: J7 sources HF; WikiConv, #ubuntu-auto and FOMC vLLM in every split (process.md).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-$HOME/dianoia-data/venv/Scripts/python.exe}
A=scripts/aws.sh
W=$(cygpath -m "$HOME/dianoia-data/work")  # Windows Python cannot open /c/... paths
if [ "${SKIP_SPLIT:-0}" != 1 ]; then
  [ "$(bash $A wait done-wc2 run-WC2.sh)" = DONE ] || { echo "new-source split failed"; exit 1; }
  bash $A download logs/sums-wc2.txt
  $PY scripts/clean-windows.py post pred-new2.jsonl pred-new2-post.jsonl new2-clean.jsonl
fi
[ "${SKIP_CHAIN:-0}" = 1 ] || $PY scripts/jepa-chain.py --preds pred-new2-post.jsonl --base "" --out chain-new2.jsonl --workers 3
$PY scripts/jepa-build.py units.jsonl chain-w2.jsonl chain-wiki.jsonl chain-wc.jsonl chain-new2.jsonl --out units-j10.jsonl
bash $A upload audit-data.py audit-selftest.py run-A10.sh jepa/units-j10.jsonl jepa/units-j10.jsonl.manifest.json \
  split/train.jsonl jepa/windows2.jsonl jepa/wiki-windows-dedup.jsonl jepa/wc2018-clean-post.jsonl jepa/new2-clean-post.jsonl
bash $A sync-src
bash $A run run-A10.sh
[ "$(bash $A wait done-a10 run-A10.sh)" = DONE ] || { bash $A ssh 'tail -2 logs/A10-selftest.log'; echo "AUDIT FAILED: units-j10"; exit 1; }
bash $A download logs/sums-a10.txt
$PY -c "
import json, hashlib; p=r'$W/jepa/units-j10.jsonl'; a=json.load(open(p + '.audit.json'))
assert a['pass'] and a['units_sha256'] == hashlib.sha256(open(p, 'rb').read()).hexdigest(), 'audit does not name this file'
print('audit passes for this file')"
bash $A upload ablation/S1/lora.pt jepa/units-j7.jsonl jepa/units-j7.jsonl.manifest.json jepa/units-j7.jsonl.audit.json remote-jepa.py
bash $A run run-J10.sh
[ "$(bash $A wait done-j10 run-J10.sh)" = DONE ] || { echo "J10 failed"; exit 1; }
bash $A download logs/sums-j10.txt
echo "J10 DONE"
