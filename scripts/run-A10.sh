#!/bin/bash
# Self-test and audit of units-j10 on the instance: the laptop ran out of memory on them (10/04/2026).
# Job contract: logs/done-a10 holds ALLDONE only if the self-test and the audit both pass.
# needs: audit-data.py audit-selftest.py
cd ~/work
export DIANOIA_DATA=/home/ubuntu PYTHONPATH=/home/ubuntu/repo/src
PY=/opt/pytorch/bin/python
$PY -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || sudo $PY -m spacy download en_core_web_sm > logs/spacy-install.log 2>&1
T="split/train.jsonl jepa/windows2.jsonl jepa/wiki-windows-dedup.jsonl jepa/wc2018-clean-post.jsonl jepa/new2-clean-post.jsonl"
$PY audit-selftest.py units-j10.jsonl $T > logs/A10-selftest.log 2>&1 || { echo FAILED > logs/done-a10; exit 1; }
$PY audit-data.py units-j10.jsonl --turns $T > logs/A10.log 2>&1 || { echo FAILED > logs/done-a10; exit 1; }
grep -q '"pass": true' jepa/units-j10.jsonl.audit.json || { echo FAILED > logs/done-a10; exit 1; }
sha256sum jepa/units-j10.jsonl.audit.json logs/A10.log logs/A10-selftest.log > logs/sums-a10.txt && echo ALLDONE > logs/done-a10
