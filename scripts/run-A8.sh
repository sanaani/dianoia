#!/bin/bash
# Self-test and audit of units-j8 on the instance: the laptop ran out of memory on them (10/04/2026).
# Job contract: logs/done-a8 holds ALLDONE only if the self-test and the audit both pass.
# needs: audit-data.py audit-selftest.py
cd ~/work
export DIANOIA_DATA=/home/ubuntu PYTHONPATH=/home/ubuntu/repo/src
PY=/opt/pytorch/bin/python
$PY -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || sudo $PY -m spacy download en_core_web_sm > logs/spacy-install.log 2>&1
T="split/train.jsonl jepa/windows2.jsonl jepa/wiki-windows-dedup.jsonl jepa/wc2018-clean-post.jsonl"
$PY audit-selftest.py units-j8.jsonl $T > logs/A8-selftest.log 2>&1 || { echo FAILED > logs/done-a8; exit 1; }
$PY audit-data.py units-j8.jsonl --turns $T > logs/A8.log 2>&1 || { echo FAILED > logs/done-a8; exit 1; }
grep -q '"pass": true' jepa/units-j8.jsonl.audit.json || { echo FAILED > logs/done-a8; exit 1; }
sha256sum jepa/units-j8.jsonl.audit.json logs/A8.log logs/A8-selftest.log > logs/sums-a8.txt && echo ALLDONE > logs/done-a8
