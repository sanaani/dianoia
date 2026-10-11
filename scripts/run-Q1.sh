#!/bin/bash
# Q1: retrieval quality of compact idea vectors (remote-quant.py) on J7's pool. Starts after J10 finishes, because a
# second process alongside J10's 1.6M-vector table ran the 15 GB instance out of memory (10/05/2026).
# Q1b (10/05/2026): rerun with the tie-break jitter on Hamming scores only.
# Job contract: logs/done-q1b holds ALLDONE only if every step succeeded.
# needs: remote-quant.py
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-q1b; exit 1; }
while [ ! -f logs/done-j10 ]; do sleep 60; done
$PY remote-quant.py --units jepa/units-j7.jsonl --vec-cache jepa/vec-cache.pt --out jepa/Q1b > logs/Q1b.log 2>&1 || fail Q1
sha256sum jepa/Q1b/quant.json logs/Q1b.log > logs/sums-q1b.txt || fail sums
echo ALLDONE > logs/done-q1b
