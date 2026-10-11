#!/bin/bash
# D1: Fastino's GLiNER2.5-Decide fine-tuned on each outcome, against idea space, on the same held-out pairs
# (remote-decide.py). GitHub first (G1b's features), then ChangeMyView (C1's). The smoke run (--smoke 40) passed first.
# Job contract: logs/done-d1 holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-decide.py --pairs cmv/pairs.jsonl --texts cmv/texts.jsonl --features cmv/C1/features.jsonl --labels "changed the view,did not" --out cmv/D1-smoke --smoke 40
# needs: remote-decide.py gh/g1/pairs.jsonl gh/g1/texts.jsonl gh/G1/features.jsonl cmv/pairs.jsonl cmv/texts.jsonl cmv/C1/features.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-d1; exit 1; }
$PY remote-decide.py --pairs gh/g1/pairs.jsonl --texts gh/g1/texts.jsonl --features gh/G1/features.jsonl \
  --labels "clean,reverted" --out gh/D1 > logs/D1-gh.log 2>&1 || fail gh
$PY remote-decide.py --pairs cmv/pairs.jsonl --texts cmv/texts.jsonl --features cmv/C1/features.jsonl \
  --labels "changed the view,did not" --out cmv/D1 > logs/D1-cmv.log 2>&1 || fail cmv
sha256sum gh/D1/decide.json gh/D1/scores.jsonl cmv/D1/decide.json cmv/D1/scores.jsonl logs/D1-gh.log logs/D1-cmv.log > logs/sums-d1.txt || fail sums
echo ALLDONE > logs/done-d1
