#!/bin/bash
# P10D4: retrain the P10D2 Fastino model (same data and seed) and score the whole val/test pool (75,053 pairs), so
# Fastino can screen the pool and Sonnet labels only the pairs it accepts. P10D2's model was lost when the instance
# was terminated (10/06/2026): its sums listed only the scores. This job tars the model and checksums it, so
# "aws.sh finish" downloads it.
# Job contract: logs/done-p10d4 holds ALLDONE only if every step succeeded.
# smoke: sudo /opt/pytorch/bin/pip install -q gliner2 && /opt/pytorch/bin/python remote-p10-decide.py --seed p10/label/seed.jsonl,p10/label/seed2.jsonl --labels p10/label/seed-sonnet.jsonl,p10/label/seed2-sonnet.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/pool.jsonl --out p10/decide4-smoke --smoke 10
# needs: remote-p10-decide.py p10/label/seed.jsonl p10/label/seed2.jsonl p10/label/seed-sonnet.jsonl p10/label/seed2-sonnet.jsonl p10/label/gold-all.jsonl p10/label/pool.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-p10d4; exit 1; }
$PY -c "import gliner2" 2>/dev/null || sudo /opt/pytorch/bin/pip install -q gliner2 || fail install
$PY remote-p10-decide.py --seed p10/label/seed.jsonl,p10/label/seed2.jsonl --labels p10/label/seed-sonnet.jsonl,p10/label/seed2-sonnet.jsonl \
  --gold p10/label/gold-all.jsonl --pool p10/label/pool.jsonl --out p10/decide4 > logs/P10D4.log 2>&1 || fail decide
grep -vE "it/s|Warning" logs/P10D4.log > logs/P10D4-clean.log
tar -czf p10/decide4/model-final.tgz -C p10/decide4/model final || fail tar
# Left behind on purpose (aws-finish.sh refuses to terminate over any other undownloaded file): the training files
# are rebuilt from the uploads, and the model ships as the tarball. (Written by hand on the P10D4 instance, 10/06.)
printf 'p10/decide4/train.jsonl\np10/decide4/val.jsonl\np10/decide4/model/\n' > logs/discard-p10d4.txt
sha256sum p10/decide4/decide.json p10/decide4/pool-scores.jsonl p10/decide4/model-final.tgz logs/P10D4-clean.log > logs/sums-p10d4.txt || fail sums
echo ALLDONE > logs/done-p10d4
