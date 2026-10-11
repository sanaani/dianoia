#!/bin/bash
# Splitter q4: retrain on the units plus their one-for-one paraphrases, predict every check version, score q4 and
# rescore q3 and Sonnet with the segmentation metric.
cd ~/work
PY=/opt/pytorch/bin/python
$PY remote-split.py train-qwen --train split/train.jsonl,split/para.jsonl --out split/q4 --epochs 2 > logs/q4-train.log 2>&1 || exit 1
$PY remote-split.py predict-qwen --model split/q4 --eval split/versions-all.jsonl --out split/pred-versions-q4.jsonl --batch 32 > logs/q4-predict.log 2>&1
$PY remote-split.py score --preds split/pred-versions-q4.jsonl --out split/score-versions-q4.json > logs/q4-score.log 2>&1
$PY remote-split.py score --preds split/pred-versions.jsonl --out split/score-versions-q3.json > logs/q3-score.log 2>&1
$PY remote-split.py score --preds split/eval.jsonl --out split/score-versions-sonnet.json > logs/sonnet-score.log 2>&1
sha256sum split/pred-versions-q4.jsonl split/score-versions-*.json > logs/sums-q.txt
echo ALLDONE > logs/done-q
