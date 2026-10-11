#!/bin/bash
# vLLM splitter on new wiki windows (10,487 new + 200 already split by HF, to check agreement), after J7.
# Job contract: logs/done-v holds ALLDONE only if every step succeeded.
cd ~/work
until [ -f logs/done-j7 ]; do sleep 30; done
/home/ubuntu/vllm-env/bin/python remote-split-vllm.py --eval jepa/wiki-windows-new.jsonl --out jepa/pred-wiki-new.jsonl \
  --check jepa/pred-wiki.jsonl > logs/V.log 2>&1 || { echo FAILED > logs/done-v; exit 1; }
sha256sum jepa/pred-wiki-new.jsonl jepa/wiki-windows-new.jsonl logs/V.log > logs/sums-v.txt && echo ALLDONE > logs/done-v || echo FAILED > logs/done-v
