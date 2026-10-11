#!/bin/bash
# G1: the GitHub outcome test. Splitter q3 over the pull requests' text (gh-build.py windows, resumable), then
# remote-cmv.py on the pairs: "win" is the PR that was not reverted. Text models read 768 tokens of each PR
# (descriptions plus review discussion run longer than ChangeMyView replies). Adds the text-plus-idea-scores model.
# On demand: the text models keep no checkpoint.
# Job contract: logs/done-g1 holds ALLDONE only if every step succeeded.
# needs: remote-split-vllm.py remote-split.py split/q3 remote-cmv.py remote-jepa.py remote-encoder.py ablation/S1/lora.pt jepa/J10/predictor.pt gh/g1/windows.jsonl gh/g1/pairs.jsonl gh/g1/texts.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
fail() { echo "FAILED: $1" > logs/done-g1; exit 1; }
export PATH=/home/ubuntu/vllm-env/bin:$PATH  # vLLM JIT-compiles a kernel with ninja
/home/ubuntu/vllm-env/bin/python remote-split-vllm.py --eval gh/g1/windows.jsonl --out gh/g1/pred.jsonl > logs/G1-split.log 2>&1 || fail split
$PY remote-cmv.py --pred gh/g1/pred.jsonl --pairs gh/g1/pairs.jsonl --texts gh/g1/texts.jsonl --adapter ablation/S1/lora.pt \
  --predictor jepa/J10/predictor.pt --out gh/G1 --reply-tokens 768 > logs/G1.log 2>&1 || fail model
sha256sum gh/g1/pred.jsonl gh/G1/cmv.json gh/G1/features.jsonl logs/G1-split.log logs/G1.log > logs/sums-g1.txt || fail sums
echo ALLDONE > logs/done-g1
