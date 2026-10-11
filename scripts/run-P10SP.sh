#!/bin/bash
# P10SP (10/06/2026): what limits Fastino scoring speed? One pair at a time and batch_classify_text both ran 30 pairs/s
# with identical scores (P10D7), so the GPU is not the batching limit. This scores 1,500 pool pairs with the base model
# as (a) one float32 process, (b) one bfloat16 process (scores compared with a), (c) three float32 processes at once,
# sampling GPU and CPU use every 2 s. Decides how to make the 5.08M-pair expansion affordable.
# Job contract: logs/done-p10sp holds ALLDONE only if every step succeeded.
# smoke: /opt/pytorch/bin/python remote-p10-decide.py --target screen --seed p10/label/stage3-pairs.jsonl --labels p10/label/stage3-labels.jsonl --gold p10/label/gold-all.jsonl --pool p10/label/pool.jsonl --out p10/sp-smoke --score-only fastino/GLiNER2.5-Decide --bf16 --smoke 20
# needs: remote-p10-decide.py p10/label/pool.jsonl
cd ~/work
PY=/opt/pytorch/bin/python
L=p10/label
fail() { echo "FAILED: $1" > logs/done-p10sp; exit 1; }
head -n 4500 $L/pool.jsonl > p10/sp-pool.jsonl
for i in 0 1 2; do sed -n "$((i*1500+1)),$((i*1500+1500))p" p10/sp-pool.jsonl > p10/sp-part$i.jsonl; done
ARGS="--target screen --seed $L/stage3-pairs.jsonl --labels $L/stage3-labels.jsonl --gold $L/gold-all.jsonl --score-only fastino/GLiNER2.5-Decide"
watch_use() { while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) $(top -bn1 | awk '/Cpu/{print 100-$8}')"; sleep 2; done; }
run() {  # run <label> <command...>: time it and record utilization
  N=$1; shift
  watch_use > logs/P10SP-$N-use.txt & W=$!
  T0=$(date +%s.%N); "$@" || { kill $W; fail $N; }; T1=$(date +%s.%N); kill $W
  echo "{\"run\": \"$N\", \"seconds\": $(echo "$T1 - $T0" | bc)}" >> logs/P10SP.txt
}
run fp32 $PY remote-p10-decide.py $ARGS --pool p10/sp-part0.jsonl --out p10/sp-fp32 > logs/P10SP-fp32.log 2>&1
run bf16 $PY remote-p10-decide.py $ARGS --pool p10/sp-part0.jsonl --out p10/sp-bf16 --bf16 > logs/P10SP-bf16.log 2>&1
# Wait on these PIDs only: a bare "wait" also waited for run's never-ending watch_use loop, and the job hung for an
# hour after the three processes had finished (10/06/2026).
par() { P=""; for i in 0 1 2; do $PY remote-p10-decide.py $ARGS --pool p10/sp-part$i.jsonl --out p10/sp-par$i > logs/P10SP-par$i.log 2>&1 & P="$P $!"; done; wait $P; }
run par3 par
$PY - >> logs/P10SP.txt <<'EOF'
import json
a = {json.loads(l)["id"]: json.loads(l)["p"] for l in open("p10/sp-fp32/pool-scores.jsonl")}
b = {json.loads(l)["id"]: json.loads(l)["p"] for l in open("p10/sp-bf16/pool-scores.jsonl")}
d = [abs(a[i] - b[i]) for i in a]
print(json.dumps({"bf16_vs_fp32": {"pairs": len(d), "max_abs_diff": round(max(d), 4),
                                   "same_side_of_0.5": sum((a[i] >= .5) == (b[i] >= .5) for i in a)}}))
EOF
for f in logs/P10SP-*-use.txt; do awk -v n=$(basename $f) '{g+=$2; c+=$4; k++} END {printf "{\"use\": \"%s\", \"gpu_pct\": %.0f, \"cpu_pct\": %.0f}\n", n, g/k, c/k}' $f >> logs/P10SP.txt; done
cat logs/P10SP.txt
sha256sum logs/P10SP.txt > logs/sums-p10sp.txt || fail sums
printf 'p10/sp-\n' > logs/discard-p10sp.txt
echo ALLDONE > logs/done-p10sp
