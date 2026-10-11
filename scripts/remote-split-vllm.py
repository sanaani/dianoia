"""Splitter q3 served through vLLM: the same model, adapter, prompt and greedy decoding as
``remote-split.py predict-qwen``, for the 10x data build (10/04/2026).

At HF ``generate`` speed (100-260 windows a minute on one A10G) ten times today's data would take 25-30 GPU hours.
vLLM batches with paged attention and continuous batching. Same output format, resumable (rows already in ``--out``
are skipped) and written as it goes.

``--check <hf predictions>``: also report how often vLLM's ideas equal the HF splitter's for the same windows
(greedy decoding in different kernels can differ in a few tokens), and both speeds. Decide on agreement, not speed
alone: a faster splitter that cuts differently changes the data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path


MAX_LEN = 8192


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="split/q3", help="the LoRA adapter directory")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", default="", help="HF predictions for the same uids, to measure agreement")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    spec = importlib.util.spec_from_file_location("remote_split", Path(__file__).with_name("remote-split.py"))
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    done = {(o["uid"], o["kind"]) for o in map(json.loads, open(args.out, encoding="utf-8"))} if Path(args.out).exists() else set()
    rows = [r for r in rows if (r["uid"], r["kind"]) not in done]
    adapter_cfg = json.loads((Path(args.model) / "adapter_config.json").read_text())
    llm = LLM(model=rs.QWEN, enable_lora=True, max_lora_rank=max(16, adapter_cfg.get("r", 16)), dtype="bfloat16",
              max_model_len=MAX_LEN, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0.0, max_tokens=700)
    # V2 (10/04/2026) died on one 2,049-token prompt at max_model_len 2048. Raised to 8192 (HF generate has no limit);
    # a prompt that still leaves no room for the 700 output tokens is skipped and counted rather than killing the run.
    tok = llm.get_tokenizer()
    fits = [len(tok(rs.qprompt(r["turns"]))["input_ids"]) + 700 <= MAX_LEN for r in rows]
    skipped = [r["uid"] for r, ok in zip(rows, fits) if not ok]
    rows = [r for r, ok in zip(rows, fits) if ok]
    print(json.dumps({"skipped_too_long": len(skipped), "examples": skipped[:5]}), flush=True)
    lora = LoRARequest("q3", 1, args.model)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as fh:
        for s in range(0, len(rows), 2000):
            b = rows[s : s + 2000]
            outs = llm.generate([rs.qprompt(r["turns"]) for r in b], params, lora_request=lora, use_tqdm=False)
            for r, o in zip(b, outs):
                t = o.outputs[0].text
                ideas = list(dict.fromkeys(x.strip() for x in t.split("\n") if x.strip()))
                fh.write(json.dumps({"uid": r["uid"], "kind": r["kind"], "medium": r["medium"], "ideas": ideas, "raw": t},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps({"rows": s + len(b), "of": len(rows), "minutes": round((time.time() - t0) / 60, 2)}), flush=True)
    minutes = (time.time() - t0) / 60
    rep = {"rows": len(rows), "skipped_too_long": len(skipped), "minutes": round(minutes, 2), "rows_per_min": round(len(rows) / max(minutes, 1e-6), 1)}
    if args.check:
        hf = {json.loads(l)["uid"]: json.loads(l)["ideas"] for l in open(args.check, encoding="utf-8")}
        vl = {json.loads(l)["uid"]: json.loads(l)["ideas"] for l in open(args.out, encoding="utf-8")}
        both = [u for u in vl if u in hf]
        rep["agreement_exact_unit"] = round(sum(vl[u] == hf[u] for u in both) / max(len(both), 1), 4)
        hi = sum(len(set(vl[u]) & set(hf[u])) for u in both)
        rep["agreement_idea_f1"] = round(2 * hi / max(sum(len(set(vl[u])) + len(set(hf[u])) for u in both), 1), 4)
        rep["same_count"] = round(sum(len(vl[u]) == len(hf[u]) for u in both) / max(len(both), 1), 4)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
