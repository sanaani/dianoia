import json, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, r"C:\users\shaun\onedrive\documents\dianoia\scripts")
from local_encode import Encoder
from collections import defaultdict
g = defaultdict(lambda: {"pos": [], "neg": []})
for l in open(r"C:\Users\shaun\dianoia-data\work\pairs\test-merge.jsonl", encoding="utf-8"):
    r = json.loads(l); g[r["a"]]["pos" if r["rel"] == "same" else "neg"].append(r["b"])
items = [(m, v["pos"][0], v["pos"][1], v["pos"][2], v["neg"]) for m, v in g.items() if len(v["pos"]) == 3 and v["neg"]]
texts = sorted({t for it in items for t in (*it[:4], *it[4])})
print(len(items), len(texts), flush=True)
for name, ad in (("S1", r"C:\Users\shaun\dianoia-data\work\ablation\S1\lora.pt"),):
    t0 = time.time(); e = Encoder(ad); V = dict(zip(texts, e.encode(texts))); print("enc min", round((time.time()-t0)/60, 1), flush=True)
    c = lambda a, b: float(a @ b)
    acc = defaultdict(float)
    for m, x, y, m2, E in items:
        s = F.normalize(V[x] + V[y], dim=0)
        ce = max(c(V[m], V[k]) for k in E)
        acc["cos_part"] += max(c(V[m], V[x]), c(V[m], V[y])); acc["cos_sum"] += c(V[m], s); acc["cos_other"] += c(V[m], V[m2]); acc["cos_edit"] += ce
        acc["sum_beats_edit"] += c(V[m], s) > ce; acc["other_beats_edit"] += c(V[m], V[m2]) > ce; acc["part_beats_edit"] += max(c(V[m], V[x]), c(V[m], V[y])) > ce
        # does the sum of parts still beat the sum built from an edited part? edit of x: compare cos(m-sum) vs cos(edit, sum)
        acc["sum_closer_to_merge_than_to_edit"] += c(V[m], s) > max(c(V[k], s) for k in E)
    print(name, {k: round(v / len(items), 3) for k, v in acc.items()})
