"""A tiny synthetic PatentsView in the real files' format, for the pipeline's dry run (10/05/2026).

    python make_fixture.py <dir>      then: snakemake --configfile config.yaml smoke.yaml --config data=<dir> ...

Same file names, columns, quoting and zip packaging as the PatentsView release, so the smoke run exercises every
parser and join the real run will. Abstracts are word salad from section-specific nouns and shared verbs.
"""

import random
import sys
import zipfile
from pathlib import Path

rng = random.Random(0)
out = Path(sys.argv[1])
raw = out / "raw" / "patentsview"
raw.mkdir(parents=True, exist_ok=True)
S = "ABCDEFGH"
NOUNS = {s: [f"{w}" for w in ws.split()] for s, ws in zip(S, [
    "wheat clover farmer tractor soil seed", "drill blade lathe press conveyor bolt", "polymer catalyst acid solvent enzyme salt",
    "fiber yarn loom fabric thread needle", "beam brick concrete door roof wall", "piston turbine valve pump engine boiler",
    "processor memory signal sensor circuit lens", "transistor antenna battery voltage cable diode"])}
VERBS = "rotate harvest gather heat cool cut join measure transmit store mix filter press grow".split()
GLUE = "a the of to and with in on by for".split()


def tsv_zip(name, header, rows):
    lines = ["\t".join(f'"{h}"' for h in header)]
    for r in rows:
        lines.append("\t".join(f'"{x}"' if isinstance(x, str) else str(x) for x in r))
    with zipfile.ZipFile(raw / f"{name}.tsv.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.tsv", "\n".join(lines) + "\n")
    (raw / f"{name}.tsv.zip.ok").touch()


pats = []
for i in range(24000):
    pid = str(9_000_000 + i)
    y = rng.randint(2005, 2024)
    pats.append((pid, f"{y}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", S[i % 8]))
tsv_zip("g_patent", ["patent_id", "patent_type", "patent_date", "patent_title", "wipo_kind", "num_claims", "withdrawn", "filename"],
        [(p, "utility", d, "A title", "B2", 10, 0, "x.xml") for p, d, s in pats] + [("D900001", "design", "2015-01-01", "t", "S1", 1, 0, "x")])
cpc = []
for p, d, s in pats:
    cpc.append((p, 0, s, s + "01", s + "01B", s + "01B1/00", "inventional"))
    if rng.random() < 0.2:
        cpc.append((p, 1, rng.choice(S + "Y"), "X01", "X01B", "X01B1/00", "additional"))
tsv_zip("g_cpc_current", ["patent_id", "cpc_sequence", "cpc_section", "cpc_class", "cpc_subclass", "cpc_group", "cpc_type"], cpc)
cites = []
by_date = sorted(pats, key=lambda x: x[1])
for k, (p, d, s) in enumerate(by_date):
    for j, (q, dq, sq) in enumerate(rng.sample(by_date[:k], min(k, 6))):
        cites.append((p, j, q, dq, "", "A", rng.choice(["cited by applicant", "cited by examiner"])))
tsv_zip("g_us_patent_citation", ["patent_id", "citation_sequence", "citation_patent_id", "citation_date", "record_name", "wipo_kind", "citation_category"], cites)
abst = []
for p, d, s in pats:
    words = []
    for _ in range(rng.randint(35, 60)):
        r = rng.random()
        words.append(rng.choice(NOUNS[s]) if r < 0.35 else rng.choice(NOUNS[rng.choice(S)]) if r < 0.45 else rng.choice(VERBS) + "s" if r < 0.6 else rng.choice(GLUE))
    abst.append((p, "A device " + " ".join(words) + "."))
tsv_zip("g_patent_abstract", ["patent_id", "patent_abstract"], abst)
print(f"fixture: {len(pats)} patents, {len(cites)} citations in {raw}")
