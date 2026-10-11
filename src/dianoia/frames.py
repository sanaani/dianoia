"""Intent frames: a wording-independent target for the decoder.

A frame is a fixed set of slots with a controlled vocabulary, serialized in a fixed order, so that two
sentences meaning the same thing produce the same string however they are worded:

    act=request; agent=you; action=call; patient=me; topic=remaining tasks; time=-; negated=no; certainty=-; quantity=-

Names of particular people, organizations, places, products, deals and documents are typed, numbered
placeholders ([PERSON_1], ...), numbered by first appearance, exactly as in the canonical sentences.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from dianoia import llm

SLOTS = ["act", "agent", "action", "patient", "recipient", "topic", "time", "negated", "certainty", "quantity"]
CORE = ["act", "agent", "action", "patient", "negated", "certainty"]
ACTS = ["report", "request", "commit", "question", "propose", "approve", "reject", "thank", "apologize", "greet", "other"]
CERTAINTY = ["-", "certain", "likely", "possible", "unlikely", "required", "permitted"]

PROMPT = """Turn each sentence from business email into an INTENT FRAME: what the sentence does and claims, with \
all wording removed. Two sentences that mean the same thing must produce IDENTICAL frames, however differently \
they are worded. Fill these slots:

- act: one of {acts}. (A polite request phrased as a question is "request".)
- agent: who does the main action: I, we, you, they, a placeholder, or a short lowercase common noun ("the board").
- action: ONE base-form verb, the simplest common English verb for the action. Map synonyms to it: phone/ring/give a \
call -> call; talk/speak/converse/have a conversation -> speak; look over/examine/go through -> review; \
forward/pass on/send over -> send; okay/sign off on/approve -> approve. Use "be" for states, "have" for possession.
- patient: what or whom the action directly affects, as a short lowercase noun phrase with no articles or adjectives \
unless they change the meaning ("contract", "revised contract" only if revision matters). "-" if none.
- recipient: to or with whom ("me", "you", a placeholder). "-" if none.
- topic: what it is about, if not the patient, as a short lowercase noun phrase. "-" if none.
- time: normalized time reference ("today", "tomorrow", "this morning", "friday", "2001-08-24", "past", "future"). "-" if none.
- negated: yes or no.
- certainty: one of {certainty}.
- quantity: material numbers or amounts exactly as written, comma-separated. "-" if none.

Replace every name of a particular person, organization, place, product, deal or document with a typed numbered \
placeholder ([PERSON_1], [ORG_1], [PLACE_1], [PRODUCT_1], [DEAL_1], [DOC_1]), numbered by first appearance. Keep I, \
we, you and these corpus-level names as written: {literal}.

Return only a JSON array: [{{"i": <number>, "frame": {{"act": ..., "agent": ..., "action": ..., "patient": ..., \
"recipient": ..., "topic": ..., "time": ..., "negated": ..., "certainty": ..., "quantity": ...}}}}, ...]

{items}"""


def serialize(frame: dict) -> str:
    return "; ".join(f"{s}={norm(frame.get(s, '-'))}" for s in SLOTS)


def parse(text: str) -> dict:
    out = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def norm(value) -> str:
    v = str(value if value not in (None, "") else "-").strip().lower()
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"\[(person|org|place|product|deal|doc|date|amount)_(\d+)\]", lambda m: f"[{m.group(1).upper()}_{m.group(2)}]", v)
    return v.strip(" .") or "-"


def frames_for(texts: list[str], literal: list[str], *, model: str = "sonnet", batch: int = 20, workers: int = 4) -> list[str | None]:
    """One serialized frame per text (None where the model returned nothing usable)."""
    prompts, spans = [], []
    for s in range(0, len(texts), batch):
        chunk = texts[s : s + batch]
        present = sorted({e for t in chunk for e in literal if re.search(rf"\b{re.escape(e)}\b", t)})
        prompts.append(PROMPT.format(acts=ACTS, certainty=CERTAINTY, literal=present or "none",
                                     items="\n".join(f"{k}. {t}" for k, t in enumerate(chunk))))
        spans.append((s, len(chunk)))

    def safe(p):
        try:
            return llm.ask_json(p, model=model)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(safe, prompts))
    out: list[str | None] = [None] * len(texts)
    for (start, n), ans in zip(spans, answers):
        for a in ans if isinstance(ans, list) else []:
            if isinstance(a, dict) and str(a.get("i", "")).isdigit() and isinstance(a.get("frame"), dict):
                k = int(a["i"])
                if 0 <= k < n:
                    out[start + k] = serialize(a["frame"])
    return out


def agreement(a: str, b: str) -> dict:
    fa, fb = parse(a), parse(b)
    per = {s: fa.get(s, "-") == fb.get(s, "-") for s in SLOTS}
    per["core"] = all(per[s] for s in CORE)
    per["all"] = all(per[s] for s in SLOTS)
    return per


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)
