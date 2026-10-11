"""Idea objects: the structure of what a unit of communication says, built from established linguistics.

An idea object is a small graph:

- **events**: one per predication the unit asserts, asks, requests, promises... Each has a dialogue act
  (ISO 24617-2 general-purpose and social functions), a predicate with PropBank-style arguments (ARG0
  agent, ARG1 patient/theme, ARG2 recipient/beneficiary/end state), polarity (ARGM-NEG), modality
  (ARGM-MOD, normalized) and time (ARGM-TMP, normalized). Events are found by meaning, not punctuation.
- **relations**: each later event attaches to an earlier one with an SDRT relation (the STAC label set,
  which covers both monologue and dialogue), so a split turn or a missing period changes nothing.
  Each label also has a coarse class: the four PDTB top-level senses plus Interaction for the
  dialogue-only relations.
- **center**: the one event the unit is for (the RST nucleus of the whole).
- **gist**: one plain sentence.

Speakers are letters (A, B, ...). ``I`` and ``you`` resolve to letters where the turn makes it clear.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dianoia import llm

ACTS = {  # ISO 24617-2 function -> coarse class
    "inform": "providing", "answer": "providing", "agreement": "providing", "disagreement": "providing",
    "correction": "providing", "confirm": "providing", "disconfirm": "providing",
    "propositionalQuestion": "seeking", "setQuestion": "seeking", "choiceQuestion": "seeking", "checkQuestion": "seeking",
    "request": "directive", "instruct": "directive", "suggest": "directive",
    "offer": "commissive", "promise": "commissive", "acceptRequest": "commissive", "declineRequest": "commissive",
    "acceptSuggestion": "commissive", "declineSuggestion": "commissive",
    "feedback": "feedback",
    "greeting": "social", "thanking": "social", "apology": "social", "goodbye": "social",
    "other": "other",
}
MODS = ["none", "possible", "likely", "certain", "necessary", "obligated", "able", "intend", "want", "permitted"]
MOD_COARSE = {"none": "none", "certain": "none", "possible": "uncertain", "likely": "uncertain", "necessary": "must",
              "obligated": "must", "able": "can", "permitted": "can", "intend": "will", "want": "will"}
RELATIONS = {  # STAC / SDRT label -> coarse class (PDTB level-1 sense, or Interaction)
    "Narration": "Temporal",
    "Result": "Contingency", "Explanation": "Contingency", "Conditional": "Contingency", "Goal": "Contingency",
    "Contrast": "Comparison", "Correction": "Comparison",
    "Elaboration": "Expansion", "Continuation": "Expansion", "Parallel": "Expansion", "Alternation": "Expansion",
    "Background": "Expansion", "Comment": "Expansion",
    "Question_answer_pair": "Interaction", "Acknowledgement": "Interaction", "Clarification_question": "Interaction",
    "Q_Elab": "Interaction",
}

PROMPT = """Each item below is a unit of communication: an email sentence, a meeting-transcript excerpt or a chat \
excerpt, one turn per line as "[<turn number>] <speaker letter>: <text>". The text may have no punctuation or capitals, fillers \
("um", "like", "you know"), false starts, repetitions, and one idea may run across consecutive turns by the same \
speaker. Turn each into an IDEA OBJECT that describes what is said, independent of how it is worded or punctuated: \
two units that say the same thing must give the same object.

EVENTS. One event per predication the speaker puts forward: asserts, asks, requests, offers, promises, agrees... \
Find them by meaning, never by punctuation. Rules:
- A verb that only frames another clause (think, guess, say, want/need/try/plan/hope to, be going to) is not its own \
event: fold it into the framed event's "mod". "I think we should send it" is one event: send, mod obligated.
- Adverbial clauses (because, if, when, so, but, although, then, and-then sequences) are separate events joined by a relation.
- Fillers, false starts and abandoned fragments are not events; a repaired phrase counts once, in its repaired form.
- An idea continued in the next turn by the same speaker is ONE event, placed in the turn where it starts.
- A pure backchannel or acknowledgement ("mm-hmm", "yeah", "ok", "thanks") is an event with act feedback or thanking \
and pred "-".
Fields:
- "turn": the number of the turn where the event is said.
- "act": the ISO 24617-2 communicative function, one of {acts}.
- "pred": ONE base-form verb, the simplest common English verb for the action (be for states, have for possession; \
synonyms map to one verb: phone/ring/call -> call; talk/speak/chat -> talk; examine/look over/review -> review).
- "arg0": who does it (agent); "arg1": what it is done to, or the theme; "arg2": recipient, beneficiary or end state. \
Short lowercase noun phrases, no articles; speakers as their letters (I -> the speaker's letter; you -> the \
addressee's letter if clear, else "you"); "-" if absent.
- "neg": true if the event is negated, else false.
- "mod": one of {mods}.
- "time": normalized ("past", "now", "future", "today", "tomorrow", "friday", "2001-08-24") or "-".

RELATIONS. Every event after the first attaches to one earlier event (the one it most directly continues, answers or \
qualifies): {{"from": later event id, "to": earlier event id, "label": one of {relations}}}. One set of labels for \
everything, within a turn or across turns.

CENTER: the id of the single event the whole unit is for, its main point (the nucleus).
GIST: one plain sentence saying what the unit says.

Return only a JSON array: [{{"i": <number>, "events": [{{"id": "e1", "turn": 1, "act": ..., "pred": ..., "arg0": ..., \
"arg1": ..., "arg2": ..., "neg": false, "mod": ..., "time": ...}}, ...], "relations": [...], "center": "e1", "gist": "..."}}, ...]

{items}"""


def valid(obj) -> dict | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("events"), list) or not obj["events"]:
        return None
    ids = set()
    for e in obj["events"]:
        if not isinstance(e, dict) or not isinstance(e.get("id"), str):
            return None
        ids.add(e["id"])
        e["act"] = e.get("act") if e.get("act") in ACTS else "other"
        e["mod"] = e.get("mod") if e.get("mod") in MODS else "none"
        e["neg"] = bool(e.get("neg"))
        e["pred"] = str(e.get("pred") or "-").strip().lower()
        for k in ("arg0", "arg1", "arg2", "time"):
            e[k] = str(e.get(k) or "-").strip().lower()
        try:
            e["turn"] = int(e.get("turn", 1)) - 1
        except (TypeError, ValueError):
            e["turn"] = 0
    rels = [r for r in obj.get("relations") or [] if isinstance(r, dict) and r.get("from") in ids and r.get("to") in ids
            and r.get("label") in RELATIONS]
    return {"events": obj["events"], "relations": rels, "center": obj.get("center") if obj.get("center") in ids else None,
            "gist": str(obj.get("gist") or "")}


def extract(units: list[list[list[str]]], *, model: str = "sonnet", batch: int = 6, workers: int = 5) -> list[dict | None]:
    """One idea object per unit (a list of [speaker, text] turns); None where nothing usable came back."""
    prompts, spans = [], []
    for s in range(0, len(units), batch):
        ch = units[s : s + batch]
        prompts.append(PROMPT.format(acts=list(ACTS), mods=MODS, relations=list(RELATIONS), items="\n\n".join(
            f"Item {k}:\n" + "\n".join(f"{sp}: {t}" for sp, t in u) for k, u in enumerate(ch))))
        spans.append((s, len(ch)))

    def safe(p):
        try:
            return llm.ask_json(p, model=model)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        answers = list(ex.map(safe, prompts))
    out: list[dict | None] = [None] * len(units)
    for (start, n), ans in zip(spans, answers):
        for a in ans if isinstance(ans, list) else []:
            if isinstance(a, dict) and str(a.get("i", "")).isdigit() and 0 <= int(a["i"]) < n:
                out[start + int(a["i"])] = valid(a)
    return out


def content(e: dict) -> str:
    return " ".join(x for x in (e["arg0"], e["pred"], e["arg1"], e["arg2"]) if x != "-") or e["act"]


class Aligner:
    """Matches the events of two idea objects: Hungarian assignment on content similarity, penalising events
    that sit in different (mapped) turns. A pair below ``threshold`` cosine is left unmatched."""

    def __init__(self, threshold: float = 0.7):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        self.threshold = threshold
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            for t, v in zip(new, self.model.encode(new, batch_size=64, normalize_embeddings=True)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])

    def align(self, a: dict, b: dict, b_turn_map: list[int]) -> dict:
        from scipy.optimize import linear_sum_assignment

        ea, eb = a["events"], b["events"]
        sim = self.embed([content(e) for e in ea]) @ self.embed([content(e) for e in eb]).T
        turn_b = [b_turn_map[e["turn"]] if 0 <= e["turn"] < len(b_turn_map) else -1 for e in eb]
        cost = 1 - sim + 0.3 * np.array([[e["turn"] != t for t in turn_b] for e in ea])
        rows, cols = linear_sum_assignment(cost)
        pairs = {ea[r]["id"]: (eb[c]["id"], float(sim[r, c])) for r, c in zip(rows, cols) if sim[r, c] >= self.threshold}
        return pairs


def compare(a: dict, b: dict, pairs: dict) -> dict:
    """Agreement of b with a, layer by layer, over the aligned events."""
    ea = {e["id"]: e for e in a["events"]}
    eb = {e["id"]: e for e in b["events"]}
    m = {x: y for x, (y, _) in pairs.items()}
    matched = [(ea[x], eb[y]) for x, y in m.items()]
    out = {
        "n_a": len(ea), "n_b": len(eb), "count_equal": len(ea) == len(eb),
        "event_f1": 2 * len(m) / (len(ea) + len(eb)),
        "multi_event": len(ea) > 1,
    }
    if matched:
        out["act"] = float(np.mean([x["act"] == y["act"] for x, y in matched]))
        out["act_coarse"] = float(np.mean([ACTS[x["act"]] == ACTS[y["act"]] for x, y in matched]))
        out["neg"] = float(np.mean([x["neg"] == y["neg"] for x, y in matched]))
        out["mod"] = float(np.mean([x["mod"] == y["mod"] for x, y in matched]))
        out["pred"] = float(np.mean([x["pred"] == y["pred"] for x, y in matched]))
        out["content_cos"] = float(np.mean([s for _, s in pairs.values()]))
    ra = {frozenset((r["from"], r["to"])): r["label"] for r in a["relations"]}
    rb = {frozenset((r["from"], r["to"])): r["label"] for r in b["relations"]}
    mapped = {frozenset(m.get(x) for x in k): lab for k, lab in ra.items() if all(x in m for x in k)}
    hit = [k for k in mapped if k in rb]
    if ra or rb:
        out["rel_f1"] = 2 * len(hit) / (len(ra) + len(rb)) if (ra or rb) else 1.0
    if hit:
        out["rel_label"] = float(np.mean([mapped[k] == rb[k] for k in hit]))
        out["rel_coarse"] = float(np.mean([RELATIONS[mapped[k]] == RELATIONS[rb[k]] for k in hit]))
    if a["center"] and b["center"]:
        out["center"] = float(m.get(a["center"]) == b["center"])
    return out


def changed_at(a: dict, b: dict, pairs: dict, turn: int, b_turn_map: list[int]) -> bool:
    """Does b's structure differ from a's at original turn ``turn``: an event there unmatched, or matched with a
    different act, polarity, modality or predicate?"""
    eb = {e["id"]: e for e in b["events"]}
    m = {x: y for x, (y, _) in pairs.items()}
    back = {y: x for x, y in m.items()}
    for e in a["events"]:
        if e["turn"] == turn:
            if e["id"] not in m:
                return True
            f = eb[m[e["id"]]]
            if (ACTS[e["act"]], e["neg"], e["mod"], e["pred"]) != (ACTS[f["act"]], f["neg"], f["mod"], f["pred"]):
                return True
    for e in b["events"]:
        t = b_turn_map[e["turn"]] if 0 <= e["turn"] < len(b_turn_map) else -1
        if t == turn and e["id"] not in back:
            return True
    return False


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)


EVENT = re.compile(r"^t(\d+) (\S+) (.+?) \[(.*?) ; (.*?) ; (.*?)\] (not|yes) (\S+) (.+)$")


def parse(text: str) -> dict | None:
    """Read back ``remote-idea.serialize``'s string; None if no event parses."""
    events, rels, center = [], [], None
    for part in (p.strip() for p in text.split(" | ")):
        m = EVENT.match(part)
        if m:
            t, act, pred, a0, a1, a2, neg, mod, tm = m.groups()
            events.append({"id": str(len(events) + 1), "turn": int(t) - 1, "act": act if act in ACTS else "other", "pred": pred,
                           "arg0": a0, "arg1": a1, "arg2": a2, "neg": neg == "not",
                           "mod": mod if mod in MODS or mod in MOD_COARSE.values() else "none", "time": tm})
        elif part.startswith(">> "):
            bits = part[3:].split()
            if len(bits) == 3 and bits[2] in RELATIONS:
                rels.append({"from": bits[0], "to": bits[1], "label": bits[2]})
        elif part.startswith("* "):
            center = part[2:].strip()
    if not events:
        return None
    ids = {e["id"] for e in events}
    return {"events": events, "relations": [r for r in rels if r["from"] in ids and r["to"] in ids],
            "center": center if center in ids else None, "gist": ""}


def reduced(obj: dict | None) -> dict | None:
    """The stage-1 target after the check (PREREGISTRATION, 10/01/2026): relations dropped, modality coarsened."""
    if obj is None:
        return None
    return {**obj, "relations": [], "events": [{**e, "mod": MOD_COARSE.get(e["mod"], e["mod"])} for e in obj["events"]]}
