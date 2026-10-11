"""Count how many distinct messages mention each named entity, and fix the literal list.

Pre-registered rule: an entity in 50 or more distinct messages of the 30,000-message corpus has
become a concept and stays literal; every other name is a variable. Counted with spaCy's
en_core_web_sm over the kept sentences, keyed by the entity's normalised surface text.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd
import spacy

from dianoia.paths import result_path, work_path

THRESHOLD = 50
LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "LAW", "WORK_OF_ART"}


def main() -> None:
    sents = pd.read_parquet(work_path("corpus", "sentences.parquet"))
    sents = sents[~sents["boilerplate"]]
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    messages: dict[str, set] = defaultdict(set)
    label: dict[str, Counter] = defaultdict(Counter)
    for mid, doc in zip(sents["message_id"], nlp.pipe(sents["text"].tolist(), batch_size=512)):
        for ent in doc.ents:
            if ent.label_ in LABELS:
                key = " ".join(ent.text.split()).strip(" .,'\"")
                if key:
                    messages[key].add(mid)
                    label[key][ent.label_] += 1
    counts = {k: len(v) for k, v in messages.items()}
    literal = sorted((k for k, n in counts.items() if n >= THRESHOLD), key=lambda k: -counts[k])
    table = pd.DataFrame(
        [{"entity": k, "messages": counts[k], "label": label[k].most_common(1)[0][0]} for k in counts]
    ).sort_values("messages", ascending=False)
    table.to_parquet(work_path("corpus", "entities.parquet"))
    out = {"threshold": THRESHOLD, "distinct_entities": len(counts), "literal": len(literal),
           "literal_entities": [{"entity": k, "messages": counts[k], "label": label[k].most_common(1)[0][0]} for k in literal]}
    result_path("phase-0-corpus", "literal-entities.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(counts), "entities;", len(literal), "literal")
    print(table.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
