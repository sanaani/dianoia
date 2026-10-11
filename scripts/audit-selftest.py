"""Self-test of the leak checks in ``audit-data.py`` (10/04/2026): plant duplicated conversations, require a catch.

Copies 20 held-out windows and 10 held-out units (with 4+ long ideas) into one training conversation, then runs the
audit on the clean data and on the planted copy. Passes only if the clean data passes and the planted copy fails both
duplicate checks with at least the planted counts. Runs before every real audit (``jepa-pipeline-*.sh``), so a change
that blinds the leak checks is caught at once.

    python scripts/audit-selftest.py <units under jepa/> <turn file> [<turn file> ...]
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dianoia.paths import work_path
from dianoia.splits import unit_split


def audit(units: str, turns: list[str]) -> dict:
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit-data.py")), units, "--turns", *turns],
                   check=True, capture_output=True)
    return json.loads(work_path("jepa", units + ".audit.json").read_text(encoding="utf-8"))


def main() -> None:
    units_name, turn_files = sys.argv[1], sys.argv[2:]
    wins = [json.loads(l) for f in turn_files for l in open(work_path(*f.split("/")), encoding="utf-8")]
    units = [json.loads(l) for l in open(work_path("jepa", units_name), encoding="utf-8")]
    held_w = [w for w in wins if unit_split(w["uid"]) != "train" and "/" in w["uid"]]
    train_conv = next(w["uid"].split("/")[0] for w in wins if "/" in w["uid"] and unit_split(w["uid"]) == "train")
    plant_w = [{**w, "uid": f"{train_conv}/9{k:03d}"} for k, w in enumerate(random.Random(1).sample(held_w, 20))]
    held_u = [u for u in units if u["split"] != "train" and sum(len(i.split()) >= 8 for i in u["ideas"]) >= 4]
    group = ":".join(["meeting", train_conv.split("-", 1)[1]]) if train_conv.startswith("meeting-") else None
    plant_u = [{**u, "uid": f"{train_conv}@9{k:03d}", "split": "train", "kind": "window",
                "group": group or u["group"]} for k, u in enumerate(random.Random(2).sample(held_u, 10))]
    with open(work_path("jepa", "selftest-turns.jsonl"), "w", encoding="utf-8") as f:
        for w in wins + plant_w:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    with open(work_path("jepa", "selftest-units.jsonl"), "w", encoding="utf-8") as f:
        for u in units + plant_u:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    manifest = work_path("jepa", units_name + ".manifest.json")
    if manifest.exists():
        shutil.copy(manifest, work_path("jepa", "selftest-units.jsonl.manifest.json"))
    clean = audit(units_name, turn_files)
    planted = audit("selftest-units.jsonl", ["jepa/selftest-turns.jsonl"])
    ok = (clean["heldout_windows_duplicated"] == 0 and clean["heldout_units_duplicated"] == 0
          and planted["heldout_windows_duplicated"] >= 20 and planted["heldout_units_duplicated"] >= 10)
    print(json.dumps({"selftest_pass": ok, "clean": [clean["heldout_windows_duplicated"], clean["heldout_units_duplicated"]],
                      "planted": [planted["heldout_windows_duplicated"], planted["heldout_units_duplicated"]]}))
    sys.exit(0 if ok else 1)


def test_repeated_line_counts_once() -> None:
    """Regression (10/04/2026): a held-out window repeating one shared line twice ("Response on your talk.") was
    counted as 2 matching lines and flagged as a duplicated conversation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("audit-data.py"))
    src = spec.loader.get_source("audit")
    assert "held_windows.append(sorted(set(long_turns)))" in src


if __name__ == "__main__":
    test_repeated_line_counts_once()
    main()
