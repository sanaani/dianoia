"""Drop #ubuntu-auto rows whose day neighbors a used day, from files built before that rule (10/05/2026).

``irc-auto-windows.py`` now skips the day on each side of every used day; this applies the same rule to already split,
cleaned and chained files, so they need no new split. uid ``chat-<day>-a<n>...``; other rows pass through.

    python scripts/drop-adjacent-days.py new2-clean-post.jsonl chain-new2.jsonl
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from dianoia.paths import work_path

spec = importlib.util.spec_from_file_location("irc_auto", Path(__file__).with_name("irc-auto-windows.py"))
ia = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ia)
AUTO = re.compile(r"^chat-(\d{4}-\d\d-\d\d)-a\d+")


def main() -> None:
    skip = ia.used_days()
    for name in sys.argv[1:]:
        path = work_path("jepa", name)
        keep, dropped = [], 0
        for line in open(path, encoding="utf-8"):
            m = AUTO.match(json.loads(line)["uid"])
            if m and m.group(1) in skip:
                dropped += 1
            else:
                keep.append(line)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        tmp.replace(path)
        print(json.dumps({"file": name, "kept": len(keep), "dropped": dropped}))


if __name__ == "__main__":
    main()
