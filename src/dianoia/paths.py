"""Where data lives. Large artifacts stay outside the repo (and outside OneDrive)."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DIANOIA_DATA", Path.home() / "dianoia-data"))
ENRON = DATA / "enron"
WORK = DATA / "work"


def result_path(*parts: str) -> Path:
    path = REPO / "results" / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def work_path(*parts: str) -> Path:
    path = WORK / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
