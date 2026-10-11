"""Headless Claude as a text utility, cached on disk.

Every call runs ``claude -p`` with no tools and a replaced system prompt, so the model sees only
what the prompt gives it. Responses are cached by the hash of (model, system, prompt): a rerun
reads the cache, which makes a transcript reproducible and spends no tokens twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import shutil

from dianoia.paths import work_path

CLAUDE = shutil.which("claude") or "claude"

SYSTEM = (
    "You are a precise text utility inside a research pipeline. Follow the output format exactly. "
    "Reply with only the requested JSON, no commentary and no code fences."
)


def _key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([model, system, prompt]).encode()).hexdigest()


def ask(prompt: str, *, model: str = "haiku", system: str = SYSTEM, retries: int = 3, effort: str | None = None,
        timeout: int = 600) -> str:
    # Effort (10/06/2026: Opus on one large labeling call) is part of the cache key, so efforts never share answers;
    # without it the key is unchanged and older caches still hit.
    tag = model if effort is None else f"{model}@{effort}"
    cache = work_path("llm-cache", tag, _key(tag, system, prompt)[:2], _key(tag, system, prompt) + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["response"]
    last = ""
    for attempt in range(retries):
        proc = subprocess.run(
            [CLAUDE, "-p", "--model", model, "--system-prompt", system, "--tools", ""] + (["--effort", effort] if effort else []),
            input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout, shell=False,
        )
        last = proc.stdout.strip()
        if proc.returncode == 0 and last:
            cache.write_text(json.dumps({"model": model, "prompt": prompt, "response": last}), encoding="utf-8")
            return last
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude -p failed: {proc.stderr[:500]} {last[:200]}")


def ask_json(prompt: str, **kw):
    text = ask(prompt, **kw)
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in response: {text[:300]}")
    return json.loads(match.group(1))


def ask_many(prompts: list[str], *, workers: int = 4, **kw) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: ask_json(p, **kw), prompts))
