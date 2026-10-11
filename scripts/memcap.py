"""Run a command under a memory cap on the laptop (10/05/2026).

    python scripts/memcap.py <MB> -- <command> [args...]

Shaun's rule: a job killed for memory restarts at once with lighter settings. This makes both halves mechanical:

- Before starting, wait (up to 30 minutes) until the machine has <MB> plus a 1 GB margin available, after setting
  aside what other capped jobs (in the shared ledger, below) may still grow into, so a job never
  starts into memory that other programs already hold.
- While it runs, the resident memory of the whole process tree is checked every second; past <MB> the tree is killed
  and the exit code is 137 with "MEMCAP" on stderr, before Windows starts paging or the reaper kills something else.

The pipeline (pipelines/) retries a capped step, and each step's settings shrink with the attempt number (smaller
chunks, fewer rows in memory), so the retry is the lighter restart.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

MARGIN_MB = 1024
WAIT_S = 1800


def tree_rss(p: psutil.Process) -> int:
    """Memory the tree has allocated. On Windows that is private bytes, not the working set: polars maps a 3 GB TSV
    into memory, the mapped file pages count in the working set but can always be dropped, and memcap killed a job
    at "3.7 GB" that had allocated far less (10/05/2026)."""
    total = 0
    for q in [p, *p.children(recursive=True)]:
        try:
            mi = q.memory_info()
            total += getattr(mi, "private", mi.rss)
        except psutil.Error:
            pass
    return total


def kill_tree(p: psutil.Process) -> None:
    procs = [p, *p.children(recursive=True)]
    for q in procs:
        try:
            q.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    cap = int(sys.argv[1]) * 2**20
    cmd = sys.argv[3:]
    LEDGER.mkdir(parents=True, exist_ok=True)
    me = LEDGER / f"{os.getpid()}.json"
    t0 = time.time()
    while True:
        with ledger_lock():
            if available_after_reservations() >= cap + MARGIN_MB * 2**20:
                me.write_text(json.dumps({"cap": cap, "cmd": " ".join(cmd)[:300]}))
                break
        if time.time() - t0 > WAIT_S:
            print(f"MEMCAP: {sys.argv[1]} MB never became available", file=sys.stderr)
            return 137
        time.sleep(30)
    try:
        # Resolve the program first: Windows' CreateProcess did not find "venv/Scripts/python.exe" (10/05/2026).
        proc = subprocess.Popen([shutil.which(cmd[0]) or os.path.abspath(cmd[0]), *cmd[1:]])
        p = psutil.Process(proc.pid)
        peak = 0
        while proc.poll() is None:
            rss = tree_rss(p)
            peak = max(peak, rss)
            if rss > cap:
                kill_tree(p)
                print(f"MEMCAP: killed at {rss / 2**20:.0f} MB (cap {sys.argv[1]} MB): {' '.join(cmd)}", file=sys.stderr)
                return 137
            time.sleep(1)
        print(f"memcap: peak {peak / 2**20:.0f} MB of {sys.argv[1]} MB", file=sys.stderr)
        return proc.returncode
    finally:
        # Another memcap reading the ledger holds this file open for a moment, and Windows then refuses the delete: that
        # failed a finished job, and Snakemake deleted its output (10/08/2026). A stale entry is cleared by the ledger.
        try:
            me.unlink(missing_ok=True)
        except OSError:
            pass


# One ledger for every memcap on the machine (10/05/2026): a smoke run beside the real run each checked free memory on
# its own, both started, both grew into their caps, and the OS reaper killed the smoke. Now a job starts only if free
# memory, minus the headroom that already-running capped jobs may still grow into, covers its cap plus the margin.
LEDGER = Path(os.environ.get("MEMCAP_LEDGER", Path.home() / "dianoia-data" / "memcap"))


@contextmanager
def ledger_lock():
    lock = LEDGER / ".lock"
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 60:  # a holder that died
                lock.unlink(missing_ok=True)
            time.sleep(0.1)
    else:
        raise RuntimeError("memcap ledger lock")
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def available_after_reservations() -> int:
    free = psutil.virtual_memory().available
    for f in LEDGER.glob("*.json"):
        pid = int(f.stem)
        if not psutil.pid_exists(pid):
            f.unlink(missing_ok=True)  # a memcap that was killed
            continue
        try:
            used = tree_rss(psutil.Process(pid))
            free -= max(0, json.loads(f.read_text())["cap"] - used)
        except (psutil.Error, ValueError, OSError):
            pass
    return free


if __name__ == "__main__":
    sys.exit(main())
