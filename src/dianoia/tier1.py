"""Tier 1's measurement: paired win rates of keep variants over move variants.

For source x, keep variant k and move variant m of the same source, a win is sim(x, k) > sim(x, m)
(a tie is a half). Rates are pooled over every (k, m) pair a source has; the interval resamples
sources, so a source with many variants cannot narrow it on its own.
"""

from __future__ import annotations

import numpy as np

KEEP = ["paraphrase", "rename", "register", "drop_modifiers", "canonical"]
MOVE = ["act_change", "role_swap", "negation", "modality", "predicate_change", "type_change", "quantity"]


def pair_wins(sim: dict[str, np.ndarray], keep: list[str] = KEEP, move: list[str] = MOVE) -> dict:
    """``sim[field]`` is (n_items,) similarity of each item's source to that field, NaN if absent.

    Returns per-item arrays of wins and pair counts, overall and by move kind and keep kind.
    """
    n = len(next(iter(sim.values())))
    out = {"all": (np.zeros(n), np.zeros(n))}
    for m in move:
        out[f"move:{m}"] = (np.zeros(n), np.zeros(n))
    for k in keep:
        out[f"keep:{k}"] = (np.zeros(n), np.zeros(n))
    for k in keep:
        for m in move:
            sk, sm = sim.get(k), sim.get(m)
            if sk is None or sm is None:
                continue
            ok = ~np.isnan(sk) & ~np.isnan(sm)
            w = np.where(ok, (sk > sm) + 0.5 * (sk == sm), 0.0)
            c = ok.astype(float)
            for key in ("all", f"move:{m}", f"keep:{k}"):
                out[key][0][:] += w
                out[key][1][:] += c
    return out


def summarise(wins: np.ndarray, counts: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> dict:
    has = counts > 0
    w, c = wins[has], counts[has]
    if c.sum() == 0:
        return {"n_sources": 0, "n_pairs": 0}
    rate = w.sum() / c.sum()
    idx = rng.integers(0, len(w), size=(draws, len(w)))
    boots = w[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_sources": int(has.sum()), "n_pairs": int(c.sum()), "win_rate": round(float(rate), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)]}


def paired_diff(wa: np.ndarray, ca: np.ndarray, wb: np.ndarray, cb: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> dict:
    """Difference in win rate between two encoders on the same sources, with a paired bootstrap."""
    has = (ca > 0) & (cb > 0)
    wa, ca, wb, cb = wa[has], ca[has], wb[has], cb[has]
    d = wa.sum() / ca.sum() - wb.sum() / cb.sum()
    idx = rng.integers(0, len(wa), size=(draws, len(wa)))
    boots = wa[idx].sum(1) / ca[idx].sum(1) - wb[idx].sum(1) / cb[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": round(float(d), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}
