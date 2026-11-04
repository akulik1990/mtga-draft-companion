"""Tables 5 and 6: what changes in decks that are low (or high) on one count, or that won, and what survives noise.

    uv run python tools/analyse.py FRA role_creatures low            # one criterion -> 5_compare/role_creatures_low_60%+.csv
    uv run python tools/analyse.py FRA trophy --group all            # trophy decks against the rest
    uv run python tools/analyse.py FRA --all                         # every main count low and high, and trophy; then 6_findings.csv

Reads tables 1-4 (<data>/tables/SET/). Decks with two or three base colours; each field taken relative to the median of
its base colours (UB*: UB with any splash) in the same group from table 4, so combinations that differ by nature do not
show up. A criterion selects decks --by SDs (of the field in those base colours) under (low) or over (high) that median, or trophy decks; every field's
mean is compared with the other decks of the group: difference in cards, z, and the same difference in each half of the
decks (split by draft id), and how much of the field sits on the criterion's own cards (mechanical: fewer creatures
means fewer flyers). Each table-5 file starts with how the selected decks did beyond the deck model.

Noise checks for table 6: a difference of at least --min-diff cards, |z| >= --z, the same sign in both halves with |z| >= --half-z in each, and mechanical share
below --mech. Table 6 gathers what passes across every criterion in table 5.
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from tools import deck_counts, tables  # noqa: E402
from tools.deck_table import cards, load  # noqa: E402

TEXT = {"deck", "format", "draft", "lane", "splash", "check", "base", "splash_ours", "combo"}
META = {"skill", "games", "wins", "match_wins", "trophy", "pred", "n_base"}
MAIN = ["role_creatures", "role_removal", "role_card_advantage", "role_evasion", "role_mana", "1", "2", "3", "4", "5", "6+",
        "avg_mv", "lands", "land_multi", "tokens", "tag_combat_trick", "pump", "rank_1_10", "rank_11_20", "double_pip"]
GROUPS = {"all": lambda r: True, "60%+": lambda r: r["skill"] >= 0.6 - 1e-9, "trophy": lambda r: r["trophy"] > 0}


@lru_cache(maxsize=4)
def medians(set_code, group):
    """({base combination*: {field: median}}, {base combination*: {field: SD}}) from table 4."""
    out, sd = defaultdict(dict), defaultdict(dict)
    with open(tables.path(set_code, tables.BASELINES), newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] == group and r["combo"].endswith("*"):
                out[r["combo"]][r["field"]] = float(r["median"])
                sd[r["combo"]][r["field"]] = float(r["sd"])
    return out, sd


@lru_cache(maxsize=2)
def _played(set_code):
    played = Counter()
    for d in load(set_code):
        for n, k in d["counts"].items():
            played[n] += k
    return cards(set_code), played


def mechanical(set_code, criterion, fields):
    """{field: share of its play-weighted card value that sits on cards counting toward the criterion}."""
    table, played = _played(set_code)
    crit = lambda c: isinstance(c.get(criterion), float) and c[criterion] > 0
    out = {}
    for f in fields:
        tot = on = 0.0
        for n, k in played.items():
            c = table.get(n)
            v = c.get(f) if c else None
            if isinstance(v, float) and v > 0:
                tot += k * v
                on += k * v * crit(c)
        out[f] = on / tot if tot else 0.0
    return out


def compare(rows, fields, sel, rest):
    X = np.array([[r[f] for f in fields] for r in rows], float)
    half = np.array([int(hashlib.md5(r["draft"].encode()).hexdigest(), 16) % 2 for r in rows])

    def diff(a, b):
        d = X[a].mean(0) - X[b].mean(0)
        z = d / np.sqrt(X[a].var(0) / max(1, a.sum()) + X[b].var(0) / max(1, b.sum()) + 1e-12)
        return d, z

    d, z = diff(sel, rest)
    d0, z0 = diff(sel & (half == 0), rest & (half == 0))
    d1, z1 = diff(sel & (half == 1), rest & (half == 1))
    return d, z, d0, z0, d1, z1


def run(a, criterion, side, group, rows_all, fields, mech_cache):
    keep = GROUPS[group]
    med, sds = medians(a.set, group)
    rows = [r for r in rows_all if keep(r) and 2 <= r["n_base"] <= 3 and r["base"] + "*" in med]
    rel = [{**r, **{f: r[f] - med[r["base"] + "*"].get(f, 0.0) for f in fields}} for r in rows]
    if criterion == "trophy":
        sel = np.array([r["trophy"] > 0 for r in rows])
        name = f"trophy_{group}"
    else:
        # --by SDs of the field in the deck's base colours (table 4) under or over their median
        x = np.array([r[criterion] / max(sds[r["base"] + "*"].get(criterion, 1.0), 1e-6) for r in rel])
        sel = x <= -a.by if side == "low" else x >= a.by
        name = f"{criterion}_{side}_{group}"
    rest = ~sel
    if sel.sum() < 30 or rest.sum() < 30:
        print(f"{name}: only {sel.sum()} decks selected, skipped")
        return name, []
    d, z, d0, z0, d1, z1 = compare(rel, fields, sel, rest)
    if criterion != "trophy" and criterion not in mech_cache:
        mech_cache[criterion] = mechanical(a.set, criterion, fields)
    mech = mech_cache.get(criterion, {})
    g = np.array([r["games"] for r in rows]); w = np.array([r["wins"] for r in rows]); p = np.array([r["pred"] for r in rows])
    beyond = lambda m: 100 * (w[m].sum() - p[m].sum()) / g[m].sum()
    out = []
    for i, f in enumerate(fields):
        if f == criterion:
            continue
        clean = (abs(d[i]) >= a.min_diff and abs(z[i]) >= a.z and np.sign(d0[i]) == np.sign(d1[i]) == np.sign(d[i]) and min(abs(z0[i]), abs(z1[i])) >= a.half_z
                 and mech.get(f, 0.0) < a.mech)
        out.append({"criterion": name, "field": f, "these": round(float(np.mean([r[f] for r, s in zip(rel, sel) if s])), 3),
                    "rest": round(float(np.mean([r[f] for r, s in zip(rel, sel) if not s])), 3), "diff": round(float(d[i]), 3),
                    "z": round(float(z[i]), 1), "diff_half_a": round(float(d0[i]), 3), "z_half_a": round(float(z0[i]), 1),
                    "diff_half_b": round(float(d1[i]), 3), "z_half_b": round(float(z1[i]), 1), "mechanical": round(mech.get(f, 0.0), 2),
                    "clean": int(clean), "decks": int(sel.sum()), "other_decks": int(rest.sum()),
                    "beyond_model_these": round(beyond(sel), 2), "beyond_model_rest": round(beyond(rest), 2)})
    out.sort(key=lambda r: -abs(r["z"]))
    folder = tables.path(a.set, tables.COMPARE)
    folder.mkdir(exist_ok=True)
    with open(folder / f"{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"{name}: {sel.sum()} decks against {rest.sum()}; beyond the deck model {beyond(sel):+.2f} vs {beyond(rest):+.2f}; "
          f"{sum(r['clean'] for r in out)} clean of {len(out)} fields")
    return name, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("criterion", nargs="?", help="a table-3 field, or trophy")
    p.add_argument("side", nargs="?", default="low", choices=["low", "high"])
    p.add_argument("--group", default="60%+", choices=list(GROUPS))
    p.add_argument("--by", type=float, default=1.0, help="SDs under (low) or over (high) the base colours' median (table 4)")
    p.add_argument("--min-diff", type=float, default=0.25, help="smallest difference, in cards, worth keeping")
    p.add_argument("--all", action="store_true", help="every MAIN count low and high, and trophy, in --group; then table 6")
    p.add_argument("--z", type=float, default=3.0)
    p.add_argument("--half-z", type=float, default=2.0)
    p.add_argument("--mech", type=float, default=0.5)
    p.add_argument("--show", type=int, default=15)
    a = p.parse_args(argv)
    rows = deck_counts.read(a.set)
    fields = [k for k in rows[0] if k not in TEXT and k not in META]
    mech_cache = {}
    if a.all:
        jobs = [(c, s) for c in MAIN for s in ("low", "high")] + [("trophy", "low")]
        found = []
        for c, s in jobs:
            _, out = run(a, c, s, a.group if c != "trophy" else "all", rows, fields, mech_cache)
            found += [r for r in out if r["clean"]]
        path = tables.path(a.set, tables.FINDINGS)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(found[0]))
            wr.writeheader()
            wr.writerows(found)
        print(f"{len(found)} clean findings -> {path}")
        return
    name, out = run(a, a.criterion, a.side, a.group, rows, fields, mech_cache)
    print(f"\n{'field':24} {'diff':>7} {'z':>6} {'half a':>7} {'half b':>7} {'mech':>5}  clean")
    for r in out[:a.show]:
        print(f"{r['field']:24} {r['diff']:+7.2f} {r['z']:+6.1f} {r['diff_half_a']:+7.2f} {r['diff_half_b']:+7.2f} {r['mechanical']:5.2f}  {'yes' if r['clean'] else ''}")


if __name__ == "__main__":
    main()
