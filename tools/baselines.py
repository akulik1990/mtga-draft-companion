"""Table 4: what decks look like, per colour combination and group: every count's mean, spread and range.

    uv run python tools/baselines.py FRA      # writes <data>/tables/FRA/4_baselines.csv

From table 3 (tools/deck_counts.py). Colour combinations: each combination with a splash (UB+G), each base pair or
triple with its splashes folded in (UB*), and all decks (*). Groups: all decks, decks by 60%+ players, trophy decks.
One row per combination, group and field: decks, mean, SD, min, p10, p25, median, p75, p90, max. Combinations with
fewer than --min decks in a group are left out of that group.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from tools import deck_counts, tables  # noqa: E402

TEXT = {"deck", "format", "draft", "lane", "splash", "check", "base", "splash_ours", "combo"}
GROUPS = {"all": lambda r: True, "60%+": lambda r: r["skill"] >= 0.6 - 1e-9, "trophy": lambda r: r["trophy"] > 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--min", type=int, default=100)
    a = p.parse_args(argv)
    rows = deck_counts.read(a.set)
    fields = [k for k in rows[0] if k not in TEXT]
    combos = defaultdict(list)
    for r in rows:
        combos[r["combo"]].append(r)
        if r["base"]:
            combos[r["base"] + "*"].append(r)
        combos["*"].append(r)
    out = []
    for combo, rs in sorted(combos.items(), key=lambda kv: (len(kv[0].rstrip("*").split("+")[0]), kv[0])):
        for g, keep in GROUPS.items():
            sub = [r for r in rs if keep(r)]
            if len(sub) < a.min:
                continue
            m = np.array([[r[f] for f in fields] for r in sub], float)
            q = np.percentile(m, [0, 10, 25, 50, 75, 90, 100], axis=0)
            mean, sd = m.mean(0), m.std(0)
            for i, f in enumerate(fields):
                out.append([combo, g, f, len(sub), round(mean[i], 3), round(sd[i], 3)] + [round(x, 2) for x in q[:, i]])
    path = tables.path(a.set, tables.BASELINES)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "group", "field", "decks", "mean", "sd", "min", "p10", "p25", "median", "p75", "p90", "max"])
        w.writerows(out)
    print(f"{len({(r[0], r[1]) for r in out})} combination x group rows x {len(fields)} fields -> {path}")


if __name__ == "__main__":
    main()
