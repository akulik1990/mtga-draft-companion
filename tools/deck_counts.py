"""Table 3: every deck's counts, calculated from table 1 (decks and their cards) and table 2 (each card's features).

    uv run python tools/deck_counts.py FRA      # writes <data>/Temp/deckcounts_SET.csv.gz

Table 1 is tools/deck_table.py (<data>/Temp/decks_SET.json.gz), table 2 tools/card_table.py (<data>/Temp/cards_SET.csv).
One row per deck: its row number in table 1 and draft id, colours, splash, player skill, games, wins, predicted wins, check bucket, then
spells, lands, average mana value, total power and toughness, spells per curve slot, and every numeric card feature
summed over its spells. Re-run after adding a card-table column: a matrix product, seconds, no game data.
"""

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from tools.deck_table import cards, load  # noqa: E402

CURVE = ("1", "2", "3", "4", "5", "6+")
META = ("deck", "format", "draft", "lane", "splash", "skill", "games", "wins", "match_wins", "trophy", "pred", "check")  # deck: row number in table 1
SKIP = {"cmc", "slot", "rarity", "land", "basic", "power", "toughness", "rank", "gihwr", "colours", "pips"}


def path(set_code):
    return DATA_DIR / "Temp" / f"deckcounts_{set_code}.csv.gz"


def read(set_code):
    """[row dict] of table 3, numbers as floats."""
    with gzip.open(path(set_code), "rt", newline="") as f:
        out = []
        for r in csv.DictReader(f):
            out.append({k: (v if k in ("format", "draft", "lane", "splash", "check") else float(v or 0)) for k, v in r.items()})
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    a = p.parse_args(argv)
    table, decks = cards(a.set), load(a.set)
    names = list(table)  # front-face aliases included; a deck's card name matches one key
    col = {n: i for i, n in enumerate(names)}
    feats = [k for k, v in next(iter(table.values())).items() if isinstance(v, float) and k not in SKIP]
    slot = lambda v: v if isinstance(v, str) else f"{int(v)}"  # the CSV loader reads "1".."5" as numbers
    num = lambda v: v if isinstance(v, float) else 0.0
    spell = np.array([0.0 if table[n].get("land") else 1.0 for n in names])
    F = np.column_stack([np.array([num(table[n].get(f)) for n in names]) for f in feats]
                        + [np.array([1.0 if slot(table[n].get("slot")) == s else 0.0 for n in names]) for s in CURVE]
                        + [np.array([num(table[n].get(k)) for n in names]) for k in ("cmc", "power", "toughness")]) * spell[:, None]
    rows, cols, vals = [], [], []
    for i, d in enumerate(decks):
        for n, k in d["counts"].items():
            if n in col:
                rows.append(i); cols.append(col[n]); vals.append(k)
    C = sparse.csr_matrix((np.array(vals, float), (rows, cols)), shape=(len(decks), len(names)))
    S = C @ F
    n_spells, n_lands = C @ spell, C @ (1 - spell)
    k = len(feats)
    with gzip.open(path(a.set), "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(META) + ["spells", "lands", "avg_mv", "power", "toughness"] + list(CURVE) + feats)
        for i, d in enumerate(decks):
            s = S[i]
            w.writerow([i, d["format"], d["draft"], d["lane"], d["splash"], d["skill"] if d["skill"] is not None else "", d["games"], d["wins"],
                        d["match_wins"], d["trophy"], d["pred"], int(d["check"]),
                        int(n_spells[i]), int(n_lands[i]), round(s[k + 6] / n_spells[i], 2) if n_spells[i] else 0, int(s[k + 7]), int(s[k + 8])]
                       + [int(x) for x in s[k:k + 6]] + [int(x) for x in s[:k]])
    print(f"{len(decks)} decks x {len(feats) + 14} columns -> {path(a.set)}")


if __name__ == "__main__":
    main()
