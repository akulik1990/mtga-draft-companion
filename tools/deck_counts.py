"""Table 3: every deck's counts, calculated from table 1 (decks and their cards) and table 2 (each card's features).

    uv run python tools/deck_counts.py FRA      # writes <data>/tables/FRA/3_counts.csv.gz

Table 1 is tools/deck_table.py (1_decks.json.gz), table 2 tools/card_table.py (2_cards.csv), all in <data>/tables/SET/.
One row per deck: its row number in table 1 and draft id, 17Lands' colours and splash, our base colours, splash and
combination (colours_of: 5+ cards of a colour = base, 1-4 cards the base cannot cast = splash), player skill, games, wins, predicted wins, check bucket, then
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
from tools import tables  # noqa: E402
from tools.deck_table import cards, load  # noqa: E402

CURVE = ("1", "2", "3", "4", "5", "6+")
META = ("deck", "format", "draft", "lane", "splash", "skill", "games", "wins", "match_wins", "trophy", "pred", "check")  # deck: row number in table 1
SKIP = {"cmc", "slot", "rarity", "land", "basic", "power", "toughness", "rank", "gihwr", "colours", "pips"}
# The user's rule: a colour with 1-4 cards (basic lands aside) is a splash, 5 or more a base colour. A card counts toward a
# colour only when it needs it: a base colour must have 5 cards the other base colours cannot cast (five W/U hybrids do
# not make a UB deck white); a splash card is one the base colours cannot cast, counted toward the colours it needs.
BASE_MIN = 5


def colours_of(spells):
    """(base, splash) colour strings for a deck's spells (dataset card records, one per copy)."""
    from companion.advisor_v2.lanes import castable, off_pips

    base = "".join(x for x in "WUBRG" if sum(x in (c.get("colors") or []) for c in spells) >= BASE_MIN)
    base = "".join(x for x in base if sum(not castable(c, base.replace(x, "")) and x in (c.get("colors") or []) for c in spells) >= BASE_MIN) or base
    need = {x: 0 for x in "WUBRG"}
    for c in spells:
        if base and not castable(c, base):
            for opts in off_pips(c, base):
                for x in opts:
                    need[x] += 1
    return base, "".join(x for x in "WUBRG" if x not in base and need[x])


def path(set_code):
    return tables.path(set_code, tables.COUNTS)


def read(set_code):
    """[row dict] of table 3, numbers as floats."""
    with gzip.open(path(set_code), "rt", newline="") as f:
        out = []
        for r in csv.DictReader(f):
            out.append({k: (v if k in ("format", "draft", "lane", "splash", "check", "base", "splash_ours", "combo") else float(v or 0)) for k, v in r.items()})
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    a = p.parse_args(argv)
    table, decks = cards(a.set), load(a.set)
    import json

    data = {}
    for c in json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
        data.setdefault(c["name"], c)
    names = list(table)  # front-face aliases included; a deck's card name matches one key
    col = {n: i for i, n in enumerate(names)}
    feats = [k for k, v in next(iter(table.values())).items() if isinstance(v, float) and k not in SKIP]
    slot = lambda v: v if isinstance(v, str) else f"{int(v)}"  # the CSV loader reads "1".."5" as numbers
    num = lambda v: v if isinstance(v, float) else 0.0
    spell = np.array([0.0 if table[n].get("land") else 1.0 for n in names])
    F = np.column_stack([np.array([num(table[n].get(f)) for n in names]) for f in feats]
                        + [np.array([1.0 if slot(table[n].get("slot")) == s else 0.0 for n in names]) for s in CURVE]
                        + [np.array([num(table[n].get(k)) for n in names]) for k in ("cmc", "power", "toughness")])
    # land_* columns count over a deck's lands, everything else over its spells
    F *= np.where(np.array([f.startswith("land_") for f in feats] + [False] * (len(CURVE) + 3)), 1 - spell[:, None], spell[:, None])
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
        w.writerow(list(META) + ["base", "splash_ours", "combo", "n_base", "spells", "lands", "avg_mv", "power", "toughness"] + list(CURVE) + feats)
        for i, d in enumerate(decks):
            s = S[i]
            base, spl = colours_of([data[n] for n, c in d["counts"].items() if n in data and "Land" not in (data[n].get("types") or []) for _ in range(c)])
            w.writerow([i, d["format"], d["draft"], d["lane"], d["splash"], d["skill"] if d["skill"] is not None else "", d["games"], d["wins"],
                        d["match_wins"], d["trophy"], d["pred"], int(d["check"]), base, spl, base + (f"+{spl}" if spl else ""), len(base),
                        int(n_spells[i]), int(n_lands[i]), round(s[k + 6] / n_spells[i], 2) if n_spells[i] else 0, int(s[k + 7]), int(s[k + 8])]
                       + [int(x) for x in s[k:k + 6]] + [int(x) for x in s[:k]])
    print(f"{len(decks)} decks x {len(feats) + 14} columns -> {path(a.set)}")


if __name__ == "__main__":
    main()
