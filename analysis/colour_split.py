"""Cards of the largest and second-largest colour in FRA decks (nonland spells; a gold or hybrid card counts for each colour).

    python3 analysis/colour_split.py

Groups: top 100 = trophy decks of the highest-skill players; all trophy decks of good players (skill >= 0.60); every deck.
"""
import gzip, json
from collections import Counter
from statistics import mean

ROOT = "/home/morris/Projects/mtga-draft-companion"
decks = json.load(gzip.open(f"{ROOT}/analysis/out/decks_FRA.json.gz", "rt"))
cards = {c["name"]: c for c in json.load(open(f"{ROOT}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}


def split(deck):
    col, spells = Counter(), 0
    for name, n in deck["counts"].items():
        c = cards.get(name) or {}
        if "Land" in (c.get("types") or []):
            continue
        spells += n
        for x in c.get("colors") or []:
            col[x] += n
    top = [v for _, v in col.most_common()] + [0, 0, 0]
    return spells, top[0], top[1], top[2]


def report(label, group):
    rows = [split(d) for d in group]
    first = Counter(r[1] for r in rows)
    print(f"\n{label}: {len(rows)} decks; spells {mean(r[0] for r in rows):.1f}; largest colour {mean(r[1] for r in rows):.1f}, "
          f"second {mean(r[2] for r in rows):.1f}, third {mean(r[3] for r in rows):.1f}")
    print("  largest-colour count: " + ", ".join(f"{k}:{first[k]}" for k in sorted(first)))
    print(f"  largest >= 15: {sum(r[1] >= 15 for r in rows) / len(rows):.0%}; second <= 7: {sum(r[2] <= 7 for r in rows) / len(rows):.0%}")


trophy_good = [d for d in decks if d.get("trophy") and d["skill"] >= 0.60]
report("Top 100 (trophy, highest-skill players)", sorted(trophy_good, key=lambda d: -d["skill"])[:100])
report("All trophy decks, good players (skill >= 0.60)", trophy_good)
report("All decks", decks)
