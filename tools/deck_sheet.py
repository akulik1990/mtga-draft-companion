"""Top players' decks written out, to read rather than test: every card feature counted per deck.

    uv run python tools/deck_sheet.py FRA [--n 100] [--min-games 5] [--seed 0]

Reads table 3 (tools/deck_counts.py: every deck's counts) and table 1 (tools/deck_table.py) for the card lists, so a
new measure is a new card-table column (tools/card_table.py) and a re-run of deck_counts, without the game data. A random sample of decks by 60%+
players with at least --min-games games, sorted by colours and record. Each deck: colours (+ splash), games, its own
win rate, the deck model's predicted win rate for the same games, then each feature summed over its spells (lands
counted apart). The pair's min/best/max from the structure section heads each colour group. Writes docs/decks-SET.md
(the main columns as a table, then every non-zero feature and the cards per deck), docs/decks-SET.csv (everything)
and docs/decks-SET.html (every column in groups to switch on, sortable, filter by colours or card, page in deck_sheet.html).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion.advisor_v2 import config  # noqa: E402
from tools import deck_counts  # noqa: E402
from tools.deck_table import cards, load  # noqa: E402

MAIN = ["role_creatures", "role_removal", "role_card_advantage", "role_evasion", "role_mana"]
CURVE = ["1", "2", "3", "4", "5", "6+"]


def write_html(a, out, feats, bounds, pool_size):
    """docs/decks-SET.html: the three tables for the sampled decks (1 decks and cards, 2 card features, 3 counts), each
    with column groups to switch on, sorting, a filter per column and a search (page: tools/deck_sheet.html)."""
    import csv as _csv
    import json

    from companion import DATA_DIR
    from tools.card_table import EFFECTS, KEYWORDS

    kw = [k.replace(" ", "_") for k in KEYWORDS]
    labels = {}

    def grouping(cols, spec):
        """[(group name, predicate, on)] -> groups over cols, each column in the first group that takes it."""
        left, groups = list(cols), []
        for name, pred, on in spec:
            take = [c for c in left if pred(c)]
            left = [c for c in left if c not in take]
            if take:
                groups.append({"name": name, "cols": take, "on": on})
        return groups

    feature_spec = [
        ("Roles", lambda c: c.startswith("role_"), True),
        ("Set mechanics", lambda c: c.startswith("mech_"), False),
        ("Keywords", lambda c: c in kw, False),
        ("Effects", lambda c: c in EFFECTS, False),
        ("Rarity & rank", lambda c: c in ("common", "uncommon", "rare", "mythic") or c.startswith("rank_"), False),
        ("Tags", lambda c: c.startswith("tag_"), False),
        ("Types & mana", lambda c: True, False)]
    for c in feats:
        labels[c] = c.removeprefix("role_").removeprefix("mech_").removeprefix("tag_").replace("_", " ")
    labels.update({f"rank_{lo}_{lo + 9}": f"rank {lo}-{lo + 9}" for lo in range(1, 100, 10)}, lane="pair")

    # 1: decks and their cards
    t1 = [dict(_k=str(i), deck=i + 1, colours=x["colours"], lane=x["lane"], skill=x["skill"], games=x["games"], wins=x["wins"],
               **{"win%": x["win%"], "pred%": x["pred%"]}, cards=x["cards"]) for i, x in enumerate(out)]
    # 2: every card's features, with its rules text (Scryfall, both faces; else the dataset's)
    sf_path = DATA_DIR / "Temp" / f"scryfall_cards_{a.set}.json"
    sf = json.loads(sf_path.read_text()) if sf_path.exists() else {}
    with open(DATA_DIR / "Temp" / f"cards_{a.set}.csv", newline="") as f:
        cards_rows = list(_csv.DictReader(f))
    t2 = []
    for r in cards_rows:
        c = sf.get(r["name"]) or sf.get(r["name"].split(" // ")[0]) or {}
        text = " // ".join(x.get("oracle_text") or "" for x in (c.get("card_faces") or [c])) if c else ""
        row = {k: (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v) for k, v in r.items()}
        row.update(_k=r["name"], text=text.replace("\n", " "))
        t2.append(row)
    card_cols = [k for k in cards_rows[0] if k != "name"]
    basics = ["rarity", "cmc", "slot", "colours", "pips", "power", "toughness", "gihwr", "rank"]
    # 3: counts per deck
    t3 = [dict({k: x[k] for k in ["colours", "lane", "cards", "games", "win%", "pred%", "spells", "lands", "avg_mv", "power", "toughness"] + CURVE + feats},
               _k=str(i), deck=i + 1) for i, x in enumerate(out)]
    bound_key = {**{c: c.removeprefix("role_") for c in MAIN}, **{c: c for c in CURVE}}
    tables = [
        {"id": "decks", "name": "1 Decks", "cols": ["deck"], "rows": t1, "search": ["colours", "cards"],
         "note": "Table 1: each deck with its record and spells. Filter a column with text (contains), =text (exact) or a comparison: >3, <=1, =0, 2-4.",
         "groups": [{"name": "Record", "cols": ["colours", "skill", "games", "wins", "win%", "pred%"], "on": True},
                    {"name": "Cards", "cols": ["cards"], "on": True}]},
        {"id": "cards", "name": "2 Cards", "cols": ["name"], "rows": t2, "search": ["name", "text"],
         "note": "Table 2: every card in the set and what it does: stats, roles, types, keywords, the set's mechanics, effects read "
                 "from its rules text, rarity, rank by GIH WR, tags. 1 = has it.",
         "groups": [{"name": "Basics", "cols": basics, "on": True}, {"name": "Text", "cols": ["text"], "on": True}]
                   + grouping([c for c in card_cols if c not in basics], feature_spec)},
        {"id": "counts", "name": "3 Counts", "cols": ["deck"], "rows": t3, "search": ["colours", "cards"], "more": "cards", "bounds": True,
         "boundKey": bound_key,
         "note": "Table 3: each deck's counts over its spells, from tables 1 and 2. Click a deck for its cards. Filter pair to one "
                 "pair (=UB: exact; splashes included) to see its min/best/max row.",
         "groups": [{"name": "Deck", "cols": ["colours", "lane", "games", "win%", "pred%", "spells", "lands", "avg_mv", "power", "toughness"], "on": True},
                    {"name": "Curve", "cols": CURVE, "on": True}] + grouping(feats, feature_spec)},
    ]
    data = {"title": f"{a.set}: {len(out)} decks by 60%+ players, their cards and counts",
            "sub": f"A random sample of {len(out)} of the {pool_size} decks with {a.min_games}+ games (seed {a.seed}). win% is the deck's own "
                   "record, pred% the deck model's predicted win chance for the same games.",
            "labels": labels, "bounds": bounds, "tables": tables}
    page = (ROOT / "tools" / "deck_sheet.html").read_text().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (ROOT / "docs" / f"decks-{a.set}.html").write_text(page)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    table, decks, counts = cards(a.set), load(a.set), deck_counts.read(a.set)
    fixed = set(deck_counts.META) | {"spells", "lands", "avg_mv", "power", "toughness"} | set(CURVE)
    feats = [k for k in counts[0] if k not in fixed]
    pool = [r for r in counts if r["skill"] >= 0.6 - 1e-9 and r["games"] >= a.min_games]
    out = []
    for r in random.Random(a.seed).sample(pool, min(a.n, len(pool))):
        spells = {n: k for n, k in decks[int(r["deck"])]["counts"].items() if n in table and not table[n].get("land")}
        x = {k: (int(v) if isinstance(v, float) and k not in ("avg_mv", "skill") else v) for k, v in r.items()}
        x.update(colours=r["lane"] + (f"+{r['splash']}" if r["splash"] else ""), games=int(r["games"]),
                 **{"win%": round(100 * r["wins"] / r["games"]), "pred%": round(100 * r["pred"] / r["games"], 1)},
                 cards=", ".join(f"{k} {n}" if k > 1 else n for n, k in sorted(spells.items(), key=lambda kv: (table[kv[0]].get("cmc") or 0, kv[0]))))
        out.append(x)
    out.sort(key=lambda x: (x["lane"], -x["win%"]))
    bounds = (config.fitted(a.set, "structure") or {}).get("bounds", {})
    head = ["colours", "games", "win%", "pred%"] + MAIN + CURVE + ["avg_mv", "lands"]
    names = {f: f.removeprefix("role_") for f in head}
    md = [f"# {a.set}: {len(out)} decks by 60%+ players", "",
          f"Random sample of decks with >= {a.min_games} games (seed {a.seed}). win% is the deck's own record; pred% the deck model's "
          "average predicted win chance over the same games (its cards, shape and the player). mana = spells that get lands or mana. "
          "Curve = spells by mana value. Each colour group starts with its min/best/max (p10/median/p90 of all 60%+ decks in the pair). "
          "Under the table, every non-zero feature per deck (rank_1_10 = cards ranked 1-10 in the set by GIH WR; mech_* = the set's "
          "own mechanics) and its spells. docs/decks-" + a.set + ".csv has every column.", "",
          "| " + " | ".join(names[h] for h in head) + " |", "|" + "---|" * len(head)]
    last = None
    for x in out:
        if x["lane"] != last:
            last = x["lane"]
            b = bounds.get(last)
            if b:
                cell = lambda k: "/".join(f"{y:g}" for y in b[k])
                md.append(f"| **{last} min/best/max** | | | | " + " | ".join(cell(k.removeprefix("role_")) for k in MAIN) + " | "
                          + " | ".join(cell(k) for k in CURVE) + " | | |")
        md.append("| " + " | ".join(str(x[h]) for h in head) + " |")
    md += ["", "## Every feature and the cards", ""]
    for x in out:
        nz = ", ".join(f"{f} {x[f]}" for f in feats if x[f] and f not in MAIN)
        md.append(f"- **{x['colours']}** {x['win%']}% of {x['games']} (pred {x['pred%']}%), power {x['power']}, toughness {x['toughness']}: {nz}")
        md.append(f"  - {x['cards']}")
    (ROOT / "docs" / f"decks-{a.set}.md").write_text("\n".join(md) + "\n")
    write_html(a, out, feats, bounds, len(pool))
    cols = ["colours", "games", "win%", "pred%", "spells", "lands", "avg_mv", "power", "toughness"] + CURVE + feats + ["cards"]
    with open(ROOT / "docs" / f"decks-{a.set}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} decks (of {len(pool)}) x {len(cols)} columns -> docs/decks-{a.set}.md, .csv, .html")


if __name__ == "__main__":
    main()
