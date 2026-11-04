"""Top players' decks written out, to read rather than test: every card feature counted per deck.

    uv run python tools/deck_sheet.py FRA [--n 100] [--min-games 5] [--seed 0]

Reads the stored tables only (tools/deck_table.py: every deck once; tools/card_table.py: every card's features), so a
new measure is a new card-table column, picked up here without reading the game data. A random sample of decks by 60%+
players with at least --min-games games, sorted by colours and record. Each deck: colours (+ splash), games, its own
win rate, the deck model's predicted win rate for the same games, then each feature summed over its spells (lands
counted apart). The pair's min/best/max from the structure section heads each colour group. Writes docs/decks-SET.md
(the main columns as a table, then every non-zero feature and the cards per deck) and docs/decks-SET.csv (everything).
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
from tools.deck_table import cards, load, measure  # noqa: E402

MAIN = ["role_creatures", "role_removal", "role_card_advantage", "role_evasion", "role_mana"]
CURVE = ["1", "2", "3", "4", "5", "6+"]
SKIP = {"cmc", "slot", "rarity", "land", "basic", "power", "toughness", "rank", "gihwr", "colours", "pips"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    table = cards(a.set)
    feats = [k for k, v in next(iter(table.values())).items() if isinstance(v, float) and k not in SKIP]
    pool = [d for d in load(a.set) if (d["skill"] or 0) >= 0.6 - 1e-9 and d["games"] >= a.min_games]
    rows = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    out = []
    for d in rows:
        spells = {n: k for n, k in d["counts"].items() if n in table and not table[n].get("land")}
        n_spells = sum(spells.values()) or 1
        x = {"colours": d["lane"] + (f"+{d['splash']}" if d["splash"] else ""), "lane": d["lane"], "games": d["games"],
             "win%": round(100 * d["wins"] / d["games"]), "pred%": round(100 * d["pred"] / d["games"], 1),
             "spells": sum(spells.values()), "lands": sum(k for n, k in d["counts"].items() if n in table and table[n].get("land")),
             "avg_mv": round(measure(d, table, "cmc") / n_spells, 2),
             "power": int(sum(k * (table[n].get("power") or 0) for n, k in spells.items() if isinstance(table[n].get("power"), float))),
             "toughness": int(sum(k * (table[n].get("toughness") or 0) for n, k in spells.items() if isinstance(table[n].get("toughness"), float)))}
        for f in feats:
            x[f] = int(measure(d, table, f))
        slot = lambda v: v if isinstance(v, str) else f"{int(v)}"  # the CSV loader reads "1".."5" as numbers
        for s in CURVE:
            x[s] = sum(k for n, k in spells.items() if slot(table[n].get("slot")) == s)
        x["cards"] = ", ".join(f"{k} {n}" if k > 1 else n for n, k in sorted(spells.items(), key=lambda kv: (table[kv[0]].get("cmc") or 0, kv[0])))
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
    cols = ["colours", "games", "win%", "pred%", "spells", "lands", "avg_mv", "power", "toughness"] + CURVE + feats + ["cards"]
    with open(ROOT / "docs" / f"decks-{a.set}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} decks (of {len(pool)}) x {len(cols)} columns -> docs/decks-{a.set}.md, .csv")


if __name__ == "__main__":
    main()
