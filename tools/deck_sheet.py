"""Top players' decks written out, to read rather than test: curve, components, other tags, record, cards.

    uv run python tools/deck_sheet.py FRA [--n 100] [--min-games 5] [--seed 0]

From 17Lands' public game data: a random sample of decks by 60%+ players with at least --min-games games, sorted by
colours and record. Each row: colours (+ splash), games and win rate, the deck model's predicted win rate for an average
player, counts of creatures, removal, card advantage, evasion, mana (gets lands or mana), combat tricks, lifegain,
mana sinks, protection, spells per curve slot, lands; then each deck's spells. The pair's min/best/max from the
structure section heads each colour group. Writes docs/decks-SET.md and docs/decks-SET.csv.
"""

import argparse
import csv
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402

TAGS = ("combat_trick", "lifegain", "mana_sink", "protection")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    import companion.advisor_v2.engine  # noqa: F401
    config.activate(a.set)
    from companion.advisor_v2 import deckbuilder
    from tools.deck_model import DeckModel

    model = DeckModel(a.set, deckbuilder.MODEL)
    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    card = {}
    for c in data.values():
        card.setdefault(c["name"], c)
        card.setdefault(c["name"].split(" // ")[0], c)
    decks = defaultdict(lambda: {"games": 0, "wins": 0})
    with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{a.set}.PremierDraft.csv.gz", "rt", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {k: i for i, k in enumerate(h)}
        cols = [(k[len("deck_"):], i) for k, i in ix.items() if k.startswith("deck_")]
        for row in r:
            sk = row[ix["user_game_win_rate_bucket"]]
            if sk in ("", "None") or float(sk) < 0.6 - 1e-9:
                continue
            d = decks[(row[ix["draft_id"]], row[ix["build_index"]])]
            if not d["games"]:
                d.update(counts={n: int(float(row[i])) for n, i in cols if row[i] not in ("", "0")},
                         lane=row[ix["main_colors"]], splash=row[ix["splash_colors"]])
            d["games"] += 1
            d["wins"] += row[ix["won"]] == "True"
    pool = [d for d in decks.values() if d["games"] >= a.min_games]
    rows = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    head = ["colours", "games", "win%", "pred%", *deckbuilder.ROLE_DIMS, *TAGS, *deckbuilder.CURVE, "lands"]
    out = []
    for d in rows:
        spells = [card[n] for n, k in d["counts"].items() if n in card and "Land" not in (card[n].get("types") or []) for _ in range(k)]
        v = dict(zip(deckbuilder.SHAPE_DIMS, deckbuilder.shape_vector(spells)))
        tags = {t: sum(t in config.tags(c) for c in spells) for t in TAGS}
        lands = sum(k for n, k in d["counts"].items() if n in card and "Land" in (card[n].get("types") or []))
        out.append({"colours": d["lane"] + (f"+{d['splash']}" if d["splash"] else ""), "lane": d["lane"], "games": d["games"],
                    "win%": round(100 * d["wins"] / d["games"]), "pred%": round(float(model.win_rate(d["counts"], d["lane"], d["splash"])), 1),
                    **{k: int(v[k]) for k in deckbuilder.SHAPE_DIMS}, **tags, "lands": lands,
                    "cards": ", ".join(f"{k} {n}" if k > 1 else n for n, k in sorted(d["counts"].items(), key=lambda kv: (card.get(kv[0], {}).get("cmc") or 0, kv[0]))
                                       if n in card and "Land" not in (card[n].get("types") or []))})
    out.sort(key=lambda x: (x["lane"], -x["win%"]))
    bounds = (config.fitted(a.set, "structure") or {}).get("bounds", {})
    md = [f"# {a.set}: {len(out)} decks by 60%+ players", "",
          f"Random sample of decks with >= {a.min_games} games (seed {a.seed}). win% is the deck's own record; pred% the deck model's win rate for an "
          "average player with it (its cards and shape). mana = spells that get lands or mana. Curve = spells by mana value. "
          "Each colour group starts with its min/best/max (p10/median/p90 of all 60%+ decks in the pair).", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    last = None
    for x in out:
        if x["lane"] != last:
            last = x["lane"]
            b = bounds.get(last)
            if b:
                md.append("| **" + last + " min/best/max** | | | | " + " | ".join("/".join(f"{y:g}" for y in b[k]) for k in deckbuilder.ROLE_DIMS)
                          + " | " * (len(TAGS) + 1) + " | ".join("/".join(f"{y:g}" for y in b[k]) for k in deckbuilder.CURVE) + " | |")
        md.append("| " + " | ".join(str(x[k]) for k in head) + " |")
    md += ["", "## Cards", ""] + [f"- **{x['colours']}** {x['win%']}% ({x['games']} games): {x['cards']}" for x in out]
    (ROOT / "docs" / f"decks-{a.set}.md").write_text("\n".join(md) + "\n")
    with open(ROOT / "docs" / f"decks-{a.set}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head + ["cards"], extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} decks (of {len(pool)} with >= {a.min_games} games) -> docs/decks-{a.set}.md, .csv")


if __name__ == "__main__":
    main()
