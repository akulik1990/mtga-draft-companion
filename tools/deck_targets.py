"""What top players' decks look like per colour pair (and per three-colour combination with enough decks): the
overlay's "your deck vs top players" targets and the deck builder's shape bounds.

    uv run python tools/deck_targets.py FRA      # writes the decks section of the set's fitted.json

From 17Lands' public game data (one row per game, the deck played): two-colour main decks of players in
the 60%+ game win-rate buckets, each draft's deck counted once; three-colour main decks too, kept when a combination
has at least --min-triple decks. Mean count per deck of creatures,
removal, card advantage, evasion, two-drops, five-plus drops, lands, and the spell curve (1..6+);
card roles from the dataset's tags.
"""

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.composition import ROLES, curve_slot, roles  # noqa: E402
from companion.advisor_v2.needs import DIMS, dims, lift  # noqa: E402

CURVE = ("1", "2", "3", "4", "5", "6+")
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--min-skill", type=float, default=0.60)
    p.add_argument("--min-triple", type=int, default=150)
    a = p.parse_args(argv)

    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    wrs = [((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0 for c in data.values()]
    wrs = [w for w in wrs if w > 0]
    mean = sum(wrs) / len(wrs)
    sd = (sum((w - mean) ** 2 for w in wrs) / len(wrs)) ** 0.5
    by_name, slot, dim = {}, {}, {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by_name.setdefault(n, roles(c))
            slot.setdefault(n, curve_slot(c))
            dim.setdefault(n, dims(c, mean, sd))
    sums, decks, seen = defaultdict(lambda: dict.fromkeys(ROLES + CURVE, 0.0)), defaultdict(int), set()
    dim_sums = defaultdict(lambda: dict.fromkeys(DIMS, 0.0))  # apart from sums: some names (removal, two_drops) are in both
    single, together = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{a.set}.PremierDraft.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[len("deck_"):]) for i, h in enumerate(header) if h.startswith("deck_")]
        for row in reader:
            skill, pair = row[col["user_game_win_rate_bucket"]], row[col["main_colors"]]
            if len(pair) not in (2, 3) or skill in ("", "None") or float(skill) < a.min_skill:
                continue
            key = (row[col["draft_id"]], row[col["build_index"]])
            if key in seen:
                continue  # one deck per draft build, not one per game
            seen.add(key)
            lane = "".join(sorted(pair, key="WUBRG".index))
            names = []
            for i, name in deck_cols:
                if row[i] and row[i] != "0":
                    n = int(float(row[i]))
                    for r in by_name.get(name, ()):
                        sums[lane][r] += n
                    for d in dim.get(name, ()):
                        dim_sums[lane][d] += n
                    if slot.get(name):
                        sums[lane][slot[name]] += n
                    if slot.get(name):  # spells only: lands go together for fixing, measured on its own
                        names.append(name)
            decks[lane] += 1
            if len(lane) == 2:
                # Deck-mates: which cards top players' decks in this pair hold together, beyond how often each is played.
                names.sort()
                for j, x in enumerate(names):
                    single[lane][x] += 1
                    for y in names[j + 1:]:
                        together[lane][(x, y)] += 1
    out = {lane: {"decks": decks[lane], **{r: round(sums[lane][r] / decks[lane], 1) for r in ROLES},
                  "curve": {k: round(sums[lane][k] / decks[lane], 1) for k in CURVE},
                  "dims": {d: round(dim_sums[lane][d] / decks[lane], 2) for d in DIMS}} for lane in sorted(decks)
           if len(lane) == 2 or decks[lane] >= a.min_triple}
    mates = {}
    for lane, pairs in together.items():
        n, one = decks[lane], single[lane]
        mates[lane] = [[x, y, round(v, 3)] for (x, y), k in pairs.items()
                       if k >= 3 and abs(v := lift(k, one[x], one[y], n)) >= 0.05]
    mate_path = DATA_DIR / "Temp" / f"deckmates_{a.set}.json"
    mate_path.write_text(json.dumps(mates))
    print(f"wrote {sum(len(v) for v in mates.values())} deck-mate lifts to {mate_path}")
    for lane, t in out.items():
        print(lane, t)
    path = config.write_fitted(a.set, "decks", {"min_skill": a.min_skill, "pairs": out},
                               f"tools/deck_targets.py: mean deck of {a.min_skill:.0%}+ players per two-colour pair and three-colour combination (>= {a.min_triple} decks), from 17Lands game data")
    print(f"wrote decks to {path}")


if __name__ == "__main__":
    main()
