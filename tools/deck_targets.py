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

CURVE = ("1", "2", "3", "4", "5", "6+")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--min-skill", type=float, default=0.60)
    p.add_argument("--min-triple", type=int, default=150)
    a = p.parse_args(argv)

    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by_name, slot = {}, {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by_name.setdefault(n, roles(c))
            slot.setdefault(n, curve_slot(c))
    sums, decks, seen = defaultdict(lambda: dict.fromkeys(ROLES + CURVE, 0.0)), defaultdict(int), set()
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
            for i, name in deck_cols:
                if row[i] and row[i] != "0":
                    n = int(float(row[i]))
                    for r in by_name.get(name, ()):
                        sums[lane][r] += n
                    if slot.get(name):
                        sums[lane][slot[name]] += n
            decks[lane] += 1
    out = {lane: {"decks": decks[lane], **{r: round(sums[lane][r] / decks[lane], 1) for r in ROLES},
                  "curve": {k: round(sums[lane][k] / decks[lane], 1) for k in CURVE}} for lane in sorted(decks)
           if len(lane) == 2 or decks[lane] >= a.min_triple}
    for lane, t in out.items():
        print(lane, t)
    path = config.write_fitted(a.set, "decks", {"min_skill": a.min_skill, "pairs": out},
                               f"tools/deck_targets.py: mean deck of {a.min_skill:.0%}+ players per two-colour pair and three-colour combination (>= {a.min_triple} decks), from 17Lands game data")
    print(f"wrote decks to {path}")


if __name__ == "__main__":
    main()
