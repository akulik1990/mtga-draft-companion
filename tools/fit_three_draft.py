"""While drafting: will this top drafter end in three main colours, this pair plus a third? A logistic rule per pick.

    uv run python tools/fit_three_draft.py FRA [--n 4000]     # writes the three_draft section of the set's fitted.json

Top drafters' drafts (60%+ over 50+ games, 17Lands draft dump, salt "three", the train/test samples left out), joined by
draft id to the main colours of the deck they registered (17Lands game data). At every pick from pack 2 on: the pool's
core pair (the pair casting the most quality), the builder's three-colour inputs on it (deckbuilder.three_features:
spells the best third colour adds above the pair's 23rd, fixing for it, the pair's playables, its duals, the pair's
depth) and the pick number. Label: the deck was that pair plus that colour. The engine (THREE_LANES) moves this share
of each likely pair's probability into the three-colour lane.
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2.deckbuilder import THREE_INPUTS, three_features  # noqa: E402
from companion.advisor_v2.lanes import LANES, castable, quality  # noqa: E402
from tools.fit_splash import fit, record  # noqa: E402
from tools.replay.run import sample, top_dump, top_sample_paths  # noqa: E402

NAMES = ["adds", "fixing", "playables", "duals", "depth", "pick"]


def core_pair(pool, mean, std):
    return max(LANES, key=lambda lane: sum(quality(c, mean, std) for c in pool if castable(c, lane)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--n", type=int, default=4000)
    a = p.parse_args(argv)

    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by = {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by.setdefault(n, c)
    rates = [float(c["deck_colors"]["All Decks"].get("gihwr") or 0) for c in data.values() if (c.get("deck_colors") or {}).get("All Decks")]
    rates = [r for r in rates if r > 0]
    mean = sum(rates) / len(rates)
    std = (sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5

    held = set()
    for path in top_sample_paths(a.set, 600).values():
        if path.exists():
            held |= {d["id"] for d in json.load(gzip.open(path, "rt"))}
    with gzip.open(top_dump(a.set), "rt", newline="", encoding="utf-8") as f:
        drafts = [d for d in sample.sample_drafts(f, a.n, "three") if d["id"] not in held]

    want, colours = {d["id"] for d in drafts}, {}
    with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{a.set}.PremierDraft.csv.gz", "rt") as f:
        r = csv.reader(f)
        ix = {k: i for i, k in enumerate(next(r))}
        for row in r:
            d = row[ix["draft_id"]]
            if d in want and d not in colours:
                colours[d] = row[ix["main_colors"]]

    X, y = [], []
    for d in drafts:
        final = colours.get(d["id"])
        if not final:
            continue
        for pack_no, pick_no, _taken, _pack, pool_names in d["picks"]:
            if pack_no < 2:
                continue
            pool = [by[n] for n in pool_names if n in by]
            pair = core_pair(pool, mean, std)
            f3 = three_features(pool, pair)
            if not f3[3]:
                continue  # no third colour adds anything: the engine leaves the pair alone
            named = dict(zip(THREE_INPUTS, f3))
            X.append([named[k] for k in NAMES[:-1]] + [len(pool) + 1])
            y.append(1.0 if set(final) == set(pair + f3[3]) else 0.0)
    print(f"{len(colours)} drafts with a registered deck, {len(X)} picks from pack 2 with a third colour in view")
    record(a.set, "three_draft", NAMES, *fit(X, y), y,
           f"tools/fit_three_draft.py: from pack 2, whether 60%+ drafters end in the pool's core pair plus the best third colour, "
           f"{len(colours)} drafts (salt three, train/test left out)")


if __name__ == "__main__":
    main()
