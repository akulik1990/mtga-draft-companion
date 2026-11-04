"""Our deck builder against the decks top players actually registered, from their own pools.

    uv run python tools/builder_check.py FRA [--n 300] [--v2 deckbuilder.SPLASH_COST=1.0 ...]

17Lands' public game data has, per draft build, the deck and the sideboard: together the pool. For a sample of
60%+ players' two-colour builds (one row per build, a fixed hash sample), build our deck from the pool and compare:
main pair, whether it splashes (and the colour), and spell overlap with theirs. A builder that keeps their pair and
splashes about as often as they do is building the decks good players build.
"""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import composition, config, deckbuilder  # noqa: E402
from companion.advisor_v2.lanes import castable  # noqa: E402
from tools.deck_model import DeckModel  # noqa: E402
from tools.mock_draft import apply_overrides  # noqa: E402

ARGS = {}


CHECK = (0, 300)  # draft-id hash buckets (of 1000) this check samples from; tools/fit_splash.py fits on the rest


def sample(set_code, n, min_skill=0.60, buckets=CHECK, colours=(2,)):
    """[(their main pair, their splash colours, {name: deck count}, {name: pool count})], one per build, from drafts
    whose id hashes into buckets (lo, hi) of 1000, with main-colour counts in colours."""
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz"
    out, seen = [], set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[5:]) for i, h in enumerate(header) if h.startswith("deck_")]
        side_cols = {h[10:]: i for i, h in enumerate(header) if h.startswith("sideboard_")}
        for row in reader:
            skill, pair = row[col["user_game_win_rate_bucket"]], row[col["main_colors"]]
            key = (row[col["draft_id"]], row[col["build_index"]])
            if len(pair) not in colours or skill in ("", "None") or float(skill) < min_skill or key in seen:
                continue
            seen.add(key)
            if not buckets[0] <= int(hashlib.md5(key[0].encode()).hexdigest(), 16) % 1000 < buckets[1]:
                continue
            deck, pool = {}, {}
            for i, name in deck_cols:
                d = int(float(row[i] or 0))
                s = int(float(row[side_cols[name]] or 0)) if name in side_cols else 0
                if d:
                    deck[name] = d
                if d + s:
                    pool[name] = d + s
            lane = "".join(sorted(pair, key="WUBRG".index))
            out.append((lane, row[col["splash_colors"]], deck, pool))
            if len(out) >= n:
                break
    return out


def init(set_code, overrides, no_rule=False):
    apply_overrides(overrides)
    config.activate(set_code)
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by_name = {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by_name.setdefault(n, dict(c, set=set_code))
    ranked = sorted({c["name"] for c in data.values() if ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr")},
                    key=lambda n: -by_name[n]["deck_colors"]["All Decks"]["gihwr"])
    ARGS.update(set=set_code, top=set(ranked[:30]), by_name=by_name, model=DeckModel(set_code, deckbuilder.MODEL), targets=config.fitted(set_code, "decks")["pairs"],
                rule=None if no_rule else config.fitted(set_code, "splash"), three=None if no_rule else config.fitted(set_code, "three"))


def one(item):
    lane, splash, deck, pool = item
    by_name = ARGS["by_name"]
    cards = [by_name[n] for n, k in pool.items() if n in by_name for _ in range(k)]
    builds = deckbuilder.build(cards, ARGS["model"], ARGS["targets"], top=1, rule=ARGS["rule"], three=ARGS["three"])
    if not builds:
        return None
    b = builds[0]
    ours = Counter(c["name"].split(" // ")[0] for c in b["spells"])
    theirs = Counter({n: k for n, k in deck.items() if n in by_name and "Land" not in (by_name[n].get("types") or [])})
    overlap = sum((ours & theirs).values()) / max(1, sum(theirs.values()))
    their_splash_cards = sum(k for n, k in theirs.items() if not castable(by_name[n], lane))
    removal = lambda cards: sum("removal" in config.tags(c) for c in cards)
    their_cards = [by_name[n] for n, k in theirs.items() for _ in range(k)]
    top = ARGS["top"]
    their_bombs = [n for n in theirs if n in top]
    kept = sum(1 for n in their_bombs if n in ours)
    return {"bombs": len(their_bombs), "bombs_kept": kept, "bombs_missed": [n for n in their_bombs if n not in ours], "predicted": b["predicted"], "our_removal": removal(b["spells"]), "their_removal": removal(their_cards),
            "our_creatures": sum(composition.is_creature(c) for c in b["spells"]), "their_creatures": sum(composition.is_creature(c) for c in their_cards), "lane": lane, "our_lane": b["lane"], "splash": splash, "our_splash": b["splash"], "overlap": overlap,
            "their_splash_cards": their_splash_cards, "our_splash_cards": sum(1 for c in b["spells"] if not castable(c, b["lane"])),
            "our_dev": deckbuilder.structure_dev(b["spells"], ARGS["targets"][b["lane"]], b["lane"], ARGS["set"]),
            "their_dev": deckbuilder.structure_dev(their_cards, ARGS["targets"][lane], lane, ARGS["set"]) if lane in ARGS["targets"] else None,
            "our_shape": deckbuilder.support(ARGS["set"], b["lane"], b["spells"]),
            "their_shape": deckbuilder.support(ARGS["set"], lane, their_cards)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--v2", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--no-rule", action="store_true", help="ignore the fitted splash rule (the deck model's own splash term decides)")
    p.add_argument("--tune", action="store_true", help="sample from the draft buckets outside the check (for tuning; confirm on the check)")
    a = p.parse_args(argv)
    init(a.set, a.v2, a.no_rule)  # fail here, not in workers: Pool respawns a worker whose initializer raises, forever
    items = sample(a.set, a.n, colours=(2, 3), buckets=(CHECK[1], 1000) if a.tune else CHECK)
    with Pool(a.jobs, initializer=init, initargs=(a.set, a.v2, a.no_rule)) as pool:
        rows = [r for r in pool.map(one, items, chunksize=2) if r]
    n = len(rows)
    same = sum(r["lane"] == r["our_lane"] for r in rows)
    they = sum(bool(r["splash"]) for r in rows)
    we = sum(bool(r["our_splash"]) for r in rows)
    both = sum(bool(r["splash"]) and bool(r["our_splash"]) for r in rows)
    colour = sum(bool(r["splash"] and r["our_splash"] and r["our_splash"] in r["splash"]) for r in rows)
    real = [r for r in rows if r["their_splash_cards"]]
    print(f"{' '.join(a.v2 + (['no splash rule'] if a.no_rule else [])) or 'as configured'}: {n} top-player pools")
    print(f"  same main pair {100 * same / n:.1f}%   spell overlap {100 * sum(r['overlap'] for r in rows) / n:.1f}%   predicted WR {sum(r['predicted'] for r in rows) / n:.2f}")
    print(f"  splash: they {100 * they / n:.1f}% (with an off-colour spell {100 * len(real) / n:.1f}%), we {100 * we / n:.1f}%; "
          f"we splash when they do {100 * both / max(1, they):.1f}%, same colour {100 * colour / max(1, they):.1f}%; "
          f"we splash when they don't {100 * (we - both) / max(1, n - they):.1f}%")
    three_they = [r for r in rows if len(r["lane"]) == 3]
    three_we = sum(len(r["our_lane"]) == 3 for r in rows)
    print(f"  three colours: they {100 * len(three_they) / n:.1f}%, we {100 * three_we / n:.1f}%; "
          f"we go three when they do {100 * sum(len(r['our_lane']) == 3 for r in three_they) / max(1, len(three_they)):.1f}% "
          f"(same three {100 * sum(r['our_lane'] == r['lane'] for r in three_they) / max(1, len(three_they)):.1f}%), "
          f"when they don't {100 * sum(len(r['our_lane']) == 3 for r in rows if len(r['lane']) == 2) / max(1, n - len(three_they)):.1f}%")
    print(f"  removal: we {sum(r['our_removal'] for r in rows) / n:.1f}, they {sum(r['their_removal'] for r in rows) / n:.1f}; "
          f"below 6: we {100 * sum(r['our_removal'] < 6 for r in rows) / n:.1f}%, they {100 * sum(r['their_removal'] < 6 for r in rows) / n:.1f}%")
    dev = lambda k: [r[k] for r in rows if r[k] is not None]
    shp = lambda k: [r[k][1] for r in rows if r[k][1] is not None]
    print(f"  structure (cards outside the range of 60%+ players' decks in the pair): we {sum(dev('our_dev')) / max(1, len(dev('our_dev'))):.1f}, "
          f"they {sum(dev('their_dev')) / max(1, len(dev('their_dev'))):.1f}; whole-shape points (real decks shaped alike, wins - predicted): "
          f"we {sum(shp('our_shape')) / max(1, len(shp('our_shape'))):+.2f} ({len(shp('our_shape'))} decks), they {sum(shp('their_shape')) / max(1, len(shp('their_shape'))):+.2f}")
    print(f"  creatures: we {sum(r['our_creatures'] for r in rows) / n:.2f}, they {sum(r['their_creatures'] for r in rows) / n:.2f}; "
          f"same main pair: we {sum(r['our_creatures'] for r in rows if r['lane'] == r['our_lane']) / max(1, same):.2f}, "
          f"they {sum(r['their_creatures'] for r in rows if r['lane'] == r['our_lane']) / max(1, same):.2f}")
    bombs = sum(r["bombs"] for r in rows)
    same_rows = [r for r in rows if r["lane"] == r["our_lane"]]
    sb = sum(r["bombs"] for r in same_rows)
    print(f"  top-30 cards in their deck: {bombs}; in ours too {100 * sum(r['bombs_kept'] for r in rows) / max(1, bombs):.1f}% "
          f"(same main colours: {100 * sum(r['bombs_kept'] for r in same_rows) / max(1, sb):.1f}% of {sb}; missed there: "
          f"{', '.join(sorted({m for r in same_rows for m in r['bombs_missed']}))})")
    print(f"  splash cards per splash deck: they {sum(r['their_splash_cards'] for r in real) / max(1, len(real)):.1f}, "
          f"we {sum(r['our_splash_cards'] for r in rows if r['our_splash']) / max(1, we):.1f}")


if __name__ == "__main__":
    main()
