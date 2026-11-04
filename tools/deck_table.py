"""Every deck in 17Lands' public game data, stored once: what deck measures are computed from.

    uv run python tools/deck_table.py FRA      # writes <data>/tables/FRA/1_decks.json.gz

One entry per deck (format, draft id + build), for Premier, Traditional and Pick-Two Draft: format, main colours, splash, the player's win-rate bucket, games, wins, the sum over
its games of the builder's deck model's predicted win chance (each game with its own player, play/draw and mulligan
context), matches won and whether that is a trophy (7 in Premier, 4 in Pick-Two, 3 Bo3 matches in Traditional),
whether it is in tools/builder_check.py's check buckets, and its card counts. Table 1 of three: table 2 is
each card's features (tools/card_table.py), table 3 every deck's counts from the two (tools/deck_counts.py).
"""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from tools import tables  # noqa: E402

CHECK = (0, 300)  # tools/builder_check.py's draft-id hash buckets (of 1000)
TROPHY = {"PremierDraft": 7, "PickTwoDraft": 4, "TradDraft": 3}  # match wins that end the event as a trophy


def path(set_code):
    return tables.path(set_code, tables.DECKS)


def load(set_code):
    """[deck dict] from the stored table."""
    with gzip.open(path(set_code), "rt") as f:
        return json.load(f)


def cards(set_code):
    """{name: {feature: value}} from tools/card_table.py's CSV."""
    with open(tables.path(set_code, tables.CARDS), newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            name = r.pop("name")
            out[name] = {k: (float(v) if v not in ("",) and v.replace(".", "", 1).lstrip("-").isdigit() else v) for k, v in r.items()}
            out.setdefault(name.split(" // ")[0], out[name])
        return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--formats", default="PremierDraft,TradDraft,PickTwoDraft",
                   help="17Lands public game data files to read (Quick Draft and Contender are not public; Sealed builds differently)")
    a = p.parse_args(argv)
    import companion.advisor_v2.engine  # noqa: F401
    config.activate(a.set)
    from companion.advisor_v2 import deckbuilder
    from tools.deck_model import DeckModel

    model = DeckModel(a.set, deckbuilder.MODEL)
    decks = {}
    for fmt in a.formats.split(","):
        read_format(a.set, fmt, model, decks)
    out = list(decks.values())
    for d in out:
        d["pred"] = round(d["pred"], 3)
        # matches won: Bo1 formats one game a match; Traditional a match is won with 2 of its games
        d["match_wins"] = sum(1 for g in d.pop("matches").values() if 2 * sum(g) > len(g))
        d["trophy"] = int(d["match_wins"] >= TROPHY.get(d["format"], 7))
    with gzip.open(path(a.set), "wt") as f:
        json.dump(out, f)
    print(f"{len(out)} decks ({', '.join(fmt + ' ' + str(sum(1 for d in out if d['format'] == fmt)) for fmt in a.formats.split(','))}), "
          f"{sum(d['trophy'] for d in out)} trophies -> {path(a.set)}")


def read_format(set_code, fmt, model, decks):
    """Add one format's games to decks (keyed by format, draft id and build)."""
    from tools.deck_model import SHAPE, context_row, shape

    src = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.{fmt}.csv.gz"
    if not src.exists():
        import requests

        r = requests.get(f"https://17lands-public.s3.amazonaws.com/analysis_data/game_data/game_data_public.{set_code}.{fmt}.csv.gz", timeout=120)
        r.raise_for_status()
        src.write_bytes(r.content)
    with gzip.open(src, "rt", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {k: i for i, k in enumerate(h)}
        cols = [(k[len("deck_"):], i) for k, i in ix.items() if k.startswith("deck_")]
        for row in r:
            counts = {n: int(float(row[i])) for n, i in cols if row[i] not in ("", "0")}
            sk, gm = row[ix["user_game_win_rate_bucket"]], row[ix["user_n_games_bucket"]]
            ctx = context_row(float(sk) if sk not in ("", "None") else None, float(gm) if gm not in ("", "None") else None,
                              row[ix["on_play"]] == "True", float(row[ix["num_mulligans"]] or 0), float(row[ix["opp_num_mulligans"]] or 0))
            mc, sp = row[ix["main_colors"]], row[ix["splash_colors"]]
            st = shape(counts, model.facts, mc, sp)
            z = (np.array([st[k] for k in SHAPE]) - model.m["shape_mu"]) / model.m["shape_sd"]
            z2 = np.concatenate([z, [z[SHAPE.index("creatures")] ** 2, z[SHAPE.index("lands")] ** 2, z[SHAPE.index("cmc2")] ** 2]])
            logit = (model.m["intercept"] + float(np.dot(model.m["context"], ctx)) + float(np.dot(model.m["shape"], z2))
                     + sum(model.m["cards"].get(n, 0.0) * k for n, k in counts.items()))
            draft = row[ix["draft_id"]]
            d = decks.setdefault((fmt, draft, row[ix["build_index"]]), {
                "format": fmt, "draft": draft, "lane": mc, "splash": sp, "skill": float(sk) if sk not in ("", "None") else None,
                "check": CHECK[0] <= int(hashlib.md5(draft.encode()).hexdigest(), 16) % 1000 < CHECK[1],
                "games": 0, "wins": 0, "pred": 0.0, "matches": {}, "counts": counts})
            won = row[ix["won"]] == "True"
            d["games"] += 1
            d["wins"] += won
            d["pred"] += 1.0 / (1.0 + np.exp(-logit))
            d["matches"].setdefault(row[ix["match_number"]], []).append(won)


if __name__ == "__main__":
    main()
