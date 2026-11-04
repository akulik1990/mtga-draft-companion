"""Real decks' whole shapes and how they did; top players' structure per pair (min/best/max per category and curve
slot) and which categories they run more of when one is low (replacements): the structure-first deck builder's inputs.

    uv run python tools/deck_shapes.py FRA      # writes <data>/Temp/deckshapes_FRA.json.gz and the structure section of fitted.json

From 17Lands' public game data, every deck (draft id + build) outside tools/builder_check.py's check buckets: its main
colours, splash, player's win-rate bucket, shape (creatures, removal, card advantage, evasion, spells per curve slot 1..6+), games, wins, and the
sum over its games of the builder's deck model's predicted win chance (each game with its own player, play/draw and
mulligan context). Wins minus predicted is what the deck's shape did beyond its cards: a deck few creatures short
that won anyway (it had the removal to make up for it) counts for that whole shape, not for "few creatures".
"""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402

CHECK = (0, 300)  # tools/builder_check.py's check buckets: left out so the check is not judged by its own decks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--range", type=float, nargs=2, default=(10, 90), help="percentiles of 60%%+ players' decks for min and max")
    p.add_argument("--repl", type=float, default=0.5, help="cards above the usual count that make a category a replacement")
    p.add_argument("--repl-z", type=float, default=4.0, help="and its z (low decks vs the rest)")
    a = p.parse_args(argv)
    import companion.advisor_v2.engine  # noqa: F401  (registers the settings activate checks)
    config.activate(a.set)
    from companion.advisor_v2 import deckbuilder
    from tools.deck_model import SHAPE, DeckModel, context_row, shape

    model = DeckModel(a.set, deckbuilder.MODEL)
    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    card = {}
    for c in data.values():
        card.setdefault(c["name"], c)
        card.setdefault(c["name"].split(" // ")[0], c)
    vec = {n: deckbuilder.shape_vector([c]) for n, c in card.items() if "Land" not in (c.get("types") or [])}
    decks = defaultdict(lambda: {"shape": None, "games": 0, "wins": 0, "pred": 0.0})
    with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{a.set}.PremierDraft.csv.gz", "rt", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {k: i for i, k in enumerate(h)}
        cols = [(k[len("deck_"):], i) for k, i in ix.items() if k.startswith("deck_")]
        for row in r:
            draft = row[ix["draft_id"]]
            if CHECK[0] <= int(hashlib.md5(draft.encode()).hexdigest(), 16) % 1000 < CHECK[1]:
                continue
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
            d = decks[(draft, row[ix["build_index"]])]
            if d["shape"] is None:
                v = np.zeros(len(deckbuilder.SHAPE_DIMS))
                for n, k in counts.items():
                    if n in vec:
                        v += k * np.array(vec[n])
                d.update(shape=v.tolist(), lane=mc, splash=sp, skill=float(sk) if sk not in ("", "None") else None)
            d["games"] += 1
            d["wins"] += row[ix["won"]] == "True"
            d["pred"] += 1.0 / (1.0 + np.exp(-logit))
    out = defaultdict(list)
    for d in decks.values():
        if 20 <= sum(d["shape"][len(deckbuilder.SHAPE_DIMS) - 6:]) <= 26:  # spells by curve slot: a 40-card deck
            out[d["lane"]].append([round(x, 1) for x in d["shape"]] + [d["games"], d["wins"], round(d["pred"], 3), d["splash"] != "", d["skill"] or 0.0])
    path = DATA_DIR / "Temp" / f"deckshapes_{a.set}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"dims": deckbuilder.SHAPE_DIMS, "fields": ["games", "wins", "pred", "splash", "skill"], "lanes": out}, f)
    print(f"{sum(len(v) for v in out.values())} decks in {len(out)} colour combinations -> {path}")
    dims, k = deckbuilder.SHAPE_DIMS, len(deckbuilder.SHAPE_DIMS)
    bounds, rel = {}, []
    for lane, rows in out.items():
        x = np.array(rows, float)
        top = x[x[:, k + 4] >= 0.6 - 1e-9, :k]
        if len(top) < 100:
            continue
        med = np.median(top, 0)
        lo, hi = np.percentile(top, a.range[0], axis=0), np.percentile(top, a.range[1], axis=0)
        bounds[lane] = {d: [round(float(lo[i]), 1), round(float(med[i]), 1), round(float(hi[i]), 1)] for i, d in enumerate(dims)}
        if len(lane) == 2:
            rel.append(top - med)
    # Replacements: when a category runs >= 2 under its pair's median in 60%+ players' decks, what they run more of
    # instead (mean excess over decks where it is not low), pooled over the two-colour pairs. Composites (two categories
# raised together) do not show in FRA: top players raise one or the other, both no more often than chance; people's
# composites go in the manual layer.
    R = np.vstack(rel)
    repl, excess = {}, {}
    for i, d in enumerate(dims):
        low = R[:, i] <= -2
        if low.sum() < 100:
            continue
        ex = R[low].mean(0) - R[~low].mean(0)
        z = ex / np.sqrt(R[low].var(0) / low.sum() + R[~low].var(0) / (~low).sum())
        # clear ones only: not every category has a substitute (evasion: none)
        repl[d] = [dims[j] for j in np.argsort(-ex) if j != i and ex[j] >= a.repl and z[j] >= a.repl_z]
        excess[d] = {dims[j]: round(float(ex[j]), 2) for j in range(k) if j != i}
    written = config.write_fitted(a.set, "structure", {"bounds": bounds, "replacements": repl, "excess": excess, "range": list(a.range)},
                                  f"tools/deck_shapes.py: min/best/max (p{a.range[0]:g}/median/p{a.range[1]:g}) of 60%+ players' decks per "
                                  f"colour combination; replacements = categories they run >= {a.repl:g} more of when one is 2+ under")
    print("replacements:", {d: v for d, v in repl.items() if v})
    print(f"bounds for {len(bounds)} colour combinations -> {written}")


if __name__ == "__main__":
    main()
