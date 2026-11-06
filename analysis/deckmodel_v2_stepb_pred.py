"""Step B inputs (project env): for every deck of analysis/out/decks_FRA.json.gz (100+ game players), out-of-fold scores.
  fold   = tools/deck_model.fold_of(draft) (md5 mod 5, five folds of drafts)
  lp_*   = win rate (%) of the deck model cross-fitted without the deck's fold (data/Temp/dm1106/cv/*.json, tools/deck_model.py cvfit), typical context:
           v1 = the stored model's specification (true cross-fit refit), v2 = deckmodel_v2 specification, v2t3 = deckmodel_top3_v2 specification
  G*     = the builder's OBJECTIVE gihwr for the 40, without shape penalty and extras: GIH_SLOPE x mean GIH WR of the spells (lane[:2] pair table, All Decks when the
           pair has < 50 GIH games for the card; zero-rated cards dropped as in the builder) - SPLASH_TERM when the deck has splash colours. GIH_SLOPE and
           SPLASH_TERM from fitted.json. G = GIH table of the other four folds (no outcome of the deck's own draft); Gall = All Decks column only (sensitivity);
           Gfile = the card ratings file the builder reads (built from games that include the deck's own: leaky, shown as sensitivity).
Writes data/Temp/dm1106/stepb_pred.csv.gz."""

import csv
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
import companion.advisor_v2.engine  # noqa: E402,F401
from tools.deck_model import DeckModel, card_facts, fold_of  # noqa: E402

config.activate("FRA")
SLOPE, SPLASH = 5.5, 3.1  # fitted.json deckbuilder.GIH_SLOPE, deckbuilder.SPLASH_TERM
fit = json.loads(config.set_file("FRA", "fitted").read_text())["values"]
assert fit["deckbuilder.GIH_SLOPE"] == SLOPE and fit["deckbuilder.SPLASH_TERM"] == SPLASH
CV = REPO / "data" / "Temp" / "dm1106" / "cv"
M = {k: [DeckModel("FRA", model=json.loads((CV / f"cv_{f}_fold{j}.json").read_text())) for j in range(5)] for k, f in (("v1", "v1_tw1"), ("v2", "v2s_tw1"), ("v2t3", "v2s_tw3"))}
folds = json.loads((REPO / "data" / "Temp" / "dm1106" / "gih_folds.json").read_text())  # fold -> key -> card -> [wins, games]
tot = {}
for f, d in folds.items():
    for key, cards in d.items():
        t = tot.setdefault(key, {})
        for c, (w, g) in cards.items():
            a = t.setdefault(c, [0.0, 0.0])
            a[0] += w; a[1] += g
train = {}  # fold -> key -> card -> win rate (%), the other four folds


def table(j):
    if j not in train:
        out = {}
        for key, cards in tot.items():
            sub = folds[str(j)].get(key, {})
            out[key] = {c: (100 * (w - sub.get(c, [0, 0])[0]) / (g - sub.get(c, [0, 0])[1]), g - sub.get(c, [0, 0])[1]) for c, (w, g) in cards.items() if g - sub.get(c, [0, 0])[1] > 0}
        train[j] = out
    return train[j]


ratings = json.loads((DATA_DIR / "Sets" / "FRA_PremierDraft_All_Data.json").read_text())["card_ratings"]
rf = {}
for c in ratings.values():
    rf.setdefault(c["name"], c.get("deck_colors") or {})
facts = card_facts("FRA")


def gscore(counts, lane, splash, j, mode):
    pair, tb = lane[:2], table(j)
    xs = []
    for name, n in counts.items():
        f = facts.get(name)
        if not f or f["land"]:
            continue
        if mode == "file":
            d = rf.get(name, {})
            x = (d.get(pair) or {}).get("gihwr") or (d.get("All Decks") or {}).get("gihwr") or 0.0
        else:
            a = tb.get(pair, {}).get(name) if mode == "pair" else None
            x = a[0] if a and a[1] >= 50 else (tb["All Decks"].get(name, (0.0, 0))[0])
        xs += [x] * n if x else []
    return SLOPE * sum(xs) / len(xs) - (SPLASH if splash else 0.0) if xs else float("nan")


rows = []
for d in json.load(gzip.open(REPO / "analysis" / "out" / "decks_FRA.json.gz", "rt")):
    if d["deck_games"] < 1 or len(d["lane"]) not in (2, 3):
        continue
    c, j = d["counts"], fold_of(d["draft"])
    r = dict(draft=d["draft"], fold=j, fmt=d["format"], lane=d["lane"], S=d["splash"], skill=round(d["skill"], 2), wins=d["deck_wins"], games=d["deck_games"],
             G=gscore(c, d["lane"], d["splash"], j, "pair"), Gall=gscore(c, d["lane"], d["splash"], j, "all"), Gfile=gscore(c, d["lane"], d["splash"], j, "file"))
    for k, ms in M.items():
        r["lp_" + k] = float(ms[j].win_rate(c, d["lane"], d["splash"]))
    rows.append(r)
with gzip.open(REPO / "data" / "Temp" / "dm1106" / "stepb_pred.csv.gz", "wt", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "decks; G missing", sum(r["G"] != r["G"] for r in rows))
