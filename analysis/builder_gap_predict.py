"""Deck-model predictions for every deck of analysis/out/decks_FRA.json.gz (project env), cached for builder_gap_actual.py (nix env with statsmodels).
Writes data/Temp/gap1106/decks_pred.csv.gz. Reads no outcome except to copy deck_wins / deck_games through."""

import gzip
import json

import csv
from builder_gap_common import *  # noqa: F401,F403

setup()
models = {k: DeckModel("FRA", k) for k in ("deckmodel", "deckmodel_top3")}
rows = []
for d in json.load(gzip.open(OUT / "decks_FRA.json.gz", "rt")):
    if d["deck_games"] < 1 or len(d["lane"]) not in (2, 3):
        continue
    c = d["counts"]
    rows.append(dict(draft=d["draft"], fmt=d["format"], lane=d["lane"], S=d["splash"], skill=round(d["skill"], 2), wins=d["deck_wins"], games=d["deck_games"],
                     p=float(models["deckmodel"].win_rate(c, d["lane"], d["splash"])), p3=float(models["deckmodel_top3"].win_rate(c, d["lane"], d["splash"])),
                     size=sum(c.values())))
with gzip.open(REPO / "data" / "Temp" / "gap1106" / "decks_pred.csv.gz", "wt", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "decks; mean card total", sum(r["size"] for r in rows) / len(rows))
