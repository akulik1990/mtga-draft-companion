"""Deck-model predictions for every deck of analysis/out/decks_FRA.json.gz under the stored model (deckmodel, deckmodel_top3, from builder_gap_pred cache)
and the staged version 2 models (data/Temp/dm1106/stage/*.json, written by `tools/deck_model.py fit2`). Project env.
Writes data/Temp/dm1106/decks_pred_v2.csv.gz. Columns: p1/p3 stored, p1t/p1t3 version 1 refit on train drafts only, p2/p23 version 2 on all games,
p2t/p2t3 version 2 on train drafts only, p2s* the same with skill dummies per 0.02 bucket (st = train only), a trailing k = at the deck's own skill bucket, test = draft in the held-out 20% (hash < 52, Premier CSV only). Reads outcomes only to copy them through."""

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import companion  # noqa: E402,F401
from companion.advisor_v2 import config  # noqa: E402
import companion.advisor_v2.engine  # noqa: E402,F401
import math  # noqa: E402

import numpy as np  # noqa: E402
from tools.deck_model import DeckModel, context_row  # noqa: E402

config.activate("FRA")
ST = REPO / "data" / "Temp" / "dm1106" / "stage2"
load = lambda n: json.loads((ST / n).read_text())
M = {"p1": DeckModel("FRA", "deckmodel"), "p3": DeckModel("FRA", "deckmodel_top3")}
for k, f in (("p1t", "v1_tw1.train.json"), ("p1t3", "v1_tw3.train.json"), ("p2", "v2_tw1.json"), ("p23", "v2_tw3.json"), ("p2t", "v2_tw1.train.json"), ("p2t3", "v2_tw3.train.json"), ("p2s", "v2s_tw1.json"), ("p2s3", "v2s_tw3.json"), ("p2st", "v2s_tw1.train.json"), ("p2st3", "v2s_tw3.train.json")):
    M[k] = DeckModel("FRA", model=load(f))
rows = []
for d in json.load(gzip.open(REPO / "analysis" / "out" / "decks_FRA.json.gz", "rt")):
    if d["deck_games"] < 1 or len(d["lane"]) not in (2, 3):
        continue
    c = d["counts"]
    r = dict(draft=d["draft"], fmt=d["format"], lane=d["lane"], S=d["splash"], skill=round(d["skill"], 2), wins=d["deck_wins"], games=d["deck_games"],
             test=int(int(hashlib.md5(d["draft"].encode()).hexdigest()[:2], 16) < 52 and d["format"] == "PremierDraft"), size=sum(c.values()))
    for k, m in M.items():
        r[k] = float(m.win_rate(c, d["lane"], d["splash"]))
        t = m.m["typical"]  # the same at the deck's own skill bucket (calibration by skill group): column k + "k"
        shift = float(np.dot(m.m["context"], context_row(d["skill"], t["games"], t["on_play"], t["mull"], t["opp_mull"], fine=bool(m.m.get("context_fine"))))) - m.ctx
        lo = math.log(r[k] / (100 - r[k])) + shift
        r[k + "k"] = 100 / (1 + math.exp(-lo))
    rows.append(r)
with gzip.open(REPO / "data" / "Temp" / "dm1106" / "decks_pred_v2.csv.gz", "wt", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "decks;", sum(r["test"] for r in rows), "in held-out Premier drafts")
