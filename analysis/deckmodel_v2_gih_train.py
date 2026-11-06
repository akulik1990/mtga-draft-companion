"""Per-card GIH win rate from the TRAIN drafts only (draft-id hash >= 52, the split tools/deck_model.py uses), by main-colour set and for all decks,
from game_data_public.FRA.PremierDraft.csv.gz. GIH = the card was in the opening hand or drawn. Streams in chunks (memory).
The card ratings file the builder reads (data/Sets/FRA_PremierDraft_All_Data.json) is built from the same games, held-out ones included;
Step B (analysis/deckmodel_v2_stepb.py) scores held-out decks with this train-only table so the card data has not seen the games it predicts.
Writes data/Temp/dm1106/gih_train.json {"All Decks"|colour set: {card: [gih wins, gih games]}}; also the full-sample table for a check against the ratings file."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
path = REPO / "data" / "Temp" / "synergy" / "game_data_public.FRA.PremierDraft.csv.gz"
head = pd.read_csv(path, nrows=0).columns
oh = [c for c in head if c.startswith("opening_hand_")]
dr = [c for c in head if c.startswith("drawn_")]
cards = [c[len("opening_hand_"):] for c in oh]
assert [c[len("drawn_"):] for c in dr] == cards
use = ["draft_id", "main_colors", "won"] + oh + dr
acc = {"train": {}, "all": {}}
for ch in pd.read_csv(path, usecols=use, chunksize=20000):
    test = ch.draft_id.map(lambda d: int(hashlib.md5(d.encode()).hexdigest()[:2], 16) < 52).values
    gih = ((ch[oh].values + ch[dr].values) > 0).astype(np.float32)
    won = (ch.won.astype(str) == "True").values.astype(np.float32)
    for which, rows in (("train", ~test), ("all", np.ones(len(ch), bool))):
        for key, idx in list(ch[rows].groupby(ch.main_colors[rows].fillna("")).indices.items()) + [("All Decks", np.arange(rows.sum()))]:
            g = gih[rows][idx]
            a = acc[which].setdefault(key, np.zeros((2, len(cards))))
            a[0] += (g * won[rows][idx][:, None]).sum(0)
            a[1] += g.sum(0)
out = {w: {k: {c: [float(v[0, i]), float(v[1, i])] for i, c in enumerate(cards) if v[1, i] > 0} for k, v in d.items()} for w, d in acc.items()}
(REPO / "data" / "Temp" / "dm1106" / "gih_train.json").write_text(json.dumps(out))
# check against the ratings file
r = json.loads((REPO / "data" / "Sets" / "FRA_PremierDraft_All_Data.json").read_text())["card_ratings"]
diffs = []
for c in r.values():
    a = out["all"]["All Decks"].get(c["name"])
    g = (c.get("deck_colors") or {}).get("All Decks", {}).get("gihwr")
    if a and g and a[1] >= 500:
        diffs.append((100 * a[0] / a[1] - g, a[1], (c["deck_colors"]["All Decks"].get("game_count") or 0)))
d = np.array(diffs)
print(f"{len(d)} cards; GIH WR (our full sample) minus ratings file: mean {d[:, 0].mean():+.2f}, sd {d[:, 0].std():.2f}, max abs {np.abs(d[:, 0]).max():.2f}; our GIH games mean {d[:, 1].mean():.0f}")
