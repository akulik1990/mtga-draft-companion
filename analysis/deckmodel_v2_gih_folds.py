"""Per-card GIH win rate (wins, games) by cross-fitting fold (tools/deck_model.py fold_of: md5 of the draft id mod 5), by main-colour set and for all decks,
from game_data_public.FRA.PremierDraft.csv.gz. GIH = in the opening hand or drawn. Step B scores a deck of fold j with the table of the other four folds, so
the card data has not seen the games it predicts (the ratings file the builder reads is built from games that include them).
Writes data/Temp/dm1106/gih_folds.json {fold: {"All Decks"|colour set: {card: [gih wins, gih games]}}}. Run in the nix pandas env."""

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
acc = {}
for ch in pd.read_csv(path, usecols=["draft_id", "main_colors", "won"] + oh + dr, chunksize=20000):
    fold = ch.draft_id.map(lambda d: int(hashlib.md5(d.encode()).hexdigest(), 16) % 5).values
    gih = ((ch[oh].values + ch[dr].values) > 0).astype(np.float32)
    won = (ch.won.astype(str) == "True").values.astype(np.float32)
    mc = ch.main_colors.fillna("").values
    for f in range(5):
        r = np.where(fold == f)[0]
        if not len(r):
            continue
        for key in list(np.unique(mc[r])) + ["All Decks"]:
            idx = r if key == "All Decks" else r[mc[r] == key]
            a = acc.setdefault((f, key), np.zeros((2, len(cards))))
            a[0] += (gih[idx] * won[idx][:, None]).sum(0)
            a[1] += gih[idx].sum(0)
out = {}
for (f, key), v in acc.items():
    out.setdefault(str(f), {})[key] = {c: [float(v[0, i]), float(v[1, i])] for i, c in enumerate(cards) if v[1, i] > 0}
(REPO / "data" / "Temp" / "dm1106" / "gih_folds.json").write_text(json.dumps(out))
print({f: len(k) for f, k in out.items()}, "colour sets per fold")
