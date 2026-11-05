"""One compact row per game: kept-hand and deck features from 17Lands FRA Premier game data (shared setup for the opening-hand tests).

    uv run --with pandas python analysis/hand_extract.py        # about 1-2 min, writes analysis/out/hand_games_FRA.csv.gz
Features come from data/tables/FRA/2_cards.csv by card name: land, landcycling (spell with landcycling), cheap = non-land cmc <= 2, big = non-land cmc >= 5, removal = role_removal > 0.
hand_size is the sum of opening_hand_* (7 minus mulligans if the column is the kept hand: checked and printed).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/morris/Projects/mtga-draft-companion")
SRC = ROOT / "data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
OUT = ROOT / "analysis/out/hand_games_FRA.csv.gz"
t0 = time.time()
cards = pd.read_csv(ROOT / "data/tables/FRA/2_cards.csv").drop_duplicates("name").set_index("name")
num = lambda c: pd.to_numeric(cards[c], errors="coerce").fillna(0)
land, cmc, rem, lcy, cre = num("land") > 0, num("cmc"), num("role_removal") > 0, num("landcycling") > 0, num("creature") > 0
header = list(pd.read_csv(SRC, nrows=0).columns)
hand = [c for c in header if c.startswith("opening_hand_")]
deck = [c for c in header if c.startswith("deck_")]
META = ["draft_id", "build_index", "match_number", "game_number", "draft_time", "rank", "on_play", "num_mulligans", "opp_num_mulligans", "num_turns", "won",
        "user_n_games_bucket", "user_game_win_rate_bucket", "main_colors", "splash_colors"]

def vec(cols, prefix, mask):
    names = [c[len(prefix):] for c in cols]
    return np.array([float(mask.get(n, False)) if n in mask.index else 0.0 for n in names], dtype=np.float32), sum(n not in cards.index for n in names)

spell = ~land
M = {"lands": land, "lc": lcy & spell, "cheap": spell & (cmc <= 2), "big": spell & (cmc >= 5), "removal": spell & rem, "creatures": spell & cre, "spells": spell}
H, D, miss_h, miss_d = {}, {}, 0, 0
for k, m in M.items():
    H[k], miss_h = vec(hand, "opening_hand_", m)
    D[k], miss_d = vec(deck, "deck_", m)
print(f"card columns not in 2_cards.csv: hand {miss_h}, deck {miss_d}", flush=True)
parts, n = [], 0
for ch in pd.read_csv(SRC, usecols=META + hand + deck, dtype={c: np.float32 for c in hand + deck}, chunksize=40000):
    h, d = ch[hand].fillna(0).to_numpy(np.float32), ch[deck].fillna(0).to_numpy(np.float32)
    o = ch[META].copy()
    o["hand_size"] = h.sum(1)
    for k in M:
        o["hand_" + k] = h @ H[k]
        o["deck_" + k] = d @ D[k]
    parts.append(o); n += len(o)
out = pd.concat(parts, ignore_index=True)
out["won"] = out["won"].astype(int)
out.to_csv(OUT, index=False)
print(f"{n} games, {out[['draft_id', 'build_index']].drop_duplicates().shape[0]} decks -> {OUT} ({time.time() - t0:.0f}s)")
print("hand_size by num_mulligans (mean):", out.groupby("num_mulligans").hand_size.mean().round(2).to_dict())
print("deck_lands share:", out.drop_duplicates(["draft_id", "build_index"]).deck_lands.value_counts(normalize=True).sort_index().round(3).head(8).to_dict())
