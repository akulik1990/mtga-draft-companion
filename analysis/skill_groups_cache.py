"""Stream the FRA Premier game CSV into a compact int8 npz (hand = opening + drawn, deck copies, meta incl. skill and games bucket).
Run: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 nix shell ... -c python3 analysis/skill_groups_cache.py"""
import os, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
hdr = pd.read_csv(GD, nrows=0).columns.tolist()
names = [h[len("deck_"):] for h in hdr if h.startswith("deck_")]
meta = ["draft_id", "build_index", "won", "on_play", "user_n_games_bucket", "user_game_win_rate_bucket"]
use = meta + [f"{p}{n}" for n in names for p in ("opening_hand_", "drawn_", "deck_")]
dt = {c: np.float32 for c in use if c not in meta + ["draft_id"]}
Hs, Ds, Ms = [], [], []
for i, ch in enumerate(pd.read_csv(GD, usecols=use, chunksize=20000, dtype=dt)):
    f = lambda p: ch[[f"{p}{n}" for n in names]].fillna(0).values
    Hs.append((f("opening_hand_") + f("drawn_")).astype(np.int8)); Ds.append(f("deck_").astype(np.int8)); Ms.append(ch[meta].copy())
    print("chunk", i, flush=True)
M = pd.concat(Ms, ignore_index=True)
np.savez_compressed(f"{SCR}/syn_games.npz", H=np.vstack(Hs), D=np.vstack(Ds), names=np.array(names), draft=M.draft_id.values.astype(str),
    build=M.build_index.values.astype(np.int16), won=M.won.values.astype(np.int8), play=M.on_play.values.astype(np.int8),
    ngames=M.user_n_games_bucket.fillna(0).values.astype(np.int16), skill=M.user_game_win_rate_bucket.values.astype(np.float32))
print("cached", M.shape)
