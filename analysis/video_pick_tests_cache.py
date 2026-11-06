"""Stage 1 for analysis/video_pick_tests_h1a.py, _h1b.py, _h5.py: stream the FRA PremierDraft pick CSV once into int8 memmaps (pack, pool) + meta.
Output data/Temp/vid1106p/{pack,pool}.i8 (n x 295), meta.pkl (draft_id, pack_number, pick_number, pick, skill, ...), cols.json.
Run: systemd-run --user --scope -q -p MemoryMax=4G nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_pick_tests_cache.py"""
import os, json, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = f"{R}/data/Temp/replay/draft_data_public.FRA.PremierDraft.csv.gz"; OUT = f"{R}/data/Temp/vid1106p"
cols = list(pd.read_csv(CSV, nrows=1).columns)
pk = [c for c in cols if c.startswith("pack_card_")]; pl = [c for c in cols if c.startswith("pool_")]
names = [c[10:] for c in pk]; assert names == [c[5:] for c in pl]
meta_cols = ["draft_id", "pack_number", "pick_number", "pick", "rank", "event_match_wins", "event_match_losses", "user_n_games_bucket", "user_game_win_rate_bucket"]
N = sum(1 for _ in pd.read_csv(CSV, usecols=["draft_id"], chunksize=500000, engine="c") for _ in [0]) if False else sum(len(c) for c in pd.read_csv(CSV, usecols=["draft_id"], chunksize=500000))
print("rows", N, flush=True)
A = np.lib.format.open_memmap(f"{OUT}/pack.npy", "w+", np.int8, (N, len(names))); B = np.lib.format.open_memmap(f"{OUT}/pool.npy", "w+", np.int8, (N, len(names)))
dt = {c: np.int8 for c in pk + pl}; metas = []; i = 0
for ch in pd.read_csv(CSV, usecols=meta_cols + pk + pl, chunksize=100000, dtype=dt):
    n = len(ch); A[i:i + n] = ch[pk].values; B[i:i + n] = ch[pl].values; metas.append(ch[meta_cols]); i += n
    print(i, flush=True)
A.flush(); B.flush(); pd.concat(metas, ignore_index=True).to_pickle(f"{OUT}/meta.pkl"); json.dump(names, open(f"{OUT}/cols.json", "w"))
print("done", i)
