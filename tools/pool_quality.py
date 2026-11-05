"""Pool-conditional check, part 2: was the removal left in the sideboard worse than what was played?

    uv run python tools/pool_quality.py FRA      # about 30 s; writes <data>/tables/SET/13_pool_quality.csv

Uses the sideboard cache of tools/pool_check.py. For decks with 3-5 removal spells that left castable removal in the sideboard: best
sideboard removal GIH WR minus the weakest played removal GIH WR (points). Then the win-rate effect of "left an upgrade" against "left
only weaker removal" against "left none", with the deck's removal count held (table 10 bins) and the other fields as in table 10.
GIH WR here is the public set data, which contains these games: a mild leak, as in the 'quality+' variant of table 10.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import build, write  # noqa: E402
from pool_check import cache_path, num, run  # noqa: E402


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    import json

    import companion.advisor_v2.engine  # noqa: F401
    from companion.advisor_v2 import config
    from companion.advisor_v2.lanes import castable

    config.activate(set_code)
    D = load_data(set_code, "all")
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    cols = dict(cols)
    for k, v in F.items():
        cols.setdefault(k, v)
    SB = pickle.loads(cache_path(set_code).read_bytes())
    table = card_table(set_code)
    rec = {}
    for c in json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            rec.setdefault(n, c)
    is_rm = lambda n: n in table and not num(table[n].get("land")) and num(table[n].get("role_removal")) > 0 and num(table[n].get("gihwr")) > 0
    N = len(D["w"])
    best_left, worst_played, n_left, n_played = np.full(N, np.nan), np.full(N, np.nan), np.zeros(N), np.zeros(N)
    for i, (fg, r, d) in enumerate(zip(D["fg"], D["rows"], D["decks"])):
        side = SB.get((fg[0], fg[1], fg[2]), {})
        lane = r["combo"].split("+")[0] + (r.get("splash") or "")
        played = [num(table[n]["gihwr"]) for n in d["counts"] if is_rm(n)]
        left = [num(table[n]["gihwr"]) for n in side if is_rm(n) and n in rec and castable(rec[n], lane)]
        n_left[i], n_played[i] = len(left), len(played)
        if played:
            worst_played[i] = min(played)
        if left:
            best_left[i] = max(left)
    dv = np.clip(np.round(np.asarray(cols["removal"], float)), 3, 9).astype(int)
    gap = best_left - worst_played
    low = (dv <= 5) & (n_left > 0) & ~np.isnan(gap)
    print(f"decks with 3-5 removal and castable removal in the sideboard: {int(low.sum())}")
    print(f"  weakest played removal GIH WR median {np.median(worst_played[low]):.1f}, best left {np.median(best_left[low]):.1f}; gap (left - played) median {np.median(gap[low]):+.1f}, "
          f"share with an upgrade (gap > 0) {100 * (gap[low] > 0).mean():.0f}%, big (> 2 points) {100 * (gap[low] > 2).mean():.0f}%", flush=True)
    for d in (3, 4, 5):
        m = low & (dv == d)
        print(f"  deck at {d}{'-' if d == 3 else ''}: {int(m.sum())} decks, upgrade {100 * (gap[m] > 0).mean():.0f}%, big {100 * (gap[m] > 2).mean():.0f}%, median gap {np.median(gap[m]):+.1f}")
    cell = np.zeros(N, int)  # 0 none left or not low-removal, 1 left weaker only, 2 small upgrade (0-2], 3 big upgrade (> 2)
    lowdeck = (dv <= 5) & (n_left > 0) & ~np.isnan(gap)
    cell[lowdeck & (gap <= 0)] = 1
    cell[lowdeck & (gap > 0) & (gap <= 2)] = 2
    cell[lowdeck & (gap > 2)] = 3
    keep = [i for i, f in enumerate(structure) if f not in {"role_removal", "rm_cheap", "rm_late"}]
    cnt = np.bincount(dv - 3, minlength=7)
    deck_cols = [(dv - 3 == i).astype(float) for i in range(7) if i != 3 and cnt[i] >= 30]  # reference 6
    out = []
    names = {1: "left only weaker", 2: "left a small upgrade (<= 2 pts)", 3: "left a big upgrade (> 2 pts)"}
    bg = np.hstack([sets["quality"], sets["colour"], sets["structure"][:, keep]])
    B = np.column_stack(deck_cols + [(cell == c).astype(float) for c in (1, 2, 3)])
    b, V = run(Xc, bg, B, D, ("quality",))
    k = len(deck_cols)
    print("\neffect vs the same deck count with no castable removal left (points, SE); deck bins held:")
    for j, c in enumerate((1, 2, 3)):
        e, s = PTS * b[k + j], PTS * np.sqrt(V[k + j, k + j])
        print(f"  {names[c]:34} {e:+.2f} (SE {s:.2f}), {int((cell == c).sum())} decks")
        out.append(dict(cell=names[c], decks=int((cell == c).sum()), points=round(e, 2), se=round(s, 2)))
    dx = np.zeros(len(b)); dx[k + 2] = 1; dx[k] = -1
    print(f"  big upgrade - weaker only: {PTS * (b[k + 2] - b[k]):+.2f} (SE {PTS * np.sqrt(dx @ V @ dx):.2f})")
    n2, n3 = int((cell == 2).sum()), int((cell == 3).sum())
    dx = np.zeros(len(b)); dx[k + 1] = n2 / (n2 + n3); dx[k + 2] = n3 / (n2 + n3); dx[k] = -1
    e, s_ = PTS * (dx @ b), PTS * np.sqrt(dx @ V @ dx)
    print(f"  any upgrade left ({n2 + n3} decks) - weaker only: {e:+.2f} (SE {s_:.2f}, z {e / s_:+.1f})")
    out.append(dict(cell="any upgrade - weaker only", decks=n2 + n3, points=round(e, 2), se=round(s_, 2)))
    dx0 = np.zeros(len(b)); dx0[k + 1] = n2 / (n2 + n3); dx0[k + 2] = n3 / (n2 + n3)
    e0, s0 = PTS * (dx0 @ b), PTS * np.sqrt(dx0 @ V @ dx0)
    print(f"  any upgrade left vs none left: {e0:+.2f} (SE {s0:.2f})")
    out.append(dict(cell="any upgrade vs none left", decks=n2 + n3, points=round(e0, 2), se=round(s0, 2)))
    write(tables.path(set_code, "13_pool_quality.csv"), out)


if __name__ == "__main__":
    main()
