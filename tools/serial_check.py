"""Is a deck's next game result tied to its previous one? (needs the cache written by tools/deck_context_controls.py)

    uv run python tools/serial_check.py

P(win) after a win vs after a loss within a skill band, and P(win) by record before the game (wins - losses).
"""
import pickle, collections
g = pickle.load(open("/home/morris/Projects/mtga-draft-companion/data/Temp/context_games_FRA.pkl", "rb"))
# g: (fmt, draft, build, time, skill, games, rank, on_play, mull, omull, won)
decks = collections.defaultdict(list)
for x in g:
    decks[(x[0], x[1], x[2])].append((x[3], x[10], x[4]))
def band(sk):
    try: v = float(sk)
    except: return "?"
    return "0.70+" if v > 0.69 else ("0.60-0.68" if v >= 0.595 else "<0.60")
for fmt in ("PremierDraft", "PickTwoDraft"):
    pw = collections.defaultdict(lambda: [0, 0])  # (band, prev result) -> [wins, n]
    rec = collections.defaultdict(lambda: [0, 0])  # (band, prior W-L) -> ...
    for (f, d, b), rows in decks.items():
        if f != fmt: continue
        rows.sort()
        W = L = 0
        for i, (t, won, sk) in enumerate(rows):
            bd = band(sk)
            if i > 0:
                k = (bd, rows[i-1][1]); pw[k][0] += won; pw[k][1] += 1
            k = (bd, W - L); rec[k][0] += won; rec[k][1] += 1
            W += won; L += (not won)
    print(f"\n{fmt}: P(win game | previous game result), games 2+ of a deck")
    for bd in ("<0.60", "0.60-0.68", "0.70+"):
        a, b2 = pw[(bd, True)], pw[(bd, False)]
        print(f"  {bd:10} after win {100*a[0]/a[1]:.1f}% (n {a[1]})   after loss {100*b2[0]/b2[1]:.1f}% (n {b2[1]})   diff {100*(a[0]/a[1]-b2[0]/b2[1]):+.1f}")
    print(f"{fmt}: P(win) by record before the game (wins - losses)")
    for bd in ("<0.60", "0.70+"):
        print("  ", bd, "  ".join(f"{d:+d}:{100*rec[(bd,d)][0]/rec[(bd,d)][1]:.0f}%({rec[(bd,d)][1]})" for d in range(-2, 7) if rec[(bd, d)][1] >= 200))
