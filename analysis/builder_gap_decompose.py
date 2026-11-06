"""Part 1(a): his 40 vs our build on the video pool, under the deck model (yardstick "deckmodel") and the builder's own
("deckmodel_top3"): logit pieces, shape terms, per-card contributions. Output: analysis/out/builder_gap_decompose.txt"""

import sys

from builder_gap_common import *  # noqa: F401,F403

setup()
pool = pool_cards()
his = his_deck(pool)
ours = session.own_builds("FRA", pool)
print("pool", len(pool), "builds", [b["label"] for b in ours])
for b in ours:
    b["model_wr"] = session.deck_win_rate("FRA", b)
b = ours[0]
print("builder settings: OBJECTIVE", deckbuilder.OBJECTIVE, "MODEL", deckbuilder.MODEL, "MAX_SPLASH", deckbuilder.MAX_SPLASH,
      "MAX_SPLASH_BASICS", deckbuilder.MAX_SPLASH_BASICS)
print(f"his deck_win_rate {session.deck_win_rate('FRA', his)}  main {his['main']} splash {his['splash']} splash_cards {his['splash_cards']}")
print(f"ours {b['label']} deck_win_rate {b['model_wr']} (builder predicted {b['deck_wr']}) main {b['main']} splash '{b['splash']}'")

for section in ("deckmodel", "deckmodel_top3"):
    m = DeckModel("FRA", section)
    A = decompose(m, counts_of(his), his["main"], his["splash"])
    B = decompose(m, counts_of(b), b["main"], b["splash"])
    print(f"\n=== {section}: his {A['wr']:.2f}  ours {B['wr']:.2f}  gap {A['wr']-B['wr']:.2f}  (logit gap {A['logit']-B['logit']:.4f})")
    # share of the gap by piece, in logit then converted by the local slope at the midpoint
    mid = (A["logit"] + B["logit"]) / 2
    slope = 100 / (1 + np.exp(-mid)) * (1 - 1 / (1 + np.exp(-mid)))  # WR points per logit unit at the midpoint
    rows = [("cards (sum of per-copy)", A["cards"] - B["cards"])]
    for k in A["terms"]:
        rows.append((f"shape {k} (his {A['raw'].get(k.split('^')[0], float('nan')):.0f} / ours {B['raw'].get(k.split('^')[0], float('nan')):.0f})",
                     A["terms"][k] - B["terms"][k]))
    print(f"{'piece':50s} logit   WR points (x{slope:.1f})")
    for n, d in rows:
        if abs(d) > 1e-4:
            print(f"{n:50s} {d:+.4f} {d*slope:+.2f}")
    print(f"{'total':50s} {A['logit']-B['logit']:+.4f} {(A['logit']-B['logit'])*slope:+.2f}")
    pc = {}
    for n in set(A["per_card"]) | set(B["per_card"]):
        pc[n] = (A["per_card"].get(n, 0) - B["per_card"].get(n, 0), counts_of(his).get(n, 0), counts_of(b).get(n, 0))
    print("per card, his minus ours (logit; his count / our count), biggest first:")
    for n, (d, h, o) in sorted(pc.items(), key=lambda kv: -abs(kv[1][0]))[:30]:
        if abs(d) > 1e-4:
            print(f"  {n:40s} {d:+.4f} ({d*slope:+.2f} WR) {h}/{o}")
