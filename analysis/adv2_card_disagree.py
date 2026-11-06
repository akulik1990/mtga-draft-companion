"""Per-card disagreement between v2 and the players (saved adv1 replays, 196 drafts, no new runs).

    python3 analysis/adv2_card_disagree.py

  overrated  = v2 ranked the card #1 but the player took something else: miss rate = misses / times v2's #1
  underrated = the player took the card but v2's #1 was another card: miss rate = misses / times he took it
Cards with at least MIN events on the denominator. GIH WR is the card's All Decks win rate in games in hand (17Lands).
Disagreement is not proof v2 is wrong; the GIH column and the within-deck test are the next check.
"""
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "Temp" / "adv1"
MIN = 12
ratings = {c["name"]: c for c in json.load(open(ROOT / "data" / "Sets" / "FRA_PremierDraft_All_Data.json"))["card_ratings"].values()}


def gih(n):
    return float(((ratings.get(n) or {}).get("deck_colors") or {}).get("All Decks", {}).get("gihwr") or 0)


top_n, top_miss, took_n, took_miss, score_sum = Counter(), Counter(), Counter(), Counter(), Counter()
for tag in ("sp3", "two"):
    picks = {}
    for d in json.load(gzip.open(D / f"{tag}.drafts.json.gz")):
        for pack, pick, taken, _p, _pl in d["picks"]:
            picks[(d["id"], pack, pick)] = taken
    for r in json.load(open(D / f"{tag}.log.json")):
        if not r[3]:
            continue
        taken = picks.get((r[0], r[1], r[2]))
        top, sc = r[3][0]
        top_n[top] += 1
        score_sum[top] += sc
        took_n[taken] += 1
        if top != taken:
            top_miss[top] += 1
            took_miss[taken] += 1


def table(title, miss, denom, extra=None):
    rows = [(n, miss[n], denom[n]) for n in denom if denom[n] >= MIN]
    rows.sort(key=lambda x: (-x[1] / x[2], -x[2]))
    out = [title, "", "| Card | Events | Disagreed | Rate | GIH WR |" , "|---|---|---|---|---|"]
    for n, m, t in rows[:12]:
        out.append(f"| {n} | {t} | {m} | {100 * m / t:.0f}% | {gih(n) or '-'} |")
    return out


overall = sum(top_miss.values()) / sum(top_n.values())
out = [f"{sum(top_n.values())} picks over 196 drafts; v2's #1 differs from the player's pick in {100 * overall:.0f}%.", ""]
out += table(f"OVERRATED: v2's #1, player took something else (min {MIN} times as #1)", top_miss, top_n)
out += [""] + table(f"UNDERRATED: the player took it, v2's #1 was another card (min {MIN} times taken)", took_miss, took_n)
# GIH WR comparison across all well-sampled cards
lo = [(n, gih(n)) for n in top_n if top_n[n] >= MIN and top_miss[n] / top_n[n] >= 0.6 and gih(n)]
hi = [(n, gih(n)) for n in took_n if took_n[n] >= MIN and took_miss[n] / took_n[n] >= 0.6 and gih(n)]
avg = lambda xs: sum(g for _, g in xs) / len(xs) if xs else 0
allg = [gih(n) for n in took_n if took_n[n] >= MIN and gih(n)]
out += ["", f"Mean GIH WR: cards v2 overrates (miss rate >= 60%, n {len(lo)}): {avg(lo):.1f}; cards v2 underrates (n {len(hi)}): {avg(hi):.1f}; all well-sampled cards: {sum(allg) / len(allg):.1f}"]
text = "\n".join(out)
(ROOT / "analysis" / "out" / "adv2_card_disagree.txt").write_text(text + "\n")
print(text)
