"""Step 4 (handover 11-05e): winning vs losing FRA decks per colour-pair strategy, with Akshin's final role tags (companion/advisor_v2/sets/FRA/roles.json).

Sample: PremierDraft, two-colour lane equal to the pair (splashes kept), skill recorded, only drafts that played one deck for the
whole run. Winners: trophies (7 wins). Losers: finished runs at 0-1 wins (3 losses). Corrected 2026-11-05 after Akshin's review: the
first run counted 6 wins as winning (6-3 is a run that lost out, 6-2 an unfinished deck record) and used per-deck records in drafts
whose deck changed mid-run; the test, tags and pass rule are unchanged. Per deck: count of each role (copies included), top_payoff present, mean 17Lands
GIH win rate of nonland cards (q). UR's enabler, spells, is every noncreature nonland card (from card types, not the
table). top_payoff is any top payoff in the deck, not only the strategy's own.

Two numbers per (strategy, tag): raw winners-minus-losers mean with SE, and the same contrast adjusted for skill and q
(linear probability model, winner ~ tag + skill + q, HC1 SE). A tag's raw difference that is only card quality shows up as a shrunk
adjusted one.

Pre-registered (2026-11-05, before the first run):
- Changes code: a strategy's enabler, payoff or top_payoff with adjusted p < 0.05 / (tests in this file), winners higher. That
  strategy's synergy credit goes into the picker/builder with that sign. Winners significantly lower: no synergy credit for that tag.
- Stops the line for good: fewer than 2 of the 9 strategies have a passing enabler/payoff/top_payoff result. Then the tags are wired
  for deck shape only (removal, draw, defence, tempo, delay, ramp), no per-strategy synergy bonus.
- In between (2+ strategies, but nothing else): wire synergy only for the passing strategies; no reruns with other cut-offs.
"""
import gzip, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf

R = Path(__file__).resolve().parents[1]
STRAT = {  # docs/archetypes-FRA.md
    "jace": (["WU", "UG"], "jace_enabler", "jace_payoff"), "gy": (["UB"], "mill", "gy_payoff"),
    "ping": (["BR"], "ping_enabler", "ping_payoff"), "ramp": (["RG"], "ramp", "ramp_payoff"),
    "lifegain": (["WG"], "lifegain_enabler", "lifegain_payoff"), "death": (["WB"], "death_enabler", "death_payoff"),
    "spells": (["UR"], "spells", "spells_payoff"), "etb": (["BG"], "etb_enabler", "etb_payoff"),
    "counters": (["WR"], "counters_enabler", "counters_payoff"),
}
SHAPE = ["removal", "tempo", "delay", "draw", "defence", "ramp"]

front = lambda n: n.split(" // ")[0]
roles = {r["name"]: set(r["tags"]) for r in json.loads((R / "companion/advisor_v2/sets/FRA/roles.json").read_text())}
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)

def tags_of(n): return roles.get(n) or roles.get(front(n)) or set()

def gih(n, lane):
    d = cards.get(front(n), {}).get("deck_colors", {})
    return next((d[k]["gihwr"] for k in (lane, "All Decks") if (d.get(k) or {}).get("gihwr")), np.nan)

pairs = {p for v in STRAT.values() for p in v[0]}
rows = []
raw = [d for d in json.loads(gzip.open(R / "analysis/out/decks_FRA.json.gz").read()) if d["format"] == "PremierDraft"]
rows_per_draft = Counter(d["draft"] for d in raw)
for d in raw:
    if rows_per_draft[d["draft"]] > 1 or d["lane"] not in pairs or sum(d["counts"].values()) != 40: continue
    w, l = d["deck_wins"], d["deck_games"] - d["deck_wins"]
    win = 1 if d["trophy"] else 0 if (d["match_wins"] <= 1 and l == 3) else None
    if d["deck_games"] < 1: continue
    r = dict(lane=d["lane"], win=win, skill=d["skill"], w=w, l=l)
    qs = []
    for n, k in d["counts"].items():
        c = cards.get(front(n), {})
        if "Land" in (c.get("types") or []) or n in ("Plains", "Island", "Swamp", "Mountain", "Forest"): continue
        qs += [gih(n, d["lane"])] * k
        for t in tags_of(n) | ({"spells"} if "Creature" not in (c.get("types") or []) else set()): r[t] = r.get(t, 0) + k
    r["q"] = np.nanmean(qs)
    rows.append(r)
df = pd.DataFrame(rows)
df["win"] = df.win.astype(float)
df = df.fillna({c: 0 for c in df.columns if c != "win"})
alld, df = df, df[df.win.notna()].copy()
df["top_payoff"] = (df.get("top_payoff", 0) > 0).astype(int)

tests = [(s, t) for s, (_, e, p) in STRAT.items() for t in (e, p, "top_payoff")]
ALPHA = 0.05 / len(tests)
out = []
for s, (lanes, e, p) in STRAT.items():
    sub = df[df.lane.isin(lanes)]
    nw, nl = int(sub.win.sum()), int((1 - sub.win).sum())
    for t in dict.fromkeys((e, p, "top_payoff", *SHAPE)):
        x = sub[t] if t in sub else pd.Series(0, index=sub.index)
        a, b = x[sub.win == 1], x[sub.win == 0]
        raw, se = a.mean() - b.mean(), np.sqrt(a.var() / len(a) + b.var() / len(b))
        adj = adj_se = pv = np.nan
        if x.std() > 0:
            m = smf.ols("win ~ x + skill + q", sub.assign(x=x)).fit(cov_type="HC1")
            # winner-probability points per extra card (or for presence, for top_payoff)
            adj, adj_se, pv = 100 * m.params["x"], 100 * m.bse["x"], m.pvalues["x"]
        out.append(dict(strategy=s, lanes="/".join(lanes), tag=t, n_win=nw, n_lose=nl, win_mean=a.mean(), lose_mean=b.mean(),
                        raw=raw, raw_se=se, adj_pts=adj, adj_se=adj_se, p=pv,
                        tested=(s, t) in tests, passes=bool((s, t) in tests and pv < ALPHA)))
res = pd.DataFrame(out)
res.to_csv(R / "analysis/out/strategy_win_loss_FRA.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 200)
print(f"decks {len(df)}; alpha {ALPHA:.4f} over {len(tests)} tests", file=sys.stderr)
print(res.round(3).to_string(index=False))

# Descriptive, not pre-registered (Akshin, 2026-11-05: show the weight itself). All decks of each pair, not only the extremes:
# match wins ~ enablers + payoffs + skill + q (binomial, wins over matches), as match-win-rate points per extra card at the pair's mean.
from scipy.special import expit
wt = []
for s, (lanes, e, p) in STRAT.items():
    sub = alld[alld.lane.isin(lanes)].copy()
    sub["E"], sub["P"] = sub.get(e, 0), sub.get(p, 0)
    m = smf.glm("w + l ~ E + P + skill + q" + (" + C(lane)" if len(lanes) > 1 else ""), sub, family=sm.families.Binomial()).fit(cov_type="HC1")
    base = sub.w.sum() / (sub.w + sub.l).sum()
    pts = lambda b: 100 * (expit(np.log(base / (1 - base)) + b) - base)
    for t in ("E", "P"):
        b, se = m.params[t], m.bse[t]
        wt.append(dict(strategy=s, tag=e if t == "E" else p, decks=len(sub), mean_count=sub[t].mean(), pts_per_card=pts(b),
                       lo=pts(b - 1.96 * se), hi=pts(b + 1.96 * se), p=m.pvalues[t]))
wt = pd.DataFrame(wt)
wt.to_csv(R / "analysis/out/strategy_weight_FRA.csv", index=False)
print("\nMatch-win-rate points per extra card, all decks of the pair (descriptive)")
print(wt.round(3).to_string(index=False))
