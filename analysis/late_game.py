"""Late-game payoffs and their enablers, FRA (Akshin, 2026-11-05, from the deck review of RG Craftwork Crusher trophies).

Payoff P6: nonland card with mana value 6+ (the ramp_payoff tag). Late-game enabler E: a card with at least two of removal, defence,
tempo, draw (companion/advisor_v2/sets/FRA/roles.json). Ramp route Rm: the ramp tag.
Sample: Premier Draft, two-colour lanes (splashes kept), drafts that played one deck for the whole run, matches won over matches played.
Model: binomial, wins ~ P6 * E + P6 * Rm + skill + q + C(lane), q = mean lane GIH WR of the nonland cards, HC1 SEs.

Pre-registered (2026-11-05, before the first run):
- Two tests, Bonferroni alpha 0.025: P6:E and P6:Rm interactions.
- Changes code: P6:E positive at p < 0.025. The value of a 6+ card then scales with the late-game enablers held (and an enabler's
  with the 6+ cards held) in picker and builder, with the fitted interaction as the weight. Same for P6:Rm.
- Stops this line: neither interaction significant. In between does not exist here (two tests, each passes or not); no reruns with
  other enabler definitions.
Descriptive only: match win rate by P6 bucket x E bucket.

Second pre-registered test (Akshin, 2026-11-05, after seeing the first: P6:Rm passed, P6:E p 0.14): payoff P6g = 6+ cards whose
17Lands GIH WR (All Decks) is at least one SD over the set mean (the engine's finisher rule, with 6+ instead of 5+). Same model with
P6g in place of P6, same two tests at alpha 0.025, same rules. This is his definition, not a retry with a looser cut.

Third (Akshin, 2026-11-05, after test 2: P6g:E p 0.10, P6g:Rm p 0.028): enablers need not be multi-tag. E1 = a card with any one of
removal, defence, tempo, draw; Ec = such a card costing 3 or less. P6g payoff. Tests P6g:E1 and P6g:Ec; six interaction tests have
now been run on this question, so a result is reported against alpha 0.05 / 6 = 0.0083 as well as the per-model bar.

Fourth (Akshin, 2026-11-05, after test 3: P6g:E1 p 0.022): support needs concentration, 5 or more of EACH of removal, defence,
tempo and draw (S = 1 when all four reach 5). Test P6g:S; eight tests so far, alpha 0.05 / 8 = 0.0063 alongside 0.025.
`--write` stores this version for the engine (STRAT_LATE), whatever the result (Akshin's direction).
"""
import gzip, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf

R = Path(__file__).resolve().parents[1]
CONTROL = {"removal", "defence", "tempo", "draw"}
front = lambda n: n.split(" // ")[0]
roles = {r["name"]: set(r["tags"]) for r in json.loads((R / "companion/advisor_v2/sets/FRA/roles.json").read_text())}
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)

wrs = [w for c in cards.values() if (w := (c.get("deck_colors", {}).get("All Decks") or {}).get("gihwr"))]
GOOD = np.mean(wrs) + np.std(wrs)
good = lambda c: (c.get("cmc") or 0) >= 6 and ((c.get("deck_colors", {}).get("All Decks") or {}).get("gihwr") or 0) >= GOOD
raw = [d for d in json.loads(gzip.open(R / "analysis/out/decks_FRA.json.gz").read()) if d["format"] == "PremierDraft"]
per_draft = Counter(d["draft"] for d in raw)
rows = []
for d in raw:
    if per_draft[d["draft"]] > 1 or len(d["lane"]) != 2 or sum(d["counts"].values()) != 40 or d["deck_games"] < 1: continue
    r = dict(lane=d["lane"], skill=d["skill"], w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"], P6=0, P6g=0, E=0, E1=0, Ec=0, Rm=0)
    qs = []
    for n, k in d["counts"].items():
        c = cards.get(front(n), {})
        if "Land" in (c.get("types") or []) or n in ("Plains", "Island", "Swamp", "Mountain", "Forest"): continue
        t = roles.get(n) or roles.get(front(n)) or set()
        g = (c.get("deck_colors", {}).get(d["lane"]) or c.get("deck_colors", {}).get("All Decks") or {}).get("gihwr")
        if g: qs += [g] * k
        r["P6"] += k * ((c.get("cmc") or 0) >= 6)
        r["P6g"] += k * good(c)
        r["E"] += k * (len(t & CONTROL) >= 2)
        r["E1"] += k * bool(t & CONTROL)
        r["Ec"] += k * (bool(t & CONTROL) and (c.get("cmc") or 0) <= 3)
        r["Rm"] += k * ("ramp" in t)
        for x in CONTROL: r[x] = r.get(x, 0) + k * (x in t)
    r["q"] = np.mean(qs)
    rows.append(r)
df = pd.DataFrame(rows)
print(f"decks {len(df)}; P6 mean {df.P6.mean():.2f}, E mean {df.E.mean():.2f} (sd {df.E.std():.2f}), Rm mean {df.Rm.mean():.2f}", file=sys.stderr)

m = smf.glm("w + l ~ P6 * E + P6 * Rm + skill + q + C(lane)", df, family=sm.families.Binomial()).fit(cov_type="HC1")
base = df.w.sum() / (df.w + df.l).sum()
pts = lambda b: 100 * (1 / (1 + np.exp(-(np.log(base / (1 - base)) + b))) - base)
print(f"match win rate {100 * base:.1f}%; alpha 0.025 for the two interactions\n")
print("term          coef      SE       p     ~match-WR pts per unit")
for t in ("P6", "E", "Rm", "P6:E", "P6:Rm"):
    print(f"{t:10s} {m.params[t]:+.4f}  {m.bse[t]:.4f}  {m.pvalues[t]:.4f}  {pts(m.params[t]):+.2f}"
          + ("  PASS" if t in ("P6:E", "P6:Rm") and m.pvalues[t] < 0.025 and m.params[t] > 0 else ""))

# what one more 6+ card is worth at low / median / high enabler counts (Rm at its mean)
print("\nOne more 6+ card, match-WR pts, by late-game enablers in the deck (ramp at its mean):")
for e in np.percentile(df.E, [10, 50, 90]):
    b = m.params["P6"] + m.params["P6:E"] * e + m.params["P6:Rm"] * df.Rm.mean()
    print(f"  E = {e:.0f}: {pts(b):+.2f}")

df["P6b"] = df.P6.clip(upper=3); df["Eb"] = pd.qcut(df.E, 3, labels=["low E", "mid E", "high E"])
g = df.groupby(["P6b", "Eb"], observed=True).agg(decks=("w", "size"), w=("w", "sum"), l=("l", "sum"))
g["match_wr"] = (100 * g.w / (g.w + g.l)).round(1)
print("\nMatch win rate by 6+ cards (3 = 3 or more) and enabler tercile (descriptive, no controls):")
print(g[["decks", "match_wr"]].unstack("Eb").to_string())
print("\nE tercile cut points:", pd.qcut(df.E, 3, retbins=True)[1].round(1).tolist())

print(f"\n=== Test 2: P6g = 6+ cards with GIH WR >= {GOOD:.1f} (set mean + 1 SD): "
      + ", ".join(sorted(front(c["name"]) for c in cards.values() if "Land" not in (c.get("types") or []) and good(c))))
print(f"decks with a P6g card: {(df.P6g > 0).sum()} of {len(df)}")
m2 = smf.glm("w + l ~ P6g * E + P6g * Rm + skill + q + C(lane)", df, family=sm.families.Binomial()).fit(cov_type="HC1")
for t in ("P6g", "E", "Rm", "P6g:E", "P6g:Rm"):
    print(f"{t:10s} {m2.params[t]:+.4f}  {m2.bse[t]:.4f}  {m2.pvalues[t]:.4f}  {pts(m2.params[t]):+.2f}"
          + ("  PASS" if t in ("P6g:E", "P6g:Rm") and m2.pvalues[t] < 0.025 and m2.params[t] > 0 else ""))
print("One more good 6+ card, match-WR pts, by late-game enablers (ramp at its mean):")
for e in np.percentile(df.E, [10, 50, 90]):
    b = m2.params["P6g"] + m2.params["P6g:E"] * e + m2.params["P6g:Rm"] * df.Rm.mean()
    print(f"  E = {e:.0f}: {pts(b):+.2f}")

for e in ("E1", "Ec"):
    m3 = smf.glm(f"w + l ~ P6g * {e} + P6g * Rm + skill + q + C(lane)", df, family=sm.families.Binomial()).fit(cov_type="HC1")
    t = f"P6g:{e}"
    print(f"\n=== Test 3 ({e}: {'any one control tag' if e == 'E1' else 'control tag and mana value 3 or less'}; mean {df[e].mean():.1f}, sd {df[e].std():.1f})")
    print(f"{t:10s} {m3.params[t]:+.4f}  {m3.bse[t]:.4f}  p {m3.pvalues[t]:.4f}  {pts(m3.params[t]):+.2f} pts per payoff per enabler"
          + ("  PASS at 0.025" if m3.pvalues[t] < 0.025 and m3.params[t] > 0 else "") + ("  PASS at 0.0083" if m3.pvalues[t] < 0.0083 and m3.params[t] > 0 else ""))
    print("  one more good 6+ card by enablers (10th / 50th / 90th percentile): " + ", ".join(
        f"{x:.0f}: {pts(m3.params['P6g'] + m3.params[t] * x + m3.params['P6g:Rm'] * df.Rm.mean()):+.2f}" for x in np.percentile(df[e], [10, 50, 90])))

df["S"] = ((df.removal >= 5) & (df.defence >= 5) & (df.tempo >= 5) & (df.draw >= 5)).astype(int)
m4 = smf.glm("w + l ~ P6g * S + P6g * Rm + skill + q + C(lane)", df, family=sm.families.Binomial()).fit(cov_type="HC1")
print(f"\n=== Test 4 (S: 5+ of each of removal, defence, tempo, draw): {df.S.sum()} of {len(df)} decks ({100 * df.S.mean():.0f}%); "
      f"with a good 6+ card and S: {((df.P6g > 0) & (df.S == 1)).sum()}")
print("  share of decks reaching 5+:", {x: f"{100 * (df[x] >= 5).mean():.0f}%" for x in sorted(CONTROL)})
for t in ("P6g", "S", "P6g:S", "P6g:Rm"):
    print(f"{t:10s} {m4.params[t]:+.4f}  {m4.bse[t]:.4f}  p {m4.pvalues[t]:.4f}  {pts(m4.params[t]):+.2f}"
          + ("  PASS at 0.025" if t == "P6g:S" and m4.pvalues[t] < 0.025 and m4.params[t] > 0 else "")
          + ("  PASS at 0.0063" if t == "P6g:S" and m4.pvalues[t] < 0.0063 and m4.params[t] > 0 else ""))
for s_ in (0, 1):
    b = m4.params["P6g"] + m4.params["P6g:S"] * s_ + m4.params["P6g:Rm"] * df.Rm.mean()
    print(f"  one more good 6+ card, {'with' if s_ else 'without'} the concentration: {pts(b):+.2f} match-WR pts")
g = df.assign(P=df.P6g.clip(upper=2)).groupby(["P", "S"]).agg(decks=("w", "size"), w=("w", "sum"), l=("l", "sum"))
g["match_wr"] = (100 * g.w / (g.w + g.l)).round(1)
print("  match win rate by good 6+ cards (2 = 2+) and concentration (descriptive, no controls):")
print(g[["decks", "match_wr"]].unstack("S").to_string())

if "--write" in sys.argv:  # Akshin: add it to synergy
    sys.path.insert(0, str(R))
    from companion.advisor_v2 import config
    out = {"points": round(float(pts(m4.params["P6g:S"])), 3), "good_wr": round(float(GOOD), 2), "cmc": 6, "tags": sorted(CONTROL), "each": 5,
           "p": round(float(m4.pvalues["P6g:S"]), 4)}
    print(out)
    config.write_fitted("FRA", "late_game", out, "analysis/late_game.py --write: P6g:S interaction, match-WR points per good 6+ card "
                        "(GIH WR >= set mean + 1 SD) when the deck holds 5+ of each of removal, defence, tempo, draw; one-deck Premier runs")
