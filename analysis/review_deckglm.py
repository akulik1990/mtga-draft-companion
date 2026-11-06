"""Review checks on the common deck-level GLM (podcast_removal.py and clones): (1) the skill bucket is the player's win rate
including the deck's own games (docs/context-controls-FRA.md), so "skill >= 0.60" selects on the outcome for low-game players:
show corr(skill, deck WR) by games bucket, and the removal coefficient by games bucket inside the primary sample;
(2) the mean-GIH "quality" control contains the removal cards' own ratings, so the removal coefficient is "a removal card against a
non-removal card of the same GIH WR": re-estimate with quality from non-removal spells only (the total effect of a removal slot at fixed
quality of the rest) and with the removal cards' own mean GIH as a separate term; (3) weights: games-weighted vs unweighted (one deck one vote).
Output analysis/out/review_deckglm.txt.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/review_deckglm.py"""
import gzip, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_counters import R, OUT, K, K1, cards, row

lines = []
def P(s=""): print(s); lines.append(s)
by = cards()
RC = json.load(open(f"{OUT}/removal_classes_FRA.json"))
rows = []
for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")):
    if len(d["lane"]) != 2 or d["deck_games"] < 1: continue
    q, qr, qn, cre, rem = [], [], [], 0, 0
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; v = by.get(n)
        if not v or "Land" in v["types"]: continue
        g = (v["deck_colors"].get(d["lane"]) or {}).get("gihwr") or (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
        if g:
            q += [g] * k
            (qr if n in RC else qn).extend([g] * k)
        cre += k * ("Creature" in v["types"]); rem += k * (n in RC)
    if len(q) < 20: continue
    rows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=d["games_bucket"], splash=int(d["splash"] != ""),
                     y=d["deck_wins"] / d["deck_games"], g=d["deck_games"], w=d["deck_wins"], quality=np.mean(q),
                     q_non=np.mean(qn) if qn else np.nan, q_rem=np.mean(qr) if qr else np.nan, cre=cre, rem=rem))
ALL = pd.DataFrame(rows)
ALL["q_rem_c"] = ALL.q_rem.fillna(ALL.q_rem.mean()) - ALL.q_rem.mean()

def fit(df, f, weighted=True):
    G = pd.factorize(df.draft)[0]
    kw = dict(freq_weights=df.g) if weighted else {}
    return smf.glm(f, df, family=sm.families.Binomial(), **kw).fit(cov_type="cluster", cov_kwds={"groups": G})
ctrl = "quality + cre + splash + skill + C(gb) + C(lane)"

P("# Review: the deck-level GLM\n")
P("## 1. Skill bucket contains the deck's own result\n")
P("corr(skill, deck WR) and share of decks at skill >= 0.60, by games bucket (1 = fewest games). If skill were a clean prior it would not\n"
  "correlate more with the deck's own WR when the player has few games.\n\n| games bucket | decks | corr(skill, y) | mean y | share skill>=.60 | mean y | skill>=.60 |\n|---|---|---|---|---|---|---|")
for gb, d in ALL.groupby("gb"):
    hi = d[d.skill >= .6]
    P(f"| {gb} | {len(d)} | {np.corrcoef(d.skill, d.y)[0,1]:+.3f} | {np.average(d.y, weights=d.g):.3f} | {(d.skill>=.6).mean():.2f} | {np.average(hi.y, weights=hi.g) if len(hi) else float('nan'):.3f} | n {len(hi)} |")

P("\nRemoval (REM48 count) coefficient, pts per card, inside the primary sample (skill >= 0.60), by games bucket group. Collider bias from\nselecting on an outcome-laden skill shows as a different slope where the leak is largest (few games).\n\n| sample | decks | pts/card | 95% CI | p |\n|---|---|---|---|---|")
PR = ALL[ALL.skill >= .6]
for lab, d in (("primary, all buckets", PR), ("primary, games bucket 1-3", PR[PR.gb <= 3]), ("primary, games bucket 4-6", PR[(PR.gb >= 4) & (PR.gb <= 6)]), ("primary, games bucket 7-9", PR[PR.gb >= 7]),
               ("ALL skills, bucket 7-9 (least leak)", ALL[ALL.gb >= 7]), ("ALL skills, bucket 1-3", ALL[ALL.gb <= 3]), ("ALL skills", ALL)):
    m = fit(d, f"y ~ rem + {ctrl}"); P(row(m, "rem").replace("| rem |", f"| {lab} | {len(d)} |"))

P("\n## 2. What the quality control does to the removal estimand\n")
P("Same sample (skill >= 0.60). (a) as run: quality = mean GIH of all spells; (b) quality = mean GIH of NON-removal spells only (removal's own\n"
  "rating no longer held, so the coefficient is the whole value of a removal slot at fixed quality of the rest, closer to the podcast claim);\n"
  "(c) both: non-removal quality + removal cards' own mean GIH (centred) as separate terms; (d) no quality control.\n\n| spec | pts/card | 95% CI | p |\n|---|---|---|---|")
for lab, f in (("(a) mean GIH all spells", f"y ~ rem + {ctrl}"), ("(b) mean GIH of non-removal spells", f"y ~ rem + q_non + cre + splash + skill + C(gb) + C(lane)"),
               ("(c) non-removal quality + removal's own GIH", f"y ~ rem + q_non + q_rem_c + cre + splash + skill + C(gb) + C(lane)"), ("(d) no quality control", f"y ~ rem + cre + splash + skill + C(gb) + C(lane)")):
    m = fit(PR, f); P(row(m, "rem").replace("| rem |", f"| {lab} |"))
    if "q_rem_c" in m.params: P(row(m, "q_rem_c").replace("| q_rem_c |", "|   (removal's own mean GIH, pts per rating point) |"))
P("\nAll skills, same four specs:\n\n| spec | pts/card | 95% CI | p |\n|---|---|---|---|")
for lab, f in (("(a)", f"y ~ rem + {ctrl}"), ("(b)", f"y ~ rem + q_non + cre + splash + skill + C(gb) + C(lane)"), ("(c)", f"y ~ rem + q_non + q_rem_c + cre + splash + skill + C(gb) + C(lane)"), ("(d)", f"y ~ rem + cre + splash + skill + C(gb) + C(lane)")):
    m = fit(ALL, f); P(row(m, "rem").replace("| rem |", f"| {lab} |"))

P("\n## 3. Weighting: games-weighted (as run) vs one deck one vote\n")
P("Deck games are 3 to 9 under the 7-win / 3-loss stop. Weighting by games gives winning decks more weight; the binomial likelihood is valid\nunder the stopping rule but the clustered SE and any misspecification are not. Unweighted = GLM on the deck win share with weight 1.\n\n| sample | weighted | unweighted |\n|---|---|---|")
for lab, d in (("primary", PR), ("all skills", ALL)):
    a, b = fit(d, f"y ~ rem + {ctrl}"), fit(d, f"y ~ rem + {ctrl}", weighted=False)
    P(f"| {lab} | {K*a.params['rem']:+.2f} [{K*(a.params['rem']-1.96*a.bse['rem']):+.2f}, {K*(a.params['rem']+1.96*a.bse['rem']):+.2f}] | {K*b.params['rem']:+.2f} [{K*(b.params['rem']-1.96*b.bse['rem']):+.2f}, {K*(b.params['rem']+1.96*b.bse['rem']):+.2f}] |")

P("\n## 4. The 'below 6 removal' shortfall (the basis of REMOVAL_SLACK) by games bucket\n")
P("REM48 < 6 dummy, pts for being short, quality control as run.\n\n| sample | decks | pts | 95% CI | p |\n|---|---|---|---|---|")
ALL["short"] = (ALL.rem < 6).astype(int)
for lab, d in (("primary", ALL[ALL.skill >= .6]), ("primary, bucket 1-3", ALL[(ALL.skill >= .6) & (ALL.gb <= 3)]), ("primary, bucket 7-9", ALL[(ALL.skill >= .6) & (ALL.gb >= 7)]),
               ("all skills", ALL), ("all skills, bucket 1-3", ALL[ALL.gb <= 3]), ("all skills, bucket 4-6", ALL[(ALL.gb >= 4) & (ALL.gb <= 6)]), ("all skills, bucket 7-9", ALL[ALL.gb >= 7])):
    m = fit(d, f"y ~ short + {ctrl}"); P(row(m, "short", K1).replace("| short |", f"| {lab} | {len(d)} |"))
open(f"{OUT}/review_deckglm.txt", "w").write("\n".join(lines))
