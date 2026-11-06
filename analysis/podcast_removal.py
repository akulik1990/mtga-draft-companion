"""B5: does removal weigh the same by class and cmc band? Deck GLM as podcast_nonboard.py (binomial on wins, SEs clustered by draft,
two-colour decks, 60%+ skill; sensitivity all skills). Controls: mean lane GIH WR, creature count, splash, lane, skill, games bucket.
Removal = 48 reviewed cards (analysis/out/removal_classes_FRA.json), replaced by (a) counts per class, (b) counts per cmc band, (c) split
unconditional+damage vs conditional+bite/fight. tempo (2) and sweeper (4) merged into "other". Effects: win-rate points per card at 55%.
Equality: Wald across class coefficients. Replication: drafts split by parity of draft-id order.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/podcast_removal.py"""
import gzip, json, os
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 100 * 0.55 * 0.45
OUT = f"{R}/analysis/out"; os.makedirs(OUT, exist_ok=True)
raw = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
by = {}
for v in raw.values():
    n = v["name"].split(" // ")[0]
    if n in by and not v.get("isprimarycard", 1): continue
    by[n] = v
RC = json.load(open(f"{OUT}/removal_classes_FRA.json"))
MERGE = {"tempo": "other", "sweeper": "other", "bite/fight": "bite"}
CL = ["unconditional", "damage", "conditional", "bite", "other"]
BANDS = {"<=2": "b_low", "3-4": "b_mid", "5+": "b_high"}
cls = {n: MERGE.get(x["cls"].rstrip("*"), x["cls"].rstrip("*")) for n, x in RC.items()}
band = {n: BANDS[x["band"]] for n, x in RC.items()}
rows = []
for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")):
    if len(d["lane"]) != 2 or d["deck_games"] < 1: continue
    q, c = [], dict.fromkeys(CL + list(BANDS.values()) + ["cre"], 0)
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; v = by.get(n)
        if not v or "Land" in v["types"]: continue
        g = (v["deck_colors"].get(d["lane"]) or {}).get("gihwr") or (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
        if g: q += [g] * k
        c["cre"] += k * ("Creature" in v["types"])
        if n in RC: c[cls[n]] += k; c[band[n]] += k
    if len(q) >= 20:
        rows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=d["games_bucket"], splash=int(d["splash"] != ""),
                         y=d["deck_wins"] / d["deck_games"], g=d["deck_games"], quality=np.mean(q), **c))
ALL = pd.DataFrame(rows)
ALL["rem"] = ALL[CL].sum(axis=1); ALL["uncd"] = ALL.unconditional + ALL.damage; ALL["cond"] = ALL.conditional + ALL.bite
ctrl = "quality + cre + splash + skill + C(gb) + C(lane)"
lines = []
def P(s=""): print(s); lines.append(s)
def run(df, tag):
    G = pd.factorize(df.draft)[0]
    fit = lambda f: smf.glm(f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": G})
    P(f"\n### {tag}: {len(df)} decks, {df.draft.nunique()} drafts, {int(df.g.sum())} games")
    P("mean cards/deck: " + ", ".join(f"{x} {df[x].mean():.2f}" for x in CL + list(BANDS.values())))
    def show(m, names, label):
        P(f"\n{label}\n\n| term | pts/card | 95% CI | p |\n|---|---|---|---|")
        for x in names:
            b, se = m.params[x], m.bse[x]
            P(f"| {x} | {K*b:+.2f} | [{K*(b-1.96*se):+.2f}, {K*(b+1.96*se):+.2f}] | {m.pvalues[x]:.3g} |")
        if len(names) > 1:
            A = np.zeros((len(names) - 1, len(m.params))); ix = [list(m.params.index).index(x) for x in names]
            for i in range(len(names) - 1): A[i, ix[0]] = 1; A[i, ix[i + 1]] = -1
            w = m.wald_test(A, scalar=True); P(f"\nWald equality of {len(names)} coefficients: chi2({len(names)-1}) = {float(w.statistic):.2f}, p {float(w.pvalue):.3g}")
    show(fit(f"y ~ rem + {ctrl}"), ["rem"], "Total removal count (baseline)")
    show(fit(f"y ~ {' + '.join(CL)} + {ctrl}"), CL, "(a) per class")
    show(fit(f"y ~ {' + '.join(BANDS.values())} + {ctrl}"), list(BANDS.values()), "(b) per cmc band (<=2, 3-4, 5+)")
    show(fit(f"y ~ uncd + cond + other + {ctrl}"), ["uncd", "cond", "other"], "(c) unconditional+damage vs conditional+bite vs other")
for lbl, df in [("PRIMARY skill>=0.60", ALL[ALL.skill >= .60]), ("SENSITIVITY all skills", ALL)]:
    P(f"\n## {lbl}"); run(df, "all drafts")
    if lbl.startswith("PRIMARY"):
        o = {d: i for i, d in enumerate(sorted(df.draft.unique()))}
        h = df.draft.map(o) % 2
        for k in (0, 1): run(df[h == k], f"replication half {k + 1}")
open(f"{OUT}/removal_test_B5.txt", "w").write("\n".join(lines))
