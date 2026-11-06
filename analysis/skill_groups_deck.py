"""Deck-level results re-reported by player skill group (CLAUDE.md evidence rule, 2026-11-06). Output analysis/out/skill_groups_deck.txt.
Groups by the deck's player skill bucket (never by the deck's result): good >= 0.60, average 0.52-0.58, weak < 0.52. Same specs as the originals
(foundations_deck.py Ca/Cb: all decks of the group, residual skill bucket + lane as controls, NO game-count term; gih_slope_heldout.py headline), only the grouping is new.
Run: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/skill_groups_deck.py"""
import os, re, sys, zlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = f"{R}/analysis/foundations_deck.py"; S = open(FD).read()
# Reuse the original's deck table, helpers (cell/est/fit) and land table verbatim; skip its diagnostics (it reads a whole game CSV) and everything after R5.
head = re.split(r"\n# -+ 0\. diagnostics", S)[0]
lands = S.split("# ------------------------------------------------------------------ 5. lands")[1].split('ls = ["O"')[0]
g = {"__file__": FD, "__name__": "fd"}; exec(compile(head, FD, "exec"), g); exec(compile(lands, FD, "exec"), g)
cell, K, POP1, POP2, LD, Q1a, Q1b, Q2a, Q2b, OUT = (g[k] for k in ("cell", "K", "POP1", "POP2", "LD", "Q1a", "Q1b", "Q2a", "Q2b", "OUT"))
POPT = POP1[POP1.tr.notna()]
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
GR = ["all", "good", "avg", "weak"]
def grp(df, k, col="skill"):
    s = df[col]
    return df if k == "all" else df[s >= .6 - 1e-9] if k == "good" else df[(s >= .52 - 1e-9) & (s <= .58 + 1e-9)] if k == "avg" else df[s < .52 - 1e-9]
SPEC = lambda df: df
def f2(e): return "n/a" if np.isnan(e[0]) else f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}]" + ("*" if e[3] < .05 else "")
def diff(a, b):
    """good minus weak, independent groups; SE from the CI half-widths."""
    if np.isnan(a[0]) or np.isnan(b[0]): return "n/a"
    sa, sb = (a[2] - a[1]) / 3.92, (b[2] - b[1]) / 3.92; se = (sa ** 2 + sb ** 2) ** .5; d = a[0] - b[0]
    from scipy.stats import norm
    return f"{d:+.2f} [{d - 1.96 * se:+.2f}, {d + 1.96 * se:+.2f}]" + ("*" if 2 * norm.sf(abs(d / se)) < .05 else "")
def block(title, pop, treat, Qa, Qb, extra, rows, make_pop=None, specs=("Ca", "Cb")):
    P(f"\n**{title}** (pts of deck WR; * p<0.05; n decks per group in the last row)\n")
    res = {k: cell(grp(pop, k), None, treat, Qa, Qb, extra, rows, specs=list(specs)) for k in GR}
    P("| term | spec | all decks | good (>=0.60) | average (0.52-0.58) | weak (<0.52) | good minus weak |\n|---|---|---|---|---|---|---|")
    for lab, _, _ in rows:
        for sp in specs:
            P(f"| {lab} | {sp} | " + " | ".join(f2(res[k][lab][sp]) for k in GR) + f" | {diff(res['good'][lab][sp], res['weak'][lab][sp])} |")
    P("| n decks | | " + " | ".join(str(len(grp(pop, k))) for k in GR) + " | |")
    return res
P("# Deck-level results by skill group (FRA)")
P(f"family-1 decks (two-colour, >=20 priced): {len(POP1)}; groups good/avg/weak: " + "/".join(str(len(grp(POP1, k))) for k in GR[1:]) + f"; all decks {len(POP2)}: " + "/".join(str(len(grp(POP2, k))) for k in GR[1:]))
P(f"lands table {len(LD)}: " + "/".join(str(len(grp(LD, k))) for k in GR[1:]))
P("\n## 3b. Lands 16 / 15 vs 17 (R5, pooled over the landcycler x dual strata)")
LD2 = LD
block("lands vs 17 (decks with 40 cards and 15-17 lands)", LD2, lambda d: " + ".join(v for v in ("l15", "l16") if d[v].sum() >= 30), "qa", "qb", "", [("16 vs 17", {"l16": 1}, K), ("15 vs 17", {"l15": 1}, K)])
block("shortfall against the builder's removal target minus 2, per card short", POPT, "sfall", Q1a("rem"), Q1b("rem"), "cre + splash", [("sfall", {"sfall": 1}, K)])
P("\n## 3d. Creatures (R9)")
block("creature count, pts per creature", POP1, "cre", Q1a("cre"), Q1b("cre"), "splash", [("cre", {"cre": 1}, K)])
block("creatures short of target minus 2, pts per creature short", POPT, "csf", Q1a("cre"), Q1b("cre"), "splash", [("csf", {"csf": 1}, K)])
P("\n## 2. Removal (R1)")
block("REM48 count, pts per card", POP1, "rem", Q1a("rem"), Q1b("rem"), "cre + splash", [("rem", {"rem": 1}, K)])
block("fewer than 6 REM48 (dummy)", POP1, "short6", Q1a("rem"), Q1b("rem"), "cre + splash", [("short6", {"short6": 1}, K)])
P("\n## 3a. Finishers (R8, deck-level analogue)")
block("finisher count, pts per card (removal, creatures, splash controlled)", POP1, "fin", Q1a("fin"), Q1b("fin"), "rem + cre + splash", [("fin", {"fin": 1}, K)])
P("\n## 3c. Two-drops (R2)")
block("cmc-2 creature count buckets vs 5", POP2, "b02 + b3 + b4 + b6p", Q2a("two"), Q2b("two"), "splash", [("0-2", {"b02": 1}, K), ("3", {"b3": 1}, K), ("4", {"b4": 1}, K), ("6+", {"b6p": 1}, K)])
# pool-level finishers (draft_features, event match record, 100+ players, n=23): residual skill bucket kept
import statsmodels.formula.api as smf, statsmodels.api as sm
D8 = pd.read_csv(f"{OUT}/draft_features_FRA.csv.gz"); D8["y"] = D8.event_match_wins / D8.trials
D8["rem"], D8["fin"], D8["dul"] = D8.removal.clip(3, 10), D8.finishers.clip(0, 4), D8.duals.clip(0, 6)
P("\n## 3a'. Finishers, pool level (event match record, n=23 pool-deck, 100+ players, M1)\n")
P("| group | drafts | finishers [CI] |\n|---|---|---|")
pe = {}
for k in GR:
    d = grp(D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100)], k, "user_game_win_rate_bucket")
    m = smf.glm("y ~ quality + C(pair) + C(user_n_games_bucket) + C(rank) + rem + fin + dul + C(user_game_win_rate_bucket)", d, family=sm.families.Binomial(), freq_weights=d.trials).fit(cov_type="HC1")
    pe[k] = g["est"](m, {"fin": 1}); P(f"| {k} | {len(d)} | {f2(pe[k])} |")
P(f"| good minus weak | | {diff(pe['good'], pe['weak'])} |")
# ---------------------------------------------------------------- GIH slope held out + splash
import gih_slope_heldout as gh   # streams the game file (capped), builds D, ratings; module-level only
P("\n## 1. GIH_SLOPE held out and splash at equal quality (gih_slope_heldout.py headline: two-colour, 100+ games players, lane ratings from the other half, threshold 100)\n")
P("Ratings always come from all players of the rating half; only the scored decks are split. Slope = deck WR pts per +1 mean-GIH pt; splash = WR pts at equal quality; threshold = -splash/slope in GIH pts (95% interval by simulation from the two CIs, assuming independence).")
rng = np.random.default_rng(0)
def thr(o):
    s = rng.normal(o["slope"], (o["hi"] - o["lo"]) / 3.92, 20000); p = rng.normal(o["splash"], (o["sphi"] - o["splo"]) / 3.92, 20000)
    return -o["splash"] / o["slope"], *np.percentile(-p / s, [2.5, 97.5])
def sgroups(pop, label):
    P(f"\n### {label}\n")
    res = {}
    for k in GR:
        sub = pop & (gh.D.skill.notna() & {"all": gh.D.skill.notna(), "good": gh.D.skill >= .6 - 1e-9, "avg": (gh.D.skill >= .52 - 1e-9) & (gh.D.skill <= .58 + 1e-9), "weak": gh.D.skill < .52 - 1e-9}[k])
        res[k] = gh.run(f"{label}: {k}", sub, 100, rel=False)
    for tag in ("held-out", "in-sample"):
        P(f"\n| {tag} | all | good | average | weak | good minus weak |\n|---|---|---|---|---|---|")
        for nm, a, lo, hi in (("slope", "slope", "lo", "hi"), ("splash", "splash", "splo", "sphi")):
            e = {k: (res[k][tag][a], res[k][tag][lo], res[k][tag][hi], 0.0) for k in GR}
            P(f"| {nm} | " + " | ".join(f"{e[k][0]:+.2f} [{e[k][1]:+.2f}, {e[k][2]:+.2f}]" for k in GR) + f" | {diff(e['good'], e['weak'])} |")
        t = {k: thr(res[k][tag]) for k in GR}
        P("| threshold (GIH pts) | " + " | ".join(f"{t[k][0]:.2f} [{t[k][1]:.2f}, {t[k][2]:.2f}]" for k in GR) + " | |")
    return res
sgroups(gh.two & (gh.D.user_n_games_bucket >= 100), "HEADLINE two-colour 100+ players")
sgroups(gh.two, "secondary: all players (skill bucket then contains the deck's own games; a poor control and a circular grouping)")
open(f"{R}/analysis/out/skill_groups_deck.txt", "w").write("\n".join(lines + ["", "---- gih_slope_heldout run output ----"] + gh.lines) + "\n")
