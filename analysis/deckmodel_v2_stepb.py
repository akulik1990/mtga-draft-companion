"""Step B: which score ranks decks' REAL win rates better, the builder's gihwr objective or the deck model? FRA, nix env (statsmodels).

PRE-REGISTERED (written 2026-11-06 before any Step B outcome was read; Step A outcomes were read, they do not enter B):
  Population: Premier decks (analysis/out/decks_FRA.json.gz, players with 100+ games), 2- or 3-colour lane, >= 1 game, no missing score. One row per
              (draft, build) with deck_wins / deck_games.
  Scores (analysis/deckmodel_v2_stepb_pred.py; all out-of-fold: the deck's draft was in neither the deck model's fit nor the GIH table):
    G = GIH_SLOPE 5.5 x mean GIH WR of the spells - SPLASH_TERM 3.1 if splash colours (the builder's OBJECTIVE gihwr for the 40, without its shape penalty/extras)
    M = logit of the cross-fitted deckmodel_v2 specification (skill-fixed context, tw 1, l2 10) at the typical context
  Primary: held-out binomial log loss per game of G minus M, D = (LL_G - LL_M); D > 0 = the deck model ranks decks better. Each score enters ONE logistic
           calibration layer: logit(win rate) = skill-bucket dummies (0.02) + a + b * score, fitted on the other four folds (Premier decks) and applied to the held-out
           fold; the skill dummies are the skill control. Cross-fitted over the five folds, so every deck is scored once.
  Inference: draft-clustered bootstrap (2000 resamples of drafts), percentile 95% CI, two-sided p = 2 min(tail); one primary, alpha 0.05, no multiplicity correction.
  Groups (fixed): good (skill >= 0.60), average (0.52-0.58), weak (< 0.52) by the player's skill bucket, and good minus weak with CI; same calibration layer, loss by group.
  Smallest effect: the CI's end on the side of the deck model being worse is reported as what is ruled out.
  Secondary (labelled): G with All Decks GIH only; G from the ratings file the builder reads (leaky: includes the deck's own games); M = specification of deckmodel_top3_v2;
    M = the stored model's specification (v1) cross-fitted; M v2 minus v1; nested additions (G given M, M given G, log loss gain of the second score); non-Premier decks
    (calibration on Premier); halves by fold parity. Limits: G is the builder's objective only before the shape penalty and structure-first search; the GIH table is
    our own train-fold count of the public game sample, not the ratings file (its GIH WR is on average 0.9 points higher, sd 0.6).
Output: analysis/out/deckmodel_v2_stepb.txt"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
lines = []
rng = np.random.default_rng(20261106)
B = 2000


def P(s=""):
    print(s, flush=True)
    lines.append(s)


DF = pd.read_csv(REPO / "data" / "Temp" / "dm1106" / "stepb_pred.csv.gz")
for k in ("lp_v1", "lp_v2", "lp_v2t3"):
    DF["M_" + k[3:]] = np.log(DF[k] / (100 - DF[k]))
DF = DF.dropna(subset=["G", "Gall", "Gfile"]).reset_index(drop=True)
DF["sk"] = DF.skill.round(2)
SKL = sorted(DF.sk.unique())
GROUPS = {"all": lambda d: np.ones(len(d), bool), "good": lambda d: (d.skill >= .6 - 1e-9).values, "avg": lambda d: ((d.skill >= .52 - 1e-9) & (d.skill <= .58 + 1e-9)).values,
          "weak": lambda d: (d.skill < .52 - 1e-9).values}


def design(d, cols):
    sk = np.clip(d.sk.values, 0.30, 0.76)
    levels = [0.32 + 0.02 * i for i in range(0, 23)]
    dummies = np.column_stack([(np.abs(sk - l) < 0.01).astype(float) for l in levels[1:]])
    keep = dummies.sum(0) >= 0  # fixed layout, empty columns dropped per fit below
    return np.column_stack([np.ones(len(d)), dummies] + [d[c].values.astype(float) for c in cols])


def crossfit(df, cols, test_mask_fn):
    """Held-out probabilities for df under the calibration layer on `cols`; test_mask_fn(k) -> (train rows, test rows)."""
    p = np.full(len(df), np.nan)
    X = design(df, cols)
    for k, (tr, te) in test_mask_fn.items():
        Xt = X[tr]
        live = Xt.std(0) > 0
        live[0] = True
        m = sm.GLM(df.wins.values[tr] / df.games.values[tr], Xt[:, live], family=sm.families.Binomial(), freq_weights=df.games.values[tr]).fit()
        p[te] = 1 / (1 + np.exp(-(X[te][:, live] @ m.params)))
    return p


def loss(df, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(df.wins.values * np.log(p) + (df.games.values - df.wins.values) * np.log(1 - p))


def boot_diff(df, la, lb, mask, label, quiet=False):
    """(LL_a - LL_b) per game within mask, draft-clustered bootstrap; returns estimate, ci, p, per-draft sums for reuse."""
    d = df[mask]
    ids = pd.factorize(d.draft)[0]
    n = ids.max() + 1
    ds = np.bincount(ids, weights=(la - lb)[mask], minlength=n)
    gs = np.bincount(ids, weights=d.games.values, minlength=n)
    est = ds.sum() / gs.sum()
    cnt = rng.multinomial(n, np.full(n, 1 / n), size=B)
    bs = (cnt @ ds) / (cnt @ gs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return est, lo, hi, p, d.draft.values, ds, gs


def fmt(r):
    return f"{r[0] * 1000:+.3f} [{r[1] * 1000:+.3f}, {r[2] * 1000:+.3f}] p {r[3]:.3f}"  # milli-nats per game


prem = DF[DF.fmt == "PremierDraft"].reset_index(drop=True)
P(f"Premier decks {len(prem)} in {prem.draft.nunique()} drafts, games {int(prem.games.sum())}; groups: " + ", ".join(f"{g} {int(f(prem).sum())}" for g, f in GROUPS.items()))
P(f"mean score G {prem.G.mean():.2f} (sd {prem.G.std():.2f}); M_v2 logit sd {prem.M_v2.std():.3f}; unit of differences: milli-nats of log loss per game (x1000)")
folds = {k: (prem.fold.values != k, prem.fold.values == k) for k in range(5)}
L = {}
for name, cols in {"skill": [], "G": ["G"], "Gall": ["Gall"], "Gfile": ["Gfile"], "M_v2": ["M_v2"], "M_v2t3": ["M_v2t3"], "M_v1": ["M_v1"], "G+M": ["G", "M_v2"]}.items():
    L[name] = loss(prem, crossfit(prem, cols, folds))
base = L["skill"].sum() / prem.games.sum()
P("\nheld-out log loss per game (cross-fitted, skill dummies + score): " + "; ".join(f"{k} {v.sum() / prem.games.sum():.5f}" for k, v in L.items()))
P("pseudo-R2 vs skill-only (%): " + "; ".join(f"{k} {100 * (1 - v.sum() / L['skill'].sum()):.3f}" for k, v in L.items() if k != "skill"))

P("\n=== PRIMARY: LL(G) - LL(M_v2), milli-nats per game [95% CI], two-sided p (positive = deck model better)")
res = {}
for g, f in GROUPS.items():
    res[g] = boot_diff(prem, L["G"], L["M_v2"], f(prem), g)
    P(f"  {g:5s} {fmt(res[g])}")
# good minus weak, same resamples as a joint bootstrap
n_all = prem.draft.nunique()
ids = pd.factorize(prem.draft)[0]
def per_draft(mask):
    return (np.bincount(ids, weights=((L['G'] - L['M_v2']) * mask), minlength=n_all), np.bincount(ids, weights=prem.games.values * mask, minlength=n_all))
dg, gg = per_draft(GROUPS["good"](prem)); dw, gw = per_draft(GROUPS["weak"](prem))
cnt = rng.multinomial(n_all, np.full(n_all, 1 / n_all), size=B)
est = dg.sum() / gg.sum() - dw.sum() / gw.sum()
bs = (cnt @ dg) / (cnt @ gg) - (cnt @ dw) / (cnt @ gw)
P(f"  good minus weak {est * 1000:+.3f} [{np.percentile(bs, 2.5) * 1000:+.3f}, {np.percentile(bs, 97.5) * 1000:+.3f}] p {min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean())):.3f}")
for h in (0, 1):
    m = (prem.fold.values % 2 == h)
    P(f"  halves (fold parity {h}) {fmt(boot_diff(prem, L['G'], L['M_v2'], m, 'h'))}")

P("\n--- secondary (not adjusted for multiplicity), all players then good / weak")
for lab, a, b in (("G vs M_v2t3", "G", "M_v2t3"), ("G vs M_v1 (stored specification)", "G", "M_v1"), ("Gall (All Decks GIH) vs M_v2", "Gall", "M_v2"), ("Gfile (ratings file, leaky) vs M_v2", "Gfile", "M_v2"),
                  ("M_v1 minus M_v2 (positive = v2 better)", "M_v1", "M_v2"), ("G vs skill only (positive = G beats no score)", "skill", "G"), ("M_v2 vs skill only", "skill", "M_v2"),
                  ("M_v2 given G: LL(G) - LL(G+M)", "G", "G+M"), ("G given M_v2: LL(M_v2) - LL(G+M)", "M_v2", "G+M")):
    P(f"  {lab:46s} " + " | ".join(f"{g} {fmt(boot_diff(prem, L[a], L[b], GROUPS[g](prem), g))}" for g in ("all", "good", "weak")))
other = DF[DF.fmt != "PremierDraft"].reset_index(drop=True)
if len(other):
    both = pd.concat([prem.assign(_t=0), other.assign(_t=1)], ignore_index=True)
    tr, te = (both._t.values == 0), (both._t.values == 1)
    Lo = {n: loss(both, crossfit(both, cols, {0: (tr, te)}))[te] for n, cols in {"G": ["G"], "M_v2": ["M_v2"]}.items()}
    o = both[te]
    P(f"  non-Premier decks {len(o)} (calibration on Premier): " + " | ".join(f"{g} {fmt(boot_diff(o, Lo['G'], Lo['M_v2'], GROUPS[g](o), g))}" for g in ("all", "good", "weak")))
(REPO / "analysis" / "out" / "deckmodel_v2_stepb.txt").write_text("\n".join(lines) + "\n")
