"""Are finishers special, or is it 'has strong cards'? Pre-registered (written before the run):

Pool-deck as in fra_step2.py (best pair, top 23 castable spells by GIH WR). Counts in it: finishers (cmc >= 5, GIH WR >= set mean + 1 SD),
cheap_top (cmc <= 4, GIH WR >= set mean + 1 SD), removal. Outcome: event match wins out of wins + losses, binomial GLM, HC1.
Model: quality + best pair + games bucket + rank + skill bucket + removal + finishers + cheap_top (Model 1 style). Primary sample: pool-decks with all 23 cards.
RULE: 'finishers are specific' if the finisher coefficient exceeds the cheap_top coefficient and the 95% CI of the difference excludes 0.
'It is strong cards' if cheap_top >= 0.75 x finishers or the difference CI includes 0. In between or either: no further runs.
Also reported: the vectorised selection reproduces draft_features_FRA.csv.gz (quality, removal, finishers) so numbers are comparable.
"""
import itertools, warnings
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
d = pd.read_csv("analysis/out/final_pools_FRA.csv.gz")
cards = pd.read_csv("data/tables/FRA/2_cards.csv").drop_duplicates("name").set_index("name")
pc = [c for c in d.columns if c.startswith("pool_")]
names = [c[5:] for c in pc if c[5:] in cards.index]
c = cards.loc[names]
P = d[["pool_" + n for n in names]].values.astype(float)
g = c.gihwr.fillna(0).values.astype(float)
nl = (c.land == 0).values
has = nl & (g > 0)
mu, sd = g[has].mean(), g[has].std()
gf = np.where(g > 0, g, mu - sd)
cf = np.stack([c["col_" + x].values for x in "WUBRG"], 1).astype(int)
cmc = c.cmc.values
rem = c.role_removal.fillna(0).values > 0
top = g >= mu + sd
fin = (cmc >= 5) & top
cheap = (cmc <= 4) & top & nl
order = np.argsort(-gf)
best = np.full(len(d), -1.0); stat = np.zeros((len(d), 5)); pair = np.zeros(len(d), int)
for pi, (a, b) in enumerate(itertools.combinations(range(5), 2)):
    mask = np.ones(5, bool); mask[[a, b]] = False
    cast = nl & (cf[:, mask].sum(1) == 0)
    Pc = P[:, order] * cast[order]
    cum = np.cumsum(Pc, 1)
    take = np.minimum(Pc, np.maximum(0, 23 - (cum - Pc)))
    n = take.sum(1)
    q = (take * gf[order]).sum(1) / np.maximum(n, 1)
    q = np.where(n > 0, q, -1)
    upd = q > best
    best = np.where(upd, q, best); pair = np.where(upd, pi, pair)
    s = np.stack([n, (take * rem[order]).sum(1), (take * fin[order]).sum(1), (take * cheap[order]).sum(1), q], 1)
    stat = np.where(upd[:, None], s, stat)
f = pd.DataFrame(stat, columns=["n", "removal", "finishers", "cheap_top", "quality"])
f["pair"] = pair
f = pd.concat([d[["draft_id", "rank", "event_match_wins", "event_match_losses", "user_n_games_bucket", "user_game_win_rate_bucket"]], f], axis=1)
f["trials"] = f.event_match_wins + f.event_match_losses
f = f[f.trials > 0].copy()
ref = pd.read_csv("analysis/out/draft_features_FRA.csv.gz").set_index("draft_id")
j = f.set_index("draft_id").join(ref[["quality", "removal", "finishers"]], rsuffix="_ref", how="inner")
print("reproduction vs Analyst features: n", len(j), "max |dq|", float((j.quality - j.quality_ref).abs().max()), "removal equal", float((j.removal == j.removal_ref).mean()), "finishers equal", float((j.finishers == j.finishers_ref).mean()))
f["rem"] = f.removal.clip(3, 10); f["fin"] = f.finishers.clip(0, 4); f["cheap"] = f.cheap_top.clip(0, 8)
f["y"] = f.event_match_wins / f.trials
k = 0.55 * 0.45 * 100
print("means: finishers %.2f cheap_top %.2f removal %.2f; corr(fin, cheap) %.2f" % (f.finishers.mean(), f.cheap_top.mean(), f.removal.mean(), f.finishers.corr(f.cheap_top)))
def run(df, label, extra=""):
    m = smf.glm("y ~ quality + C(pair) + C(user_n_games_bucket) + C(rank) + C(user_game_win_rate_bucket) + rem + fin + cheap" + extra, df, family=sm.families.Binomial(), freq_weights=df.trials).fit(cov_type="HC1")
    ci = m.conf_int()
    out = [f"{v} {k * m.params[v]:+.2f} [{k * ci.loc[v, 0]:+.2f},{k * ci.loc[v, 1]:+.2f}]" for v in ("rem", "fin", "cheap")]
    cv = m.cov_params()
    diff = m.params["fin"] - m.params["cheap"]; se = np.sqrt(cv.loc["fin", "fin"] + cv.loc["cheap", "cheap"] - 2 * cv.loc["fin", "cheap"])
    print(f"{label:34} n={len(df):6d}  " + "   ".join(out) + f"   fin - cheap {k * diff:+.2f} [{k * (diff - 1.96 * se):+.2f},{k * (diff + 1.96 * se):+.2f}]")
full = f[f.n >= 23]
run(full, "n == 23 (PRIMARY)")
run(full[full.user_n_games_bucket >= 100], "n == 23, 100+ games players")
run(f, "all drafts, n as control", " + n")
