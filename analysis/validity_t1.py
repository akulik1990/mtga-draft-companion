"""Validity check of the pick-level test (declared before running): short pool-decks (n < 23 castable spells in the best pair, 52% of drafts)
could drive the removal/finisher coefficients. Refit Model 1 (the memo's) with n as a control, and on drafts with n == 23 only."""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
d = pd.read_csv("analysis/out/draft_features_FRA.csv.gz")
d = d[d.trials > 0].copy()
d["wins"] = d.event_match_wins; d["losses"] = d.event_match_losses
d["removal"] = d.removal.clip(3, 10); d["finishers"] = d.finishers.clip(0, 4); d["duals"] = d.duals.clip(0, 6)
print("drafts", len(d), "n==23:", int((d.n >= 23).sum()), "share", round((d.n >= 23).mean(), 3), "| mean n", round(d.n.mean(), 1))
print("corr(n, removal) %.2f  corr(n, finishers) %.2f  corr(n, quality) %.2f  corr(n, skill) %.2f" % tuple(d.n.corr(d[c]) for c in ("removal", "finishers", "quality", "user_game_win_rate_bucket")))
def run(df, label, extra=""):
    X = smf.glm(f"wins ~ quality + C(pair) + C(user_n_games_bucket) + C(rank) + C(user_game_win_rate_bucket) + removal + finishers + duals{extra}", data=df.assign(wins=df.wins / df.trials), family=sm.families.Binomial(), freq_weights=df.trials).fit(cov_type="HC1")
    out = []
    for k in ("removal", "finishers", "duals"):
        b, lo, hi = X.params[k], *X.conf_int().loc[k]
        out.append(f"{k} {100*0.55*0.45*b:+.2f} [{100*0.55*0.45*lo:+.2f}, {100*0.55*0.45*hi:+.2f}]")
    print(f"{label:48} n={len(df):6d}  " + "   ".join(out))
run(d, "Model 1 as run by the Analyst (all drafts)")
run(d, "+ n (castable spells in pool-deck) as control", " + n")
full = d[d.n >= 23]
run(full, "n == 23 only (full pool-deck)")
run(full[full.user_n_games_bucket.astype(str).isin(["100", "500"])], "n == 23 and 100+ games players")
