"""B4: do decks with more non-board spells (draw-only, lifegain-only, aura/equipment, tricks) win less than their cards' ratings say?
Rule (fixed before the run): 60%+ players' two-colour decks, binomial GLM on game wins, SEs clustered by draft. Controls: mean lane GIH WR of
spells, creature count, removal count, splash flag, lane, skill, games bucket. Primary: total non-board count. Secondary: the four classes
separately, and trick x removal count (removal centred). Effects in win-rate points per spell at 55%.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/podcast_nonboard.py"""
import gzip, json, os, re, sys
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from companion.advisor_v2 import config
K = 100 * 0.55 * 0.45
raw = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
roles = {r["name"]: set(r["tags"]) for r in json.load(open(f"{R}/companion/advisor_v2/sets/FRA/roles.json"))}
by, cls, lists = {}, {}, {k: [] for k in ("draw", "life", "aura", "trick")}
for v in raw.values():
    n = v["name"].split(" // ")[0]
    if n in by and not v.get("isprimarycard", 1):
        continue
    by[n] = v
for n, v in by.items():
    t, ty = set(config.tags(v, "FRA")), v["types"]
    txt = re.sub(r"\([^)]*\)", "", v.get("oracle_text") or "").lower()
    ro = roles.get(n, set())
    if "Land" in ty or "Creature" in ty or "Planeswalker" in ty:
        continue
    if "removal" in t or "removal" in ro:
        cls[n] = "removal"; continue
    if "Aura" in v["subtypes"] or "Equipment" in v["subtypes"]:
        cls[n] = "aura"
    elif "combat_trick" in t and "Instant" in ty:
        cls[n] = "trick"
    elif "empower jace" in txt:
        continue  # makes a planeswalker token: board presence
    elif ("card_advantage" in t or "draw" in ro) and "fixing_ramp" not in t and not re.search(r"return[^.]*to the battlefield", txt) \
            and re.search(r"\bdraws?\b|surveil|return target|cards you own", txt):
        cls[n] = "draw"
    elif ("lifegain" in t or "lifegain_enabler" in ro) and "gain" in txt and "life" in txt:
        cls[n] = "life"
    else:
        continue
    lists[cls[n]].append(f'{n} ({v["cmc"]}; {sorted(t)}; {txt[:70]!r})')
for k, l in lists.items():
    print(f"\n## {k}: {len(l)} cards"); [print(" -", x) for x in sorted(l)]
rem = {n for n, c in cls.items() if c == "removal"} | {n for n, v in by.items() if "Land" not in v["types"] and ("removal" in config.tags(v, "FRA") or "removal" in roles.get(n, ()))}

rows = []
for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")):
    if d["skill"] < 0.60 or len(d["lane"]) != 2 or d["deck_games"] < 1:
        continue
    q, c = [], dict(draw=0, life=0, aura=0, trick=0, cre=0, rem=0)
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; v = by.get(n)
        if not v or "Land" in v["types"]:
            continue
        g = (v["deck_colors"].get(d["lane"]) or {}).get("gihwr") or (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
        if g: q += [g] * k
        c["cre"] += k * ("Creature" in v["types"]); c["rem"] += k * (n in rem)
        if cls.get(n) in c: c[cls[n]] += k
    if len(q) >= 20:
        rows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=d["games_bucket"], splash=int(d["splash"] != ""),
                         y=d["deck_wins"] / d["deck_games"], g=d["deck_games"], quality=np.mean(q), **c))
df = pd.DataFrame(rows); df["nb"] = df.draw + df.life + df.aura + df.trick; df["remc"] = df.rem - df.rem.mean()
print(f"\ndecks {len(df)}, drafts {df.draft.nunique()}, games {int(df.g.sum())}")
print(df[["draw", "life", "aura", "trick", "nb", "rem", "cre"]].describe().loc[["mean", "std", "max"]].round(2).to_string())
print("share of decks with >=1:", {k: round((df[k] > 0).mean(), 3) for k in ("draw", "life", "aura", "trick", "nb")})
G = pd.factorize(df.draft)[0]
def fit(f):
    return smf.glm(f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": G})
def show(m, names):
    for x in names:
        b, se, p = m.params[x], m.bse[x], m.pvalues[x]
        print(f"  {x:12s} {K*b:+.2f} pts/spell [{K*(b-1.96*se):+.2f}, {K*(b+1.96*se):+.2f}] p {p:.3g}")
ctrl = "quality + cre + rem + splash + skill + C(gb) + C(lane)"
print("\nPRIMARY total non-board:"); m = fit(f"y ~ nb + {ctrl}"); show(m, ["nb"])
print("\nclasses together:"); m2 = fit(f"y ~ draw + life + aura + trick + {ctrl}"); show(m2, ["draw", "life", "aura", "trick"])
print("\ntrick x removal (removal centred; main = trick effect at mean removal):")
m3 = fit(f"y ~ trick*remc + draw + life + aura + quality + cre + splash + skill + C(gb) + C(lane)"); show(m3, ["trick", "remc", "trick:remc"])
print("\nwithout quality control (descriptive; ratings already hold some of the effect):"); m4 = fit("y ~ nb + cre + rem + splash + skill + C(gb) + C(lane)"); show(m4, ["nb"])
print("\nrobustness: >=1 non-board vs 0 :"); df["any_nb"] = (df.nb > 0).astype(int); m5 = fit(f"y ~ any_nb + {ctrl}"); print(f"  any_nb {100*0.2475*m5.params['any_nb']:+.2f} p {m5.pvalues['any_nb']:.3g}")
