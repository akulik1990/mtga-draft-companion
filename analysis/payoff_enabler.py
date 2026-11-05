"""Pre-registered payoff/enabler analysis, FRA. Method fixed in docs/payoff-enabler-FRA.md. Output: analysis/out/payoff_enabler_FRA.json + stdout."""
import gzip, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.special import expit, logit
from scipy.stats import norm

R = Path(__file__).resolve().parents[1]
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
THEMES = json.loads((R / "companion/advisor_v2/sets/FRA/manual.json").read_text())["themes"]
TESTED = [t for t, v in THEMES.items() if v.get("payoff") or v.get("payoff_names") or v.get("payoff_min_cmc")]
ORDER = ["jace", "graveyard"] + [t for t in TESTED if t not in ("jace", "graveyard")]
ALPHA = 0.05 / len(TESTED)
MIN_DECKS, MIN_PAIR, BASE = 150, 300, 0.55

front = lambda n: n.split(" // ")[0]
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    n = front(c["name"])
    if n not in cards or (c.get("isprimarycard") and c.get("oracle_text") and not cards[n].get("oracle_text")):
        cards[n] = c

def roles(card, th):  # mirrors companion/advisor_v2/synergy.py roles(): reminder text stripped, lower-cased
    text = re.sub(r"\([^)]*\)", "", (card.get("oracle_text") or "").lower())
    ty = card.get("types") or []
    e = bool(th.get("enabler") and re.search(th["enabler"], text)) or any(t in ty for t in th.get("enabler_types", ()))
    p = bool(th.get("payoff") and re.search(th["payoff"], text)) or card["name"] in th.get("payoff_names", ()) or front(card["name"]) in th.get("payoff_names", ())
    if th.get("payoff_min_cmc"):
        p = p or ((card.get("cmc") or 0) >= th["payoff_min_cmc"] and all(t in ty for t in th.get("payoff_types", ())))
    return e, p

def gih(card, lane):
    d = card.get("deck_colors", {})
    for k in (lane, "All Decks"):
        v = (d.get(k) or {}).get("gihwr")
        if v: return v
    return np.nan

def mana_src(card):
    t = (card.get("oracle_text") or "").lower()
    return any(s in t for s in ("add {", "search your library for a basic land", "landcycling"))

raw = json.loads(gzip.open(R / "analysis/out/decks_FRA.json.gz").read())
rows, unmatched = [], set()
for d in raw:
    if d["skill"] < 0.60 or not d["deck_games"] or d["deck_games"] < 1 or sum(d["counts"].values()) != 40: continue
    r = dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=str(d["games_bucket"]), w=d["deck_wins"], g=d["deck_games"], lands=0, ramp=0)
    q, ns, acc = 0.0, 0, {t: [0, 0, 0] for t in TESTED}  # P, E, P6
    for n, k in d["counts"].items():
        c = cards.get(front(n))
        if n in BASICS or (c and "Land" in (c.get("types") or [])): r["lands"] += k; continue
        if not c: unmatched.add(n); continue
        q += gih(c, d["lane"]) * k; ns += k
        r["ramp"] += k * mana_src(c)
        for t in TESTED:
            e, p = roles(c, THEMES[t])
            if p: acc[t][0] += k; acc[t][2] += k * ((c.get("cmc") or 0) >= 6)
            elif e: acc[t][1] += k
    r["q"] = q / max(ns, 1)
    for t in TESTED: r[f"P_{t}"], r[f"E_{t}"], r[f"X_{t}"] = acc[t]
    rows.append(r)
df = pd.DataFrame(rows).dropna(subset=["q"])
df["mana"] = df.lands + df.ramp
print("decks", len(df), "unmatched names", sorted(unmatched)[:10], file=sys.stderr)

def fit(d, terms):
    """binomial GLM wins/losses, cluster by draft; returns {term: (pts, lo, hi, p, n)} at BASE."""
    d = d.copy(); d["L"] = d.g - d.w
    f = "w + L ~ q + skill + C(gb)" + (" + C(lane)" if d.lane.nunique() > 1 else "") + " + " + " + ".join(terms)
    m = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
    out = {}
    for t in terms:
        b, se, p = m.params[t], m.bse[t], m.pvalues[t]
        pts = lambda x: 100 * (expit(logit(BASE) + x) - BASE)
        out[t] = dict(pts=pts(b), lo=pts(b - 1.96 * se), hi=pts(b + 1.96 * se), p=p, n=int(d[t].sum()))
    return out

def bucket(x, cuts): return next(i for i, c in enumerate(cuts) if x <= c)
res = {}
for t in ORDER:
    th = THEMES[t]
    s = df[df.lane.isin(th["pairs"])]
    scope = "theme pairs " + "/".join(th["pairs"])
    if len(s) < MIN_PAIR: s, scope = df, "all lanes (pairs left %d decks)" % len(s)
    s = s.copy(); P, E = s[f"P_{t}"], s[f"E_{t}"]
    r = dict(scope=scope, n=len(s), n_P=P.value_counts().sort_index().to_dict())
    for k, lab in ((1, "P1"), (2, "P2"), (3, "P3p")):
        s[lab] = ((P == k) if k < 3 else (P >= 3)).astype(int)
    terms = [x for x in ("P1", "P2", "P3p") if s[x].sum() >= 30]
    s["E02"], s["E35"], s["E6"] = [((P >= 1) & m).astype(int) for m in (E <= 2, (E >= 3) & (E <= 5), E >= 6)]
    if len(s) >= MIN_DECKS and terms and (P == 0).sum() >= 30:
        r["P"] = fit(s, terms)
        # E buckets: P>=1 decks by E bucket, each vs P=0 (decision-rule contrast), same model
        et = [x for x in ("E02", "E35", "E6") if s[x].sum() >= 30]
        r["E_vs_none"] = fit(s, et)
    elif (P == 0).sum() < 30:  # EXPLORATORY, not pre-registered: no P=0 reference exists, so compare payoff-count bands with P<=3
        for lab, m in (("P4_6", (P >= 4) & (P <= 6)), ("P7_9", (P >= 7) & (P <= 9)), ("P10p", P >= 10)): s[lab] = m.astype(int)
        r["P_explore_vs_le3"] = fit(s, [x for x in ("P4_6", "P7_9", "P10p") if s[x].sum() >= 30])
    # within P>=1 only: E buckets vs 0-2
    w = s[P >= 1].copy()
    wt = [x for x in ("E35", "E6") if w[x].sum() >= 30 and w["E02"].sum() >= 30]
    w = w.drop(columns="E02")
    if wt and len(w) >= MIN_DECKS: r["E_within"] = fit(w, wt)
    # expensive payoffs
    x6 = s[f"X_{t}"] >= 1
    if x6.sum() >= 60:
        s["hi"] = (s.mana >= 18).astype(int)
        s["EXlo"] = (x6 & (s.mana <= 17)).astype(int); s["EXhi"] = (x6 & (s.mana >= 18)).astype(int)
        s["Pcheap"] = ((P >= 1) & ~x6).astype(int)
        xt = [x for x in ("EXlo", "EXhi", "Pcheap") if s[x].sum() >= 30]
        try: r["expensive"] = fit(s, xt + ["hi"]) ; r["expensive"].pop("hi", None)
        except Exception as ex: r["expensive_err"] = str(ex)
    res[t] = r
    print(t, scope, len(s), file=sys.stderr)

def line(k, v): return f"| {k} | {v['pts']:+.2f} | {v['lo']:+.2f} to {v['hi']:+.2f} | {v['p']:.3g}{' *' if v['p'] < ALPHA else ''} | {v['n']} |"
print(f"decks {len(df)}; Bonferroni alpha {ALPHA:.4f} over {len(TESTED)} themes")
for t in ORDER:
    r = res[t]; print(f"\n### {t} ({r['scope']}, n={r['n']}, P counts {r['n_P']})")
    for key in ("P", "P_explore_vs_le3", "E_vs_none", "E_within", "expensive"):
        if key in r:
            print(f"\n{key}\n| term | pts | 95% CI | p | n |\n|---|---|---|---|---|")
            for k, v in r[key].items(): print(line(k, v))
