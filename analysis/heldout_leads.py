"""Held-out confirmation of three post-hoc FRA leads (docs/heldout-leads-FRA.md). Output analysis/out/heldout_leads.txt.
  python3 analysis/heldout_leads.py build   # streams the Trad / Pick-Two (and, with 'premier', the Premier) game CSV into per-deck rows -> analysis/out/heldout_decks_<fmt>_FRA.json.gz
  python3 analysis/heldout_leads.py run     # estimates
Run as: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 ...

PRE-REGISTRATION (written before any held-out estimate was computed; repeated in the doc).
Specs are those of the Premier findings, re-implemented on the deck tables (replication of the Premier numbers on decks_FRA.json.gz is printed first as a check):
 L1 two-drops: foundations_deck.py R2 / skill_groups_deck.py 3c: decks of every lane, cmc-2 creature buckets (0-2, 3, 4, 6+ vs 5), controls quality (a) = mean All-Decks GIH + top-10 mean (spec Ca, PRIMARY)
    or quality outside the two-drops (Cb, robustness), splash, skill, lane FE; binomial GLM on wins/games, freq weights games, SE clustered by draft, NO game-count term. Target: 6+ in good players, and good minus weak.
 L2 lands: podcast_lands18.py build(): two-colour lane, 40 cards, 16/17/18 lands; 16 vs 17 among decks with 2+ cantrips (cmc <= 2 non-land, oracle text 'draw a card|scry|surveil'), 18-land decks excluded;
    y ~ x + quality + skill + C(lane). Target: all skill groups pooled (primary); groups reported.
 L3 slowness: b3b8_speed_tapped.py B3 P1: two-colour lane, mean cmc of non-land cards within lane centred and scaled by the sample's within-lane SD; y ~ mcw + mcw^2 + q34 (mean GIH of cmc 3-4 spells) + skill + splash + C(lane);
    target = coefficient of mcw (pts per +1 SD).
 Groups by the player's skill bucket: good >= 0.60, average 0.52-0.58, weak < 0.52. Effects in WR points = 24.75 x logit coefficient.
 Confirmation rule: same sign as the Premier estimate and the 95% CI excludes 0 (two-sided; no one-sided reading). 'Confirmatory sample':
   L1, L2: Trad + Pick-Two decks of players with fewer than 100 games (user_n_games_bucket < 100), because decks_FRA.json.gz (the Premier sample) holds the Premier, Trad AND Pick-Two decks of 100+ games players, so the
           100+ Trad/Pick-Two decks were already inside the original L1 and L2 estimates (L3 was Premier-only). The <100 decks are not in the original sample.
   L3:     Trad + Pick-Two decks of 100+ games players (same population, other formats, none in the original); <100 decks reported as secondary.
 Secondary (reported, never the verdict): the other of the two sample definitions, Premier-only re-estimates, pooled Trad+Pick-Two (format FE).
"""
import csv, gzip, json, os, re, sys, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"; SYN = f"{R}/data/Temp/synergy"
K = 24.75

def load_by():
    by = {}
    for v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
        n = v["name"].split(" // ")[0]
        if n in by and not v.get("isprimarycard", 1): continue
        by[n] = v
    return by

# ================================================================ build
def build(fmt):
    out = {}
    with gzip.open(f"{SYN}/game_data_public.FRA.{fmt}.csv.gz", "rt", newline="") as f:
        r = csv.reader(f); h = next(r); ix = {k: i for i, k in enumerate(h)}
        cols = [(k[5:], i) for k, i in ix.items() if k.startswith("deck_")]
        for row in r:
            key = (row[ix["draft_id"]], row[ix["build_index"]])
            sk = row[ix["user_game_win_rate_bucket"]]
            d = out.get(key)
            if d is None:
                d = out[key] = dict(draft=key[0], fmt=fmt, lane=row[ix["main_colors"]], splash=row[ix["splash_colors"]], skill=float(sk) if sk not in ("", "None") else None,
                                    nb=int(float(row[ix["user_n_games_bucket"]] or 0)), g=0, w=0, counts={n: int(float(row[i])) for n, i in cols if row[i] not in ("", "0")})
            d["g"] += 1; d["w"] += row[ix["won"]] == "True"
    json.dump(list(out.values()), gzip.open(f"{OUT}/heldout_decks_{fmt}_FRA.json.gz", "wt")); print(fmt, len(out), "decks")

# ================================================================ features
by = load_by()
isl = lambda v: "Land" in v["types"]
gA = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
BASIC = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
byL = {}  # podcast_lands18 style card dict (last duplicate wins)
for _v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values(): byL[_v["name"].split(" // ")[0]] = _v
def is_cantrip(c):
    t = (c.get("oracle_text") or "").lower()
    return "Land" not in c["types"] and c["cmc"] <= 2 and bool(re.search(r"draw a card|scry|surveil", t))

def features(d):
    """one row per deck: L1 (two-drops), L2 (lands/cantrips), L3 (speed) fields; counts keys are card names as in the game CSV columns."""
    its, lands, n40, ct = [], 0, 0, 0
    qL = []
    for name, k in d["counts"].items():
        n = name.split(" // ")[0]; n40 += k
        c = byL.get(n)
        if name in BASIC: lands += k; continue
        if c is None: continue
        if "Land" in c["types"]: lands += k; continue
        if is_cantrip(c): ct += k
        gg = gA(c)
        if gg: qL += [gg] * k
        v = by.get(n)
        if v is not None and not isl(v): its.append((n, k, v["cmc"] or 0, gA(v), "Creature" in v["types"]))
    r = dict(draft=d["draft"], fmt=d["fmt"], lane=d["lane"], nlane=len(d["lane"]), splash=int(d["splash"] != ""), skill=d["skill"], nb=d.get("nb", 100), w=d["w"], g=d["g"],
             y=d["w"] / d["g"], n40=n40, lands=lands, ct=ct, qland=np.mean(qL) if qL else np.nan)
    if its:
        k = np.array([x[1] for x in its], float); c = np.array([x[2] for x in its], float); g = np.array([x[3] for x in its], float); cre = np.array([x[4] for x in its])
        def qm(m): s = m & (g > 0); return np.average(g[s], weights=k[s]) if s.any() else np.nan
        def t10(m):
            s = np.repeat(g[m & (g > 0)], k[m & (g > 0)].astype(int)); return np.sort(s)[-10:].mean() if len(s) else np.nan
        al = np.ones(len(k), bool); two = cre & (c == 2)
        r.update(two=k[two].sum(), qA=qm(al), t10A=t10(al), qA_x2=qm(~two), t10A_x2=t10(~two), mc=(k * c).sum() / k.sum(), q34=qm((c >= 3) & (c <= 4)))
    return r

# ================================================================ estimation
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
def fit(df, f):
    return smf.glm("y ~ " + f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})
def est(m, w):
    nm = list(m.params.index); v = np.zeros(len(nm))
    for t, c in w.items(): v[nm.index(t)] = c
    b = float(v @ m.params.values); se = float(np.sqrt(v @ m.cov_params().values @ v)); return (K * b, K * se)
def safe(df, f, w, minn=0):
    try:
        if len(df) < max(minn, 30): return None
        return est(fit(df, f), w)
    except Exception as e: print("FAIL", repr(e)[:90], flush=True); return None
GR = ["all", "good", "avg", "weak"]
def grp(df, k):
    s = df.skill
    return {"all": df.skill.notna(), "good": s >= .6 - 1e-9, "avg": (s >= .52 - 1e-9) & (s <= .58 + 1e-9), "weak": s < .52 - 1e-9}[k].pipe(lambda m: df[m])
def fe(e, star=True):
    if e is None: return "n/a"
    lo, hi = e[0] - 1.96 * e[1], e[0] + 1.96 * e[1]; return f"{e[0]:+.2f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if star and (lo > 0 or hi < 0) else "")
def diff(a, b):
    if a is None or b is None: return None
    return (a[0] - b[0], float(np.hypot(a[1], b[1])))

def prep3(d):
    d = d.copy(); dev = d.mc - d.groupby("lane").mc.transform("mean"); d["mcw"] = dev / dev.std(); d["mcw2"] = d.mcw ** 2; return d, float(dev.std())

def l1(df, spec, fmtfe=False):
    d = df[df.qA.notna() & df.t10A.notna() & df.skill.notna()].copy()
    d["b02"], d["b3"], d["b4"], d["b6p"] = (d.two <= 2).astype(int), (d.two == 3).astype(int), (d.two == 4).astype(int), (d.two >= 6).astype(int)
    Q = "qA + t10A" if spec == "Ca" else "qA_x2 + t10A_x2"
    if spec == "Cb": d = d[d.qA_x2.notna() & d.t10A_x2.notna()]
    out = {}
    for k in GR:
        s = grp(d, k); out[k] = safe(s, f"b02 + b3 + b4 + b6p + {Q} + splash + skill + C(lane)" + (" + C(fmt)" if fmtfe else ""), {"b6p": 1}); out[k + "_n"] = (len(s), int(s.b6p.sum()), int((s.two == 5).sum()))
    return out
def l2(df, fmtfe=False, extra=''):
    d = df[(df.nlane == 2) & (df.n40 == 40) & df.lands.isin([16, 17, 18]) & df.qland.notna() & df.skill.notna() & (df.ct >= 2) & (df.lands != 18)].copy()
    d["x"] = (d.lands == 16).astype(int); out = {}
    for k in GR:
        s = grp(d, k); nx, nr = int(s.x.sum()), int((s.lands == 17).sum())
        out[k] = safe(s, "x + qland + skill + C(lane)" + extra + (" + C(fmt)" if fmtfe else ""), {"x": 1}) if nx >= 20 and nr >= 20 else None; out[k + "_n"] = (len(s), nx, nr)
    return out
def l3(df, fmtfe=False):
    d = df[(df.nlane == 2) & df.qA.notna() & df.t10A.notna() & df.q34.notna() & df.skill.notna()].copy(); d, sd = prep3(d); out = {}
    for k in GR:
        s = grp(d, k); out[k] = safe(s, "mcw + mcw2 + q34 + skill + splash + C(lane)" + (" + C(fmt)" if fmtfe else ""), {"mcw": 1}); out[k + "_n"] = (len(s), round(sd, 3))
    return out

LINES = []
def P(s=""): print(s, flush=True); LINES.append(s)
def show(title, res, nfmt):
    P(f"\n{title}")
    P("| group | n decks (detail) | estimate [95% CI] |\n|---|---|---|")
    for k in GR: P(f"| {k} | {res[k+'_n']} | {fe(res[k])} |")
    P(f"| good minus weak | | {fe(diff(res['good'], res['weak']))} |")

def run():
    P("# Held-out confirmation of three leads (FRA)")
    P(f"\nBuilt from the Trad / Pick-Two game CSV; see header of analysis/heldout_leads.py for the pre-registration.")
    ORI = {"L1": "good 6+ vs 5: Premier -2.06 [-3.43, -0.69], good minus weak -2.94 (Ca)", "L2": "16 vs 17 lands, 2+ cantrips: Premier +0.96 [+0.13, +1.79] (all)", "L3": "slowness premium Q(b): Premier +0.80 [+0.47, +1.14]"}
    # --- replication of the originals on decks_FRA.json.gz
    O = []
    for d in json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt")):
        if d["deck_games"] < 1: continue
        O.append(features(dict(draft=d["draft"], fmt=d["format"], lane=d["lane"], splash=d["splash"], skill=d["skill"], nb=100, w=d["deck_wins"], g=d["deck_games"], counts=d["counts"])))
    O = pd.DataFrame(O); O["tag"] = "orig"
    P(f"\n## 0. Replication on decks_FRA.json.gz ({len(O)} decks: " + ", ".join(f"{a} {b}" for a, b in O.fmt.value_counts().items()) + ")")
    P("Originals: " + "; ".join(f"{k}: {v}" for k, v in ORI.items()))
    show("L1 Ca, all formats in decks_FRA (original sample)", l1(O, "Ca"), 0)
    show("L2 all formats in decks_FRA (original sample)", l2(O), 0)
    show("L3 Premier only in decks_FRA (original sample)", l3(O[O.fmt == "PremierDraft"]), 0)
    P("\n## 0b. Premier-only and Trad+PT(100+) splits of the original sample (format splits; the cells were selected in the pooled sample, so these are NOT held out for L1, L2)")
    for nm, s in (("Premier only", O[O.fmt == "PremierDraft"]), ("Trad + PT, 100+ players", O[O.fmt != "PremierDraft"])):
        show(f"L1 Ca, {nm}", l1(s, "Ca", nm != "Premier only" and False), 0); show(f"L2, {nm}", l2(s), 0)
    # --- held-out tables
    H = []
    for fmt in ("TradDraft", "PickTwoDraft"):
        for d in json.load(gzip.open(f"{OUT}/heldout_decks_{fmt}_FRA.json.gz", "rt")): H.append(features(d))
    H = pd.DataFrame(H); H = H[H.g >= 1]
    big, small = H[H.nb >= 100], H[H.nb < 100]
    P(f"\n## 1. Held-out decks: Trad+PT {len(H)}; 100+ players {len(big)} (Trad {int((big.fmt=='TradDraft').sum())}, PT {int((big.fmt=='PickTwoDraft').sum())}); <100 players {len(small)} (Trad {int((small.fmt=='TradDraft').sum())}, PT {int((small.fmt=='PickTwoDraft').sum())}). Skill missing: {int(H.skill.isna().sum())}")
    P("Skill group sizes (decks) <100: " + ", ".join(f"{k} {len(grp(small, k))}" for k in GR[1:]) + "; 100+: " + ", ".join(f"{k} {len(grp(big, k))}" for k in GR[1:]))
    for lead, fn in (("L1", lambda s, f=False: l1(s, "Ca", f)), ("L1 Cb", lambda s, f=False: l1(s, "Cb", f)), ("L2", l2), ("L3", l3)):
        P(f"\n## {lead}: {ORI[lead.split()[0]]}")
        for nm, s, role in (("<100-games players, Trad+PT", small, "CONFIRMATORY for L1, L2; secondary for L3"), ("100+ players, Trad+PT", big, "CONFIRMATORY for L3; contaminated for L1, L2 (inside the original sample)")):
            show(f"{nm} [{role}]", fn(s), 0)
        show("pooled Trad+PT, all players (format FE) [secondary]", fn(H, True), 0)
        if lead.startswith("L1") or lead == "L2":
            for fm in ("TradDraft", "PickTwoDraft"): show(f"{fm}, <100 players", fn(small[small.fmt == fm]), 0)
    P("\n## L2 sensitivity (post hoc, labelled): add mean cmc of the spells as a control for curve (a 16-land deck is usually a lower-curve deck)")
    Hm = H[H.mc.notna()]; Om = O[O.mc.notna()]
    for nm, s in (("Premier-only (original sample)", Om[Om.fmt == "PremierDraft"]), ("all formats in decks_FRA (original sample)", Om), ("<100-games Trad+PT", Hm[Hm.nb < 100]), ("100+ Trad+PT", Hm[Hm.nb >= 100]), ("pooled Trad+PT (format FE)", Hm)):
        show(f"L2 + mc control, {nm}", l2(s, nm.startswith("pooled"), " + mc"), 0)
    open(f"{OUT}/heldout_leads.txt", "w").write("\n".join(LINES))

if __name__ == "__main__":
    if sys.argv[1] == "build":
        for fmt in sys.argv[2:] or ("TradDraft", "PickTwoDraft"): build(fmt)
    else: run()
