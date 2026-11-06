"""Foundations check, deck-level effects behind builder rules (FRA). Question: does the deck-level evidence for each shipped builder rule survive a
sample that does not select on the outcome, and does it survive a quality control that does not contain the tested cards' own ratings?
Writes analysis/out/foundations_deck.txt and the section "Deck-level effects behind builder rules" of docs/foundations-check-FRA.md.

Specs (fixed before any leak-free outcome was looked at; the only addition after the diagnostics in section 0 is the C columns, see below):
  O    original spec and sample of the named script (skill >= 0.60 and C(gb) where that script had them).
  Ia   the sample asked for in the test plan: decks with games bucket 7-9, all skills, original controls (skill and C(gb) kept, as review_deckglm.py).
  Ib   the same with quality (b): GIH of the spells outside the tested class.
  Is60 Ia restricted to skill >= 0.60 (sensitivity).
  Ca   corrected sample: ALL decks of the rule's population, skill kept, NO games-bucket term. Reason in section 0: in decks_FRA.json.gz `games_bucket` is the
       DECK's own game count (not the 17Lands user_n_games_bucket) and every deck belongs to a 100+ games player, so C(gb) and the 7-9 cut condition on the outcome
       through the 7-win / 3-loss stop, while skill is nearly clean for 100+ players.
  Cb   Ca with quality (b).   Cns Ca without the skill term.   CA / CB  Ca on the two halves of the drafts (crc32(draft) parity).
Quality (a) = mean GIH WR of all spells (family of the original: lane GIH, else All Decks, for the video/podcast removal family; All Decks mean + top-10 for the
two-drop / top-end / empower family). Quality (b) = the same without the tested class. Effects: WR points at 55% (24.75 x logit coefficient), 95% CI from SEs clustered by draft.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/foundations_deck.py"""
import contextlib, gzip, io, json, os, re, runpy, sys, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{R}/analysis/out"; DOC = f"{R}/docs/foundations-check-FRA.md"
sys.path.insert(0, R); sys.path.insert(0, f"{R}/analysis")
import video_counters as vc, video_sinks as vs
K = 100 * .55 * .45
SCRATCH = os.environ.get("FOUND_VERDICTS", "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad/verdicts.json")
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
SECTION = {}  # rule id -> markdown block (tables), for the doc

def quiet(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(path)

# ------------------------------------------------------------------ cards and class sets
by = vc.cards()
gA = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
gL = lambda v, lane: (v["deck_colors"].get(lane) or {}).get("gihwr") or gA(v)
isl = lambda v: "Land" in v["types"]
RC = set(json.load(open(f"{OUT}/removal_classes_FRA.json")))
CORE = set(vc.CORE)
SINK = set(vs.tag(by))
ns = quiet(f"{R}/analysis/podcast_nonboard.py")
NB = {n for n, c in ns["cls"].items() if c in ("draw", "life", "aura", "trick")}
REMNB = set(ns["rem"])
EL = json.load(open(f"{OUT}/video_empower_lists.json")); WAYS, EMP = set(EL["ways"]), set(EL["empower"])
nl = {n: v for n, v in by.items() if not isl(v)}
CRE = {n for n, v in nl.items() if "Creature" in v["types"]}
TWO = {n for n in CRE if nl[n]["cmc"] == 2}
BIG = {n for n, v in nl.items() if (v["cmc"] or 0) >= 5}
C7 = {n for n, v in nl.items() if (v["cmc"] or 0) >= 7}
pr = np.array([gA(v) for v in nl.values() if gA(v) > 0]); MU, SD = pr.mean(), pr.std()
FIN = {n for n in BIG if gA(nl[n]) > 0 and gA(nl[n]) >= MU + SD}
CYC = {n for n, v in nl.items() if re.search(r"landcycling", v.get("oracle_text") or "", re.I)}
CLASSES = dict(rem=RC, ctr=CORE, remctr=RC | CORE, sink=SINK, nb=NB, cre=CRE, two=TWO, big=BIG, c7=C7, fin=FIN, wemp=WAYS | EMP, cyc=CYC)
fitted = json.load(open(f"{R}/companion/advisor_v2/sets/FRA/fitted.json"))["decks"]["pairs"]
def target(lane, key):
    p = fitted.get(lane) or fitted.get(lane[::-1])
    return p[key] if p else np.nan

# ------------------------------------------------------------------ deck table
DECKS = json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt"))
rows = []
for d in DECKS:
    lane = d["lane"]; items = []
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; v = by.get(n)
        if v and not isl(v): items.append((n, k, gL(v, lane), gA(v), v.get("colors") or []))
    names = [n for n, k, *_ in items for _ in range(k)]
    gl = np.array([x[2] for x in items for _ in range(x[1])], float); ga = np.array([x[3] for x in items for _ in range(x[1])], float)
    off = np.array([bool(x[4]) and not set(x[4]) <= set(lane) for x in items for _ in range(x[1])])
    r = dict(draft=d["draft"], fmt=d["format"], lane=lane, nlane=len(lane), splash_c=d["splash"], splash=int(d["splash"] != ""), skill=d["skill"], gb=d["games_bucket"],
             w=d["deck_wins"], g=d["deck_games"], y=d["deck_wins"] / d["deck_games"], nsp=len(names), npl=int((gl > 0).sum()))
    def qm(a, mask): s = a[(a > 0) & mask]; return s.mean() if len(s) else np.nan
    def t10(a, mask): s = np.sort(a[(a > 0) & mask])[-10:]; return s.mean() if len(s) else np.nan
    allm = np.ones(len(names), bool)
    r["qL"], r["qA"], r["t10A"] = qm(gl, allm), qm(ga, allm), t10(ga, allm)
    for c, S in CLASSES.items():
        m = np.array([n in S for n in names], bool)
        r[c] = int(m.sum()); r[f"qL_x_{c}"] = qm(gl, ~m); r[f"qA_x_{c}"] = qm(ga, ~m); r[f"t10A_x_{c}"] = t10(ga, ~m)
    r["qA_x_off"], r["t10A_x_off"] = qm(ga, ~off), t10(ga, ~off)
    r["remnb"] = sum(k for n, k, *_ in items if n in REMNB)
    rows.append(r)
DF = pd.DataFrame(rows)
DF["half"] = DF.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
DF["tr"] = [target(l, "removal") if len(l) == 2 else np.nan for l in DF.lane]
DF["tc"] = [target(l, "creatures") if len(l) == 2 else np.nan for l in DF.lane]
DF["tb"] = [target(l, "five_plus") if len(l) == 2 else np.nan for l in DF.lane]
DF["short6"] = (DF.rem < 6).astype(int)
DF["sfall"] = np.maximum(0, DF.tr - 2 - DF.rem); DF["sfd"] = (DF.sfall > 0).astype(int)
DF["csf"] = np.maximum(0, DF.tc - 2 - DF.cre); DF["bigx"] = np.maximum(0, DF.big - (DF.tb + 2))
for nm, lo, hi in (("b02", 0, 2),): DF[nm] = (DF.two <= hi).astype(int)
DF["b3"], DF["b4"], DF["b6p"] = (DF.two == 3).astype(int), (DF.two == 4).astype(int), (DF.two >= 6).astype(int)
DF["t01"], DF["t3"], DF["t4"], DF["t5p"] = (DF.big <= 1).astype(int), (DF.big == 3).astype(int), (DF.big == 4).astype(int), (DF.big >= 5).astype(int)
POP1 = DF[(DF.nlane == 2) & (DF.g >= 1) & (DF.npl >= 20) & DF.qL.notna()].copy()   # as video_counters.load / podcast_removal
POP2 = DF[DF.qA.notna() & DF.t10A.notna()].copy()                                    # as podcast_twodrop / topend: every deck
POPP = POP2[POP2.fmt == "PremierDraft"].copy()                                       # as podcast_splash_speed / video_empower

# ------------------------------------------------------------------ estimation helpers
def fit(df, f):
    G = pd.factorize(df.draft)[0]
    return smf.glm("y ~ " + f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": G})

def est(m, w, k=K):
    names = list(m.params.index)
    try:
        v = np.zeros(len(names))
        for t, c in w.items(): v[names.index(t)] = c
        b = float(v @ m.params.values); se = float(np.sqrt(v @ m.cov_params().values @ v))
        from scipy.stats import norm
        return k * b, k * (b - 1.96 * se), k * (b + 1.96 * se), 2 * norm.sf(abs(b / se))
    except Exception:
        return (np.nan,) * 4

SPECS = ["O", "Ia", "Ib", "Is60", "Ca", "Cb", "Cns", "CA", "CB"]
CTRL = dict(O="skill + C(gb) + C(lane)", Ia="skill + C(gb) + C(lane)", Ib="skill + C(gb) + C(lane)", Is60="skill + C(gb) + C(lane)",
            Ca="skill + C(lane)", Cb="skill + C(lane)", Cns="C(lane)", CA="skill + C(lane)", CB="skill + C(lane)", Onq="skill + C(gb) + C(lane)")
def sample(pop, s, orig):
    d = {"O": lambda: orig(pop), "Onq": lambda: orig(pop), "Ia": lambda: pop[pop.gb >= 7], "Ib": lambda: pop[pop.gb >= 7], "Is60": lambda: pop[(pop.gb >= 7) & (pop.skill >= .6)],
         "Ca": lambda: pop, "Cb": lambda: pop, "Cns": lambda: pop, "CA": lambda: pop[pop.half == 0], "CB": lambda: pop[pop.half == 1]}[s]()
    return d

def cell(pop, orig, treat, Qa, Qb, extra, rows, specs=SPECS, make=None):
    """Fit `treat + Q + extra + controls` once per spec; rows = [(label, {coef: weight}, k)]. Returns {label: {spec: est+(n,)}}."""
    out = {lab: {} for lab, _, _ in rows}
    for s in specs:
        d = sample(pop, s, orig)
        if make: d = make(d)
        Q = "" if s == "Onq" else (Qb if s in ("Ib", "Cb") else Qa)
        tr = treat(d) if callable(treat) else treat
        f = " + ".join(x for x in (tr, Q, extra, CTRL[s]) if x)
        try: m = fit(d, f)
        except Exception as e: m = None; print("fit failed", s, f, e)
        for lab, w, k in rows:
            out[lab][s] = (est(m, w, k) if m is not None else (np.nan,) * 4) + (len(d),)
    return out

def fmt(e, star=True):
    if np.isnan(e[0]): return "n/a"
    return f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}]" + ("*" if star and e[3] < .05 else "")

def emit(rid, title, res, specs=SPECS, note=""):
    hd = f"\n**{title}**" + (f" ({note})" if note else "") + "\n\n| term | " + " | ".join(specs) + " |\n|---|" + "---|" * len(specs)
    body = []
    for lab, cs in res.items():
        body.append(f"| {lab} | " + " | ".join(fmt(cs[s]) if s in cs else "" for s in specs) + " |")
    first = next(iter(res.values()))
    body.append("| n decks | " + " | ".join(str(first[s][4]) if s in first else "" for s in specs) + " |")
    blk = "\n".join([hd] + body)
    SECTION.setdefault(rid, []).append(blk); P(blk)

skill60 = lambda p: p[p.skill >= .6]
allp = lambda p: p
Q1a, Q1b = lambda c: "qL", lambda c: f"qL_x_{c}"
Q2a, Q2b = lambda c: "qA + t10A", lambda c: f"qA_x_{c} + t10A_x_{c}"

# ------------------------------------------------------------------ 0. diagnostics
P("# Foundations check, deck-level (FRA)\n")
P("## 0. Diagnostics that set the design\n")
x = pd.crosstab(DF.gb, DF.g)
P(f"decks_FRA.json.gz: {len(DF)} decks. Share of decks with games_bucket == deck_games: {(DF.gb == DF.g).mean():.4f} (cross-tab is diagonal). Mean deck WR by games_bucket: "
  + ", ".join(f"{b}: {v:.3f}" for b, v in DF.groupby('gb').y.mean().items()))
raw = pd.read_csv(f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz", usecols=["draft_id", "build_index", "won", "user_n_games_bucket"])
rg = raw.groupby(["draft_id", "build_index"]).agg(n=("won", "size"), ub=("user_n_games_bucket", "first")).reset_index()
mm = DF[DF.fmt == "PremierDraft"].merge(rg, left_on=["draft", "g"], right_on=["draft_id", "n"], how="left")
P(f"Raw Premier game file: user_n_games_bucket takes {sorted(raw.user_n_games_bucket.unique().tolist())} over {len(rg)} decks. Matching decks_FRA Premier decks (draft, game count) to it gives "
  f"user_n_games_bucket in {sorted(mm.ub.dropna().unique().tolist())} only (unmatched {int(mm.ub.isna().sum())}): decks_FRA holds 100+ games players only, and its `games_bucket` column is the deck's own game count.")
P("So 'games bucket 7-9' is 'the deck played 7-9 games', which under the 7-win / 3-loss stop means a winning run (mean deck WR 0.68-0.72), and C(gb) conditions on the number of games, a function of the outcome.")
rng = np.random.default_rng(1); N = 200000; b_true = 0.10
xs = rng.normal(size=N); sk = rng.normal(0, .3, size=N); p = 1 / (1 + np.exp(-(0.2 + b_true * xs + sk)))
w = np.zeros(N); l = np.zeros(N); alive = np.ones(N, bool)
for _ in range(9):
    win = rng.random(N) < p; w += alive & win; l += alive & ~win; alive &= (w < 7) & (l < 3)
nn = w + l
def sl(m, dum=False):
    X = pd.DataFrame({"x": xs[m]})
    if dum: X = pd.concat([X, pd.get_dummies(nn[m].astype(int), drop_first=True).astype(float).reset_index(drop=True)], axis=1)
    return sm.GLM(np.column_stack([w[m], l[m]]), sm.add_constant(X), family=sm.families.Binomial()).fit().params["x"]
allm = np.ones(N, bool)
P(f"\nSimulation (200,000 events, no skill leak, 7 wins / 3 losses stop, true logit slope {b_true}): slope with all decks and no games term {sl(allm):.3f}; all decks with C(games) {sl(allm, True):.3f}; "
  f"games 7-9 with C(games) {sl(nn >= 7, True):.3f}. corr(skill, y) given n games: " + ", ".join(f"n={k}: {np.corrcoef(sk[nn == k], (w / nn)[nn == k])[0,1]:+.2f}" for k in (7, 8, 9))
  + " (observed in decks_FRA: +0.20, +0.16, +0.04). The fall of corr(skill, y) with games, read as a skill leak in review-tests-2026-11-06 s1.1, is reproduced with no leak; conditioning on games removes about three quarters of a true slope.")
P("Consequence: O is the original; Ia/Ib/Is60 are the test plan's 7-9 sample (reported because it was asked for, biased towards zero by construction); Ca/Cb/Cns/CA/CB are the corrected leak-free estimates and carry the verdicts. Verdict rule fixed now: basis holds if Ca and Cb keep the sign with a CI excluding 0 (or, for a null, if no effect beyond the stated bound); weakened if the point estimate falls by more than half or the CI includes 0 while O excluded it; gone if both estimates are within noise of 0; reversed if the CI excludes 0 on the other side.")
P(f"\nPopulations: two-colour, >=20 priced spells (family 1) {len(POP1)} decks; every deck (family 2) {len(POP2)}; Premier {len(POPP)}. Games 7-9 in family 1: {int((POP1.gb >= 7).sum())}.")

# ------------------------------------------------------------------ 1. removal
P("\n## R1. REMOVAL_SLACK 2 / REMOVAL_POINTS 6 (removal)\n")
P("Original: 'decks under 6 removal lose ~0.6 WR points beyond their cards' (synergy-2026-11-02.md, deck tables, different spec); review_deckglm s4 primary dummy -1.22; podcast_removal total +0.19 [-0.11, +0.48]. O here = review_deckglm spec (REM48, skill>=0.60, C(gb)).")
emit("R1", "REM48 count, pts per card", cell(POP1, skill60, "rem", *map(lambda f: f("rem"), (Q1a, Q1b)), "cre + splash", [("rem", {"rem": 1}, K)]))
emit("R1", "fewer than 6 REM48 (dummy), pts", cell(POP1, skill60, "short6", Q1a("rem"), Q1b("rem"), "cre + splash", [("short6", {"short6": 1}, K)]))
POPT = POP1[POP1.tr.notna()]
emit("R1", "shortfall against the pair's top-player target minus 2 (what the builder uses): pts per card short", cell(POPT, skill60, "sfall", Q1a("rem"), Q1b("rem"), "cre + splash", [("sfall", {"sfall": 1}, K)]))
emit("R1", "any shortfall against that target (dummy), pts", cell(POPT, skill60, "sfd", Q1a("rem"), Q1b("rem"), "cre + splash", [("sfd", {"sfd": 1}, K)]),
     note=f"{int(POPT.sfd.mean()*100)}% of decks short; targets come from fitted.json decks.pairs, built from the same 60%+ players, so the shortfall is not an independent yardstick")

# ------------------------------------------------------------------ 2. two-drops
P("\n## R2. Two-drop floor off, curve check skips 2-drops (TWO_DROP_SLACK 99, CURVE_SKIP 2)\n")
P("Original (podcast_twodrop.py, docs/podcast-test-twodrops-FRA.md): 35,646 decks, buckets of cmc-2 creatures against 5: 0-2 +1.15 [+0.26, +2.04], 3 -0.65 [-1.41, +0.12], 4 -0.07, 6+ -0.57; no shortfall, floor off. O here: all decks, skill and C(gb) as in that script.")
emit("R2", "two-drop creature buckets vs 5, pts", cell(POP2, allp, "b02 + b3 + b4 + b6p", Q2a("two"), Q2b("two"), "splash",
     [("0-2", {"b02": 1}, K), ("3", {"b3": 1}, K), ("4", {"b4": 1}, K), ("6+", {"b6p": 1}, K)]))

# ------------------------------------------------------------------ 3. top end
P("\n## R3. TOP_END_SLACK 2 / FINISHER_UP_TO (top end)\n")
P("Original (podcast_topend.py, docs/podcast-test-topend-FRA.md): cmc>=5 count buckets against 2: 0-1 +0.21, 3 +0.58, 4 +0.14, 5+ +0.44, all n.s.; no saturation. 'No saturation' is the basis for the loose top-end slack.")
emit("R3", "cmc>=5 spell count buckets vs 2, pts (plus a separate cmc>=7 count)", cell(POP2, allp, "t01 + t3 + t4 + t5p + c7", Q2a("big"), Q2b("big"), "splash",
     [("0-1", {"t01": 1}, K), ("3", {"t3": 1}, K), ("4", {"t4": 1}, K), ("5+", {"t5p": 1}, K), ("per cmc>=7 card", {"c7": 1}, K)]))
emit("R3", "five-plus drops above the pair's target + 2 (the builder's TOP_END_SLACK penalty), pts per card over", cell(POPT, allp, "bigx", Q1a("big"), Q1b("big"), "cre + splash", [("bigx", {"bigx": 1}, K)]),
     note="new spec for the shipped rule, family-1 sample")

# ------------------------------------------------------------------ 4. split
P("\n## R4. MIN_PER_COLOUR 4 (deep main / thin second colour)\n")
P("Original (podcast_split.py): +0.37 pts per +0.1 main share [-0.06, +0.80], p 0.094 (all skills, C(gb) in the model, no splash, 22-24 spells). The test never found a penalty for lopsided decks; the 4-card floor rests on that plus conformity to top players. Quality (a) = All Decks mean and top-10 of all spells; (b) = the same over the non-mono-coloured spells (gold and colourless), so the main/thin mono split is outside the control.")
sp_ns = quiet(f"{R}/analysis/podcast_split.py"); cls_, by_s = sp_ns["cls"], sp_ns["by"]
srows = []
for d in DECKS:
    if d["splash"] != "" or len(d["lane"]) != 2 or not d["deck_games"]: continue
    spells = []
    for n, k in d["counts"].items():
        c = by_s.get(n) or by_s.get(n.split(" // ")[0])
        if c is None or "Land" in c["types"]: continue
        spells += [c] * k
    if not 22 <= len(spells) <= 24: continue
    L = set(d["lane"]); cl = [cls_(c, L) for c in spells]
    if any(x is None for x in cl): continue
    mono = {x_: [c for c, (f_, _) in zip(spells, cl) if f_ == {x_}] for x_ in L}
    a, b = sorted(L, key=lambda x_: -len(mono[x_])); tot = len(mono[a]) + len(mono[b])
    if tot == 0: continue
    g = [c["deck_colors"]["All Decks"]["gihwr"] for c in spells]
    if min(g) <= 0: continue
    gn = sorted([gi for gi, (f_, _) in zip(g, cl) if len(f_) != 1], reverse=True); gs = sorted(g, reverse=True)
    srows.append(dict(draft=d["draft"], lane="".join(sorted(L)), skill=d["skill"], gb=d["games_bucket"], w=d["deck_wins"], g=d["deck_games"], y=d["deck_wins"] / d["deck_games"],
                      ms10=len(mono[a]) / tot * 10, gold=sum(len(f_) == 2 or (not f_ and col) for f_, col in cl), thin=int(len(mono[b]) < 4), minmono=len(mono[b]),
                      q=np.mean(g), q10=np.mean(gs[:10]), qn=np.mean(gn) if gn else np.nan, q10n=np.mean(gn[:10]) if gn else np.nan))
SP = pd.DataFrame(srows).dropna(); SP["half"] = SP.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
P(f"split sample: {len(SP)} decks, {SP.draft.nunique()} drafts; thin second colour (<4 mono cards): {int(SP.thin.sum())} decks ({SP.thin.mean():.1%}); mean minmono {SP.minmono.mean():.2f}")
emit("R4", "main share, pts per +0.1", cell(SP, allp, "ms10", "q + q10", "qn + q10n", "gold", [("per +0.1 main share", {"ms10": 1}, K)]))
emit("R4", "thin second colour (<4 mono-coloured cards) vs 4+, pts (the shipped floor; new spec)", cell(SP, allp, "thin", "q + q10", "qn + q10n", "gold", [("thin", {"thin": 1}, K)]))

# ------------------------------------------------------------------ 5. lands
P("\n## R5. Land counts / LAND_STEPS (Akshin's rule; data reported, not a revert test)\n")
P("Original (land_count_check.py, 60%+, two-colour, 40 cards, 15-17 lands): 16 vs 17 not significantly worse in any landcycler x dual stratum (0 cyclers +0.24, 1 cycler +1.72, 2+ +1.17; CIs about +-2 to +-8). Strata and quality (All Decks mean of spells) as original; (b) = quality without landcyclers.")
BAS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
lrows = []
for d in DECKS:
    if len(d["lane"]) != 2 or d["deck_games"] < 1: continue
    lands = cy = du = n = 0; qa_, qb_ = [], []
    for name, k in d["counts"].items():
        c = by.get(name.split(" // ")[0]); n += k
        if name in BAS: lands += k; continue
        if not c: continue
        if isl(c):
            lands += k
            if len(c.get("colors") or []) >= 2 and set(c["colors"]) <= set(d["lane"]): du += k
            continue
        iscy = "landcycling" in (c.get("oracle_text") or "").lower(); cy += k * iscy
        g = gA(c)
        if g: qa_ += [g] * k; (qb_ if not iscy else []).extend([g] * k)
    if n != 40 or lands not in (15, 16, 17) or not qa_: continue
    lrows.append(dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=d["games_bucket"], w=d["deck_wins"], g=d["deck_games"], y=d["deck_wins"] / d["deck_games"],
                      lands=lands, cyc=min(cy, 2), duals=min(du, 2) // 2, qa=np.mean(qa_), qb=np.mean(qb_) if qb_ else np.nan, fmt=d["format"]))
LD = pd.DataFrame(lrows).dropna(); LD["half"] = LD.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
LD["l15"], LD["l16"] = (LD.lands == 15).astype(int), (LD.lands == 16).astype(int)
P(f"decks {len(LD)}; lands 15/16/17: {int((LD.lands==15).sum())}/{int((LD.lands==16).sum())}/{int((LD.lands==17).sum())}; games 7-9: {int((LD.gb>=7).sum())} decks, 16-land {int(((LD.gb>=7)&(LD.l16==1)).sum())}, 15-land {int(((LD.gb>=7)&(LD.l15==1)).sum())}")
ls = ["O", "Ia", "Ib", "Ca", "Cb", "Cns", "CA", "CB"]
res5 = {}
for c in (0, 1, 2):
    for u in (0, 1):
        s = LD[(LD.cyc == c) & (LD.duals == u)]
        lab = f"cyc {c if c < 2 else '2+'}, duals {'2+' if u else '0-1'}"
        if s.l16.sum() < 30: res5[f"{lab}: 16 vs 17"] = {sp: (np.nan,) * 4 + (len(s),) for sp in ls}; continue
        rr = cell(s, skill60, lambda d: " + ".join(v for v in ("l15", "l16") if d[v].sum() >= 30), "qa", "qb", "", [("16", {"l16": 1}, K), ("15", {"l15": 1}, K)], specs=ls)
        for k_ in ("16", "15"):
            # fewer than 30 decks at that count in a spec's sample: report n/a, as the original did
            for sp in ls:
                d_ = sample(s, sp, skill60)
                if d_[f"l{k_}"].sum() < 30: rr[k_][sp] = (np.nan,) * 4 + (len(d_),)
            res5[f"{lab}: {k_} vs 17"] = rr[k_]
emit("R5", "lands vs 17 by stratum, pts (n/a: under 30 decks at that count)", res5, specs=ls)
allr = cell(LD, skill60, lambda d: " + ".join(v for v in ("l15", "l16") if d[v].sum() >= 30), "qa", "qb", "", [("16 vs 17, all strata pooled", {"l16": 1}, K), ("15 vs 17, all strata pooled", {"l15": 1}, K)], specs=ls)
emit("R5", "pooled over strata (not in the original)", allr, specs=ls)

# ------------------------------------------------------------------ 6. GIH slope
P("\n## R6. GIH_SLOPE 1.72 (deck WR points per +1 mean spell GIH WR)\n")
P("Origin: deckbuilder.py comment 'FRA decks of 100+ games players, 2026-11-05' and the 'gihwr' objective commit 156b7cc; no script or doc in the repo contains the fit, so the original is NOT reproducible from the repo. Candidate specs below (family-1 decks, lane GIH mean, coefficient x 24.75).")
sl_rows = []
def slope_row(lab, d, ctrl):
    m = fit(d, " + ".join(x for x in ("qL", ctrl) if x)); e = est(m, {"qL": 1}); sl_rows.append((lab, len(d)) + e)
S1 = POP1
for lab, d, ctrl in (("all 2-colour decks, no controls", S1, ""), ("+ lane", S1, "C(lane)"), ("+ lane + splash", S1, "C(lane) + splash"), ("+ lane + skill", S1, "C(lane) + skill"), ("+ lane + skill + C(gb) [original-style controls]", S1, "C(lane) + skill + C(gb)"),
                      ("skill>=0.60, + C(gb) + lane + skill", skill60(S1), "C(lane) + skill + C(gb)"), ("games 7-9 (Ia), + C(gb) + lane + skill", S1[S1.gb >= 7], "C(lane) + skill + C(gb)"),
                      ("games 7-9, no controls", S1[S1.gb >= 7], ""), ("half A, + lane + skill", S1[S1.half == 0], "C(lane) + skill"), ("half B, + lane + skill", S1[S1.half == 1], "C(lane) + skill"),
                      ("Premier only, + lane + skill", S1[S1.fmt == "PremierDraft"], "C(lane) + skill"), ("skill>=0.60, + lane + skill, no C(gb)", skill60(S1), "C(lane) + skill")):
    slope_row(lab, d, ctrl)
blk = "\n**Deck WR points per +1 GIH WR point of the mean spell**\n\n| sample / controls | n | slope [95% CI] |\n|---|---|---|\n" + "\n".join(f"| {a} | {n} | {fmt(tuple(r), False)} |" for a, n, *r in sl_rows)
SECTION.setdefault("R6", []).append(blk); P(blk)
P("Caveats fixed in advance: (1) the card GIH ratings are computed from the same games as the deck win rates (circular: a deck's mean GIH contains its own games, so the slope is inflated by construction, most for rare cards); (2) conditioning on games or keeping skill moves it, so 1.72 has no unique controls; (3) mean GIH is an error-ridden measure of power (attenuation) and contains the game-length bias (review s1.5).")

# ------------------------------------------------------------------ 7. splash
P("\n## R7. SPLASH_TERM 4.07 and the B9 splash cost\n")
P("Original (podcast_splash_speed.py, B9): extended minus clean -1.66 / -1.58 / -1.68 by speed tercile (no quality control), -2.8 / -2.6 / -2.9 with deck quality; splash-only -1.7 / -1.7 / -1.5. SPLASH_TERM 4.07 is the deck model's splash term, used next to a GIH-based quality score, so the comparable number is the equal-quality splash cost. O = Premier, skill + C(gb) + lane, splash-only vs clean, pooled across terciles; Onq = original without quality. (b) = quality of the non-splashed spells (cards outside the lane colours removed).")
POPS = POPP[(POPP.nlane == 2)]; POPS = POPS.assign(spl=POPS.splash)
POPE = POPP.assign(ext=((POPP.splash == 1) | (POPP.nlane >= 3)).astype(int), spl=((POPP.splash == 1) | (POPP.nlane >= 3)).astype(int))
emit("R7", "splash-only vs clean 2-colour, pts", cell(POPS, allp, "spl", "qA", "qA_x_off", "", [("splash", {"spl": 1}, K)], specs=["Onq"] + SPECS), specs=["Onq"] + SPECS)
emit("R7", "extended (splash or 3+ colours) vs clean, pts", cell(POPE, allp, "spl", "qA", "qA_x_off", "", [("extended", {"spl": 1}, K)], specs=["Onq"] + SPECS), specs=["Onq"] + SPECS)

# ------------------------------------------------------------------ 8. finishers
P("\n## R8. FINISHER_POINTS 2 (finishers)\n")
P("Original (pick-level-FRA.md, fra_step2.py): pool-deck (best pair, top 23 castable by GIH), event match record, n=23, 100+ games players: finishers +2.05 [+1.77, +2.33] with the skill bucket (M1); +2.43 all drafts M0. Finisher = cmc>=5 and GIH >= format mean + 1 SD. Pool-level rows use the real user_n_games_bucket (100+ players are the leak-free stratum there); the deck-level analogue uses decks_FRA.")
D8 = pd.read_csv(f"{OUT}/draft_features_FRA.csv.gz"); D8["y"] = D8.event_match_wins / D8.trials
D8["rem"], D8["fin"], D8["dul"] = D8.removal.clip(3, 10), D8.finishers.clip(0, 4), D8.duals.clip(0, 6)
D8["half"] = D8.draft_id.map(lambda x: zlib.crc32(x.encode()) % 2)
def pool(df, skill_term):
    f = "y ~ quality + C(pair) + C(user_n_games_bucket) + C(rank) + rem + fin + dul" + (" + C(user_game_win_rate_bucket)" if skill_term else "")
    return smf.glm(f, df, family=sm.families.Binomial(), freq_weights=df.trials).fit(cov_type="HC1")
pl = []
for lab, df, st in (("n=23, all players, M1 (original row 3)", D8[D8.n == 23], True), ("n=23, 100+ players, M1 (original row 4)", D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100)], True),
                    ("n=23, 100+ players, M0 (no skill)", D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100)], False),
                    ("n=23, 100+, M1, half A", D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100) & (D8.half == 0)], True), ("n=23, 100+, M1, half B", D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100) & (D8.half == 1)], True),
                    ("n=23, 100+, M0, half A", D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100) & (D8.half == 0)], False), ("n=23, 100+, M0, half B", D8[(D8.n == 23) & (D8.user_n_games_bucket >= 100) & (D8.half == 1)], False)):
    m = pool(df, st); pl.append((lab, len(df), est(m, {"fin": 1}), est(m, {"rem": 1}), est(m, {"dul": 1})))
blk = "\n**Pool-level (event match record), pts per extra card**\n\n| sample / model | drafts | finishers | removal | duals |\n|---|---|---|---|---|\n" + "\n".join(f"| {a} | {n} | {fmt(f)} | {fmt(r)} | {fmt(d)} |" for a, n, f, r, d in pl)
SECTION.setdefault("R8", []).append(blk); P(blk)
P("Pool-level (b) quality (pool-deck quality without finishers) needs the pool-deck selection loop of fra_step2.py (about 54k drafts x 10 pairs) and was not rerun; the deck-level analogue below carries control (b).")
emit("R8", "deck-level analogue: finisher count in the final deck, pts per card (removal and creatures controlled)", cell(POP1, skill60, "fin", Q1a("fin"), Q1b("fin"), "rem + cre + splash", [("fin", {"fin": 1}, K)]))

# ------------------------------------------------------------------ 9. creatures
P("\n## R9. Creature count (CREATURE_BELOW 2 / CREATURE_POINTS 3)\n")
P("Original: the coefficient on `cre` in every deck GLM of this family (it is a control there, not a finding); the creature floor itself rests on conformity to top players. Spec: creature count with quality (a) all spells or (b) non-creature spells; splash controlled.")
emit("R9", "creature count, pts per creature", cell(POP1, skill60, "cre", Q1a("cre"), Q1b("cre"), "splash", [("cre", {"cre": 1}, K)]))
emit("R9", "creatures short of the pair's target minus 2 (builder penalty), pts per creature short", cell(POPT, skill60, "csf", Q1a("cre"), Q1b("cre"), "splash", [("csf", {"csf": 1}, K)]))

# ------------------------------------------------------------------ 10. today's nulls
P("\n## R10. Deck-level nulls of 2026-11-06, rerun\n")
P("Same families as the original scripts; B2 top end is R3; the weak-pool test is pool-level (needs final pools) and its overall extended-vs-clean cost is R7.")
emit("R10", "B4 non-board spells (draw-only, lifegain-only, aura/equipment, tricks), pts per spell; removal (engine tags) and creatures controlled",
     cell(POP1, skill60, "nb", Q1a("nb"), Q1b("nb"), "cre + remnb + splash", [("nb", {"nb": 1}, K)]))
emit("R10", "counterspells (CORE 4 cards) next to REM48, pts per card; (b) quality without both sets",
     cell(POP1, skill60, "rem + ctr", Q1a("remctr"), Q1b("remctr"), "cre + splash", [("ctr", {"ctr": 1}, K), ("rem48", {"rem": 1}, K)]))
emit("R10", "mana sinks (36-card tag), pts per sink", cell(POP1, skill60, "sink", Q1a("sink"), Q1b("sink"), "cre + splash", [("sink", {"sink": 1}, K)]))

# double-pip splash: sources
BASIC = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
dbl, lands_ = {}, {}
for n, v in by.items():
    s_ = re.findall(r"\{([WUBRG])\}", (v.get("mana_cost") or "").split("//")[0]); dbl[n] = next((c for c in "WUBRG" if s_.count(c) >= 2), None)
    if isl(v):
        base = n.replace("Snow-Covered ", ""); lands_[n] = set("WUBRG") if n == "Room of Refuge" else {BASIC[base]} if base in BASIC else {c for c in (v.get("colors") or []) if c in "WUBRG"}
cycd = CYC
dprows = []
P1i = POP1.index
dk = {(d["draft"], d["lane"], d["deck_games"], d["deck_wins"], d["splash"]): d for d in DECKS}
for i, r in POP1[POP1.splash == 1].iterrows():
    d = dk[(r.draft, r.lane, r.g, r.w, r.splash_c)]
    sp_ = d["splash"]
    if len(sp_) != 1: continue
    c = {n.split(" // ")[0]: k for n, k in d["counts"].items()}
    src = sum(k for n, k in c.items() if (n in lands_ and sp_ in lands_[n]) or n in cycd or n == "Murmuring Volume")
    dpn = {n for n in c if dbl.get(n) == sp_ and n not in cycd}
    if not dpn: continue
    q = [gL(by[n], d["lane"]) for n, k in c.items() if n in by and not isl(by[n]) and n not in dpn and gL(by[n], d["lane"]) for _ in range(k)]
    dprows.append(dict(idx=i, srcc=min(max(src, 3), 10), qL_x_dp=np.mean(q) if q else np.nan))
DP = POP1.loc[[x_["idx"] for x_ in dprows]].copy(); DP["srcc"] = [x_["srcc"] for x_ in dprows]; DP["qL_x_dp"] = [x_["qL_x_dp"] for x_ in dprows]; DP["s8"] = (DP.srcc >= 8).astype(int)
DP = DP.dropna(subset=["qL_x_dp"])
P(f"double-pip splash decks (single-colour splash with a DP copy): {len(DP)}; games 7-9: {int((DP.gb>=7).sum())}; skill>=0.60: {int((DP.skill>=.6).sum())}")
emit("R10", "double-pip splash decks: pts per source (3..10, linear; the review's pre-specifiable estimate); (b) quality without the DP cards", cell(DP, allp, "srcc", "qL", "qL_x_dp", "cre", [("per source", {"srcc": 1}, K)]),
     note="original = review_dpsplash.py +0.67 [-0.10, +1.43] all skills, all games buckets")
emit("R10", "double-pip splash: 8+ sources vs fewer (post hoc cut, winner's curse; reported for the slope's sake only)", cell(DP, allp, "s8", "qL", "qL_x_dp", "cre", [("8+ vs <8", {"s8": 1}, K)]))

# empower
# need separate E and W counts: recount from decks
ER = {}
for d in DECKS:
    if d["format"] != "PremierDraft": continue
    c = {n.split(" // ")[0]: k for n, k in d["counts"].items()}
    ER[(d["draft"], d["lane"], d["deck_games"], d["deck_wins"], d["splash"])] = (sum(c.get(n, 0) for n in EMP), sum(c.get(n, 0) for n in WAYS))
EW = [ER[(r.draft, r.lane, r.g, r.w, r.splash_c)] for r in POPP.itertuples()]
POPW = POPP.assign(E=[a for a, b in EW], W=[b for a, b in EW])
POPW["Ec"] = POPW.E - 4; POPW["W1"] = (POPW.W >= 1).astype(float); POPW["W1xEc"] = POPW.W1 * POPW.Ec
for k_, cnd in (("Wn1", POPW.W == 1), ("Wn2", POPW.W == 2), ("Wn3", POPW.W == 3), ("Wn4", POPW.W >= 4)): POPW[k_] = cnd.astype(int)
emit("R10", "empower Jace: extra WR pts per empower card when a Way is present (original +0.24 [0.00, +0.48]) and per empower card with no Way (original -0.18); (b) quality without Ways and empower cards",
     cell(POPW, allp, "W1xEc + Ec + W1", Q2a("wemp"), Q2b("wemp"), "splash", [("Way present x E", {"W1xEc": 1}, K), ("E, no Way", {"Ec": 1}, K)]))
emit("R10", "Way count vs 0 (original 3 minus 2: +0.82 [-0.52, +2.17]); contrasts",
     cell(POPW, allp, "Wn1 + Wn2 + Wn3 + Wn4 + Ec", Q2a("wemp"), Q2b("wemp"), "splash", [("2 minus 1", {"Wn2": 1, "Wn1": -1}, K), ("3 minus 2", {"Wn3": 1, "Wn2": -1}, K), ("4+ minus 3", {"Wn4": 1, "Wn3": -1}, K)]))


VERDICTS = {
"intro": "Two corrections came out of the diagnostics (log section 0). First, in `decks_FRA.json.gz` the `games_bucket` column equals the deck's own game count in every row, and every deck belongs to a 100+ games player (checked against the raw game file, where `user_n_games_bucket` takes 1, 5, 10, 50, 100, 500). The test plan's 'games buckets 7-9' sample is therefore the decks that played 7 to 9 games, a winning run under the 7-win / 3-loss stop (deck WR 0.68 to 0.72), and `C(gb)`, in every original GLM, conditions on a function of the outcome. In a simulation with no skill leak, `C(games)` cuts a true logit slope of 0.10 to 0.023 (all decks) or 0.040 (games 7-9), and the fall of corr(skill, deck WR) with games (0.30 to 0.04, read in the review as a skill leak) is reproduced. For 100+ games players the skill bucket holds about 1% of the deck's own record (context-controls-FRA.md), so skill is kept. Second, the verdicts below use Ca / Cb (all decks of the rule's population, skill kept, no games term, halves by draft hash); Ia / Ib / Is60 are the 7-9 sample as asked, shown because it was requested, and are attenuated by construction (removal slope +0.30 in Ca, -0.01 in Ia; finishers +1.47 against +0.18). The review's finding that the removal premium and the 'below 6' shortfall vanish in buckets 7-9 is this artefact. The originals with `C(gb)` were biased towards zero, not away from it. Quality (b) credits the tested class with its own ratings (slot value), so it exceeds (a) for classes rated above the average spell (removal, finishers, 5+ drops) and falls below it for creatures. About 40 estimates are reported with no multiplicity correction; a verdict needs (a) and (b) to agree and both halves to keep the sign. The C columns were fixed after the diagnostics and before any C estimate was seen; no cut points are new except where marked.",
"R1": "Basis holds, and is stronger than stated. Fewer than 6 removal costs -1.11 [-1.77, -0.44] at equal all-spell quality (a) and -1.99 [-2.65, -1.32] at equal quality of the other spells (b); halves -1.23 / -0.96; original -1.22 (skill>=0.60, C(gb)) and -0.6 in the deck tables. Per-card slope +0.30 [+0.13, +0.46] (a), +0.60 (b), against +0.19 originally. Against the pair's own floor (top-player count minus 2, the builder's `REMOVAL_SLACK` form): -0.64 [-1.33, +0.05] per card short (a), -1.37 (b); any shortfall -1.11 / -2.13. `REMOVAL_POINTS` 6 for a pool fully short (about 5 cards at the pair floors, so about 1.2 per card) sits between (a) and (b). The target comes from the same 60%+ players, so the shortfall rows are not independent of the rule.",
"R2": "Basis holds (null). Three two-drops against 5: -0.49 [-1.32, +0.34] (a), -0.68 [-1.52, +0.15] (b); four +0.12; six or more -0.42 [-1.16, +0.32]. The 0-2 bucket, +1.15 originally, is +1.00 [+0.05, +1.95] (a), +0.79 n.s. (b), +0.13 without skill and +0.32 / +1.71 across halves, so it is not stable. The leak-free CIs rule out a penalty worse than about -1.3 (a) / -1.5 (b) for three two-drops and worse than 0 (a) / -0.2 (b) for 0-2. The floor stays off.",
"R3": "Basis holds (null; the penalty has no support). Five or more against two: +0.76 [-0.11, +1.63] (a), +2.75 (b, the class carries above-average ratings); no bucket is negative and halves agree. Above the pair target plus 2, the builder's penalty: +0.33 [-0.38, +1.04] per card (a), +0.95 (b); it rules out a penalty steeper than 0.4 per card, the point estimate has the wrong sign for a penalty. cmc>=7 adds +0.20 [-0.21, +0.61]. GIH carries a game-length premium for expensive cards (review 1.5), so (a) is the conservative reading.",
"R4": "Basis holds (null, low power). Main share per +0.1: +0.32 [-0.15, +0.79] (a), +0.34 (b), halves +0.60 / +0.06 (original +0.37). Thin second colour (fewer than 4 mono-coloured cards, 336 decks, 2.6%): +1.68 [-0.94, +4.30], so no penalty and a bound of about -0.9. `MIN_PER_COLOUR` 4 binds in 2.6% of decks and has no deck-level evidence of harm behind it; it is a conformity rule.",
"R5": "Basis holds on harm; the data lean the other way. Pooled 16 against 17: +0.81 [+0.05, +1.56] (a), +0.78 (b), halves +0.48 / +1.12; no stratum is significantly negative and the one significant stratum (2+ cyclers, 0-1 duals) is +1.28 [+0.11, +2.45]; lower CI bounds by stratum run from -6.6 to +0.1, so single strata are weak. 15 against 17: +4.33 [+1.08, +7.59] on 272 decks (selection into 15 lands is not controlled), strata mostly n/a. Original +0.24 to +1.72. Not a revert candidate.",
"R6": "Weakened: not reproducible and the scale is off. No script or doc in the repo contains the 1.72 fit. The slope is 1.1 to 7.9 depending on controls: leak-free (all 100+ decks, lane and skill) 5.95 [5.55, 6.36], halves 6.10 / 5.79; lane only 7.65; with `C(gb)` 3.5, in the 7-9 sample 2.9, and 1.1 to 2.1 without lane. 1.72 is only reached by specs that condition on games or omit lane fixed effects. The card GIH ratings come from the same games, so every slope here is an upper bound (circular), and 1.72 is not shown to be wrong, only unsupported. What matters is the ratio: the 'gihwr' objective asks `SPLASH_TERM` / `GIH_SLOPE` = 2.4 points of mean GIH for a splash; with a leak-free splash cost of 3.9 and a slope near 6 the ratio is 0.7. Proposed follow-up (to Akshin, not done): refit the slope with ratings built from one half of the games and the deck win rate from the other.",
"R7": "Basis holds. Splash-only against clean at equal all-spell quality: -3.92 [-4.61, -3.23] (a), -3.05 [-3.72, -2.37] (b), halves -3.81 / -4.03; the original spec reproduces (O -2.73, original -2.6 to -2.9). `SPLASH_TERM` 4.07 lies inside the (a) CI. B9's -1.6 is the no-quality number (-1.45 [-2.03, -0.87] pooled here): splash decks hold better cards, so it understates the equal-quality cost by about 2.5 points, and the builder's term is rightly the equal-quality one. The 7-9 sample gives -1.9 (attenuated). The speed interaction (null, interval +-1.4) was not rerun.",
"R8": "Basis holds. Pool level, 100+ games players, n=23: finishers +2.05 [+1.77, +2.33] (reproduced exactly), +2.04 without the skill bucket, halves +2.17 / +1.91. Deck-level analogue: +1.47 [+1.05, +1.89] (a), +3.40 (b), halves +1.19 / +1.75; the 7-9 sample gives +0.18 (artefact). Still an association with 'has top cards' (pick-level-FRA.md), not a price; pool-level (b) was not rerun. `FINISHER_POINTS` 2 is at the (a) deck estimate.",
"R9": "Gone as a deck-level basis (it only ever rested on conformity). Creatures at equal all-spell quality: +0.05 [-0.09, +0.19] per creature (a), which rules out more than +0.19 or less than -0.09; (b) -0.39 [-0.53, -0.26] because creatures are rated below the average spell. Creatures short of the pair target minus 2: +0.08 [-0.21, +0.36] per creature (a), so no penalty steeper than 0.21; (b) +0.73. `CREATURE_BELOW` 2 and `CREATURE_POINTS` 3 stay as conformity rules; the data show no cost of a creature shortfall.",
"R10": "B4 non-board spells: +0.06 [-0.23, +0.34] per spell (a), -0.28 [-0.56, -0.00] (b); null stands, rules out worse than -0.23 per spell at equal rating. B2 top end is R3. Counterspells: -0.54 [-0.98, -0.11] per card (a), -0.03 (b) (original -0.38 n.s.); still opposite to the within-deck +0.84 (review 1.6), as expected since the deck-level (a) compares equal ratings. Mana sinks: -0.23 [-0.39, -0.06] (a and b), halves -0.26 / -0.19 (original -0.18 n.s.); no benefit, rules out anything above -0.06 per sink. Double-pip splash: per source +0.87 [+0.07, +1.67] (a), +0.92 (b), halves +0.73 / +0.94 (original +0.67 n.s.), 1,847 decks; the post hoc 8+ cut gives +6.15 [+3.00, +9.30], halves +7.2 / +4.9, but the cut was chosen on all decks so the halves are not held out; the slope is the pre-specifiable number. Empower: Way present x empower count +0.52 [+0.24, +0.79] per card (a), +0.56 (b), halves +0.29 / +0.73 (original +0.24 [0.00, +0.48]); empower cards with no Way -0.36. Way count: 3 minus 2 +2.34 [+0.77, +3.90] (original +0.81 n.s.), 2 minus 1 -1.00 [-1.97, -0.02], 4+ minus 3 -1.43 [-4.11, +1.26]: no stacking penalty, but the shape (-1.0 then +2.3) and about 14 tests in the family make it a lead, not a finding; the within-deck test stays the arbiter. The weak-pool test is pool-level and not rerun; its overall extended-vs-clean cost is R7.",
"table": """| # | Rule | Original basis | Leak-free (Ca, a / b) | Rules out (null rows) | Verdict |
|---|---|---|---|---|---|
| 1 | REMOVAL_SLACK 2, REMOVAL_POINTS 6 | <6 removal -1.22; slope +0.19 | -1.11 / -1.99; slope +0.30 / +0.60 | | basis holds (stronger) |
| 2 | TWO_DROP_SLACK 99, CURVE_SKIP 2 | no shortfall (3: -0.65) | 3: -0.49 / -0.68 | 3-drop deficit worse than -1.3 | basis holds |
| 3 | TOP_END_SLACK 2 | no saturation (5+: +0.44) | 5+: +0.76 / +2.75; over target+2 +0.33 / +0.95 | penalty steeper than 0.4 per card | basis holds; penalty unsupported |
| 4 | MIN_PER_COLOUR 4 | main share +0.37 per 0.1 n.s. | +0.32 / +0.34; thin +1.68 | thin penalty worse than -0.9 | basis holds (low power) |
| 5 | LAND_STEPS | 16 vs 17 not worse | 16 vs 17 +0.81 / +0.78 | 16 worse than -0.05 pooled | holds; not a revert candidate |
| 6 | GIH_SLOPE 1.72 | not locatable | 5.95 (lane, skill); 1.1 to 7.9 by spec | | weakened (unreproducible, ratio to SPLASH_TERM off) |
| 7 | SPLASH_TERM 4.07 | B9 -1.6; model 4.07 | -3.92 / -3.05 | | basis holds |
| 8 | FINISHER_POINTS 2 | +2.05 pool level | +2.05 pool; deck +1.47 / +3.40 | | basis holds |
| 9 | CREATURE_BELOW 2, CREATURE_POINTS 3 | conformity | +0.05 / -0.39 per creature; short +0.08 | penalty steeper than 0.21 per creature | gone (null; conformity only) |
| 10 | B4, counters, sinks, DP splash, empower | nulls and leads | B4 +0.06; counters -0.54; sinks -0.23; DP +0.87 per source; Way x E +0.52 | B4 worse than -0.23; sinks any benefit | nulls hold; DP slope and Way x E leads strengthen; none reversed |""",
}

# ------------------------------------------------------------------ write
open(f"{OUT}/foundations_deck.txt", "w").write("\n".join(lines))
verd = VERDICTS
TITLES = dict(R1="1. Removal (REMOVAL_SLACK 2, REMOVAL_POINTS 6)", R2="2. Two-drop floor off (TWO_DROP_SLACK 99, CURVE_SKIP 2)", R3="3. Top end (TOP_END_SLACK 2, FINISHER_UP_TO)",
              R4="4. Colour split (MIN_PER_COLOUR 4)", R5="5. Land counts (LAND_STEPS)", R6="6. GIH_SLOPE 1.72", R7="7. SPLASH_TERM 4.07 and the splash cost", R8="8. Finishers (FINISHER_POINTS 2)",
              R9="9. Creature count (CREATURE_BELOW 2, CREATURE_POINTS 3)", R10="10. Deck-level nulls of 2026-11-06")
sec = ["<!-- BEGIN deck-level -->", "## Deck-level effects behind builder rules", "", verd.get("intro", ""), "",
       "Columns: O original spec (Onq: without the quality control, R7 only); Ia / Ib games 7-9 as asked (quality a / b); Is60 Ia with skill>=0.60; Ca / Cb corrected sample (all decks, skill kept, no games term; quality a / b); Cns Ca without skill; CA / CB halves of Ca. "
       "Effects in WR points at 55% (per card unless stated), 95% CI clustered by draft, * p<0.05. Full log: analysis/out/foundations_deck.txt; script analysis/foundations_deck.py.", ""]
for rid, t in TITLES.items():
    sec += [f"### {t}", "", "", ""] + SECTION.get(rid, []) + ["", verd.get(rid, "(verdict pending)"), ""]
sec += ["### Verdicts", "", verd.get("table", "(pending)"), "<!-- END deck-level -->", ""]
block = "\n".join(sec)
if os.path.exists(DOC):
    old = open(DOC).read()
    if "<!-- BEGIN deck-level -->" in old:
        new = re.sub(r"<!-- BEGIN deck-level -->.*?<!-- END deck-level -->\n?", lambda m_: block, old, flags=re.S)
    else: new = old.rstrip("\n") + "\n\n" + block
else:
    new = "# Foundations check, FRA\n\nRe-test of the evidence behind shipped builder and picker rules on samples and controls that do not select on the outcome. Sections by workstream.\n\n" + block
open(DOC, "w").write(new)
print("done")
