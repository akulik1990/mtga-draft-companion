"""Synergy tests re-reported by player skill (Akshin's evidence rule, 2026-11-06): good (skill >= 0.60), average (0.52-0.58), weak (< 0.52),
plus good minus weak. Groups come from the player's user_game_win_rate_bucket, never from the deck's result. Primary population: players with
user_n_games_bucket >= 100 (P100); all players (ALL) as the secondary line. Deck-level spec: skill kept as a control, NO game-count term.
Designs are those of the originals, re-implemented (tools/within_deck.py and tools/card_needs.py are NOT run; they rewrite shared files):
  S1  tools/within_deck.py: reviewed card_wants lists + themes. Deck-demeaned linear probability of winning on enablers in hand, cards and lands
      in hand held fixed; "with" = games with the payoff in hand, "alone" = payoff in deck but not in hand; synergy = with - alone. Keyed by draft id
      as the original. Bootstrap over drafts (shared multiplicities across every test/group, so differences between groups come from one joint draw).
  S2  tools/card_needs.py: logistic, game result on [deck enabler count, card in hand, product, deck GIH quality, skill, on play], games whose deck
      held the card; the product is the test (per enabler in hand, WR points at 55%).
  S3  video_empower.py / foundations_deck.py rule 10: deck-level GLM (empower x Way present, Way count) and within-deck W2 (empower in hand x Way
      in hand) and W4 (Ways in hand).
  S4  payoff_enabler.py: deck-level themes, per group, without games term.
Usage: python3 analysis/skill_groups_synergy.py close|full   (needs the cache from analysis/skill_groups_cache.py; run under systemd-run, MemoryMax=4G)"""
import gzip, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import norm

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MTGA_COMPANION_DATA", f"{R}/data")
sys.path.insert(0, R)
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
MODE = sys.argv[1] if len(sys.argv) > 1 else "close"
OUT = f"{R}/analysis/out/skill_groups_synergy{'' if MODE == 'full' else '_' + MODE}.txt"
lines = []
def P(s=""): print(s, flush=True); lines.append(s)

B = 400
rng = np.random.default_rng(20261106)
GROUPS = ["all", "good", "average", "weak"]
K = 100 * .55 * .45
def gmask(s):
    s = np.round(s.astype(float), 2)
    return {"all": np.ones(len(s), bool), "good": s >= .60, "average": (s >= .52) & (s < .60), "weak": s < .52}

# ------------------------------------------------------------------ data
Z = np.load(f"{SCR}/syn_games.npz", allow_pickle=True)
H, D, names = Z["H"], Z["D"], [str(x) for x in Z["names"]]
won = Z["won"].astype(float); skill = Z["skill"]; ngames = Z["ngames"]; play = Z["play"].astype(float)
draft = np.asarray(Z["draft"], dtype=str); build = Z["build"]
nidx = {n: i for i, n in enumerate(names)}
dcode, dnames = pd.factorize(draft); ND = len(dnames)
kcode, _ = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))
ratings = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
lands = {c["name"] for c in ratings.values() if "Land" in (c.get("types") or [])}
lcols = [nidx[n] for n in lands if n in nidx]
ncard = H.sum(1, dtype=np.int32).astype(float); Lh = H[:, lcols].sum(1, dtype=np.int32).astype(float)
POPS = {"P100": ngames >= 100, "ALL": np.ones(len(won), bool)}
if MODE == "heldout": POPS = {"LT100": ngames < 100}   # players with fewer than 100 games: disjoint from the primary P100 sample
G = gmask(skill)
MASKS = {(p, g): pm & G[g] for p, pm in POPS.items() for g in GROUPS}
P(f"games {len(won)}, drafts {ND}, decks (draft x build) {kcode.max()+1}; skill missing {np.isnan(skill).mean():.4f}; ngames>=100 share {(ngames>=100).mean():.3f}")
for p in POPS:
    P(f"  {p}: " + ", ".join(f"{g} {MASKS[(p, g)].sum()} games / {len(np.unique(dcode[MASKS[(p, g)]]))} drafts" for g in GROUPS))
W = np.array([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)], dtype=np.float64)

def wfit(rows, X, key, keep, need_std=True):
    """deck-demeaned OLS of won on X[rows] (n x p); returns (est*100 for first `keep` coefs, bootstrap B x keep, number of keys) or None.
    Keys with a single game are dropped; bootstrap is over drafts through per-draft sufficient statistics and the shared multiplicities W."""
    rows = rows[np.bincount(pd.factorize(key[rows])[0])[pd.factorize(key[rows])[0]] >= 2]
    if len(rows) == 0: return None
    k, ku = pd.factorize(key[rows]); nk = len(ku)
    A = np.column_stack([X[rows], won[rows]]).astype(float)
    cnt = np.bincount(k).astype(float)
    for j in range(A.shape[1]): A[:, j] -= (np.bincount(k, A[:, j], minlength=nk) / cnt)[k]
    if nk < 50 or A[:, 0].std() == 0: return None
    p1 = A.shape[1]; p = p1 - 1
    uc, ci = np.unique(dcode[rows], return_inverse=True)
    S = np.zeros((len(uc), p1, p1))
    for i in range(p1):
        for j in range(i, p1): S[:, i, j] = S[:, j, i] = np.bincount(ci, A[:, i] * A[:, j], minlength=len(uc))
    def solve(T): return 100 * np.linalg.pinv(T[..., :p, :p]) @ T[..., :p, p:p + 1]
    est = solve(S.sum(0))[:keep, 0]
    T = np.einsum("bc,cij->bij", W[:, uc], S)
    bs = solve(T)[:, :keep, 0]
    return est, bs, nk

# ------------------------------------------------------------------ S1 within-deck synergy (tools/within_deck.py)
def syn_rows(Pc, Ec, mask):
    en = H[:, Ec].sum(1, dtype=np.int32).astype(float)
    X = np.column_stack([en, ncard, Lh])
    inh = H[:, Pc].sum(1) > 0; ind = D[:, Pc].sum(1) > 0
    res = {}
    for nm, rows in (("with", np.where(inh & mask)[0]), ("alone", np.where(ind & ~inh & mask)[0])):
        res[nm] = wfit(rows, X, dcode, 1)
    return res

def build_tests():
    from companion.advisor_v2 import needs, config
    tests = [(card, [nidx[card]] if card in nidx else [], set(en)) for card, en in needs.synergy_lists("FRA")]
    fit = config.fitted("FRA", "synergy") or {}
    for name, t in (fit.get("themes") or {}).items():
        if t.get("enablers") and t.get("payoffs"):
            tests.append((f"theme: {name}", [nidx[n] for n in t["payoffs"] if n in nidx], set(t["enablers"]) - set(t["payoffs"])))
    out, seen = [], Counter()
    for label, Pc, en in tests:
        Ec = [nidx[e] for e in en if e in nidx and e not in lands]
        if not Pc or not Ec: continue
        base = label.split(" (")[0]; seen[base] += 1
        out.append((label if seen[base] == 1 else f"{label} (list {seen[base]})", Pc, Ec))
    return out

def diffz(a, b):  # a, b = (est, bs) -> estimate, se, z, ci
    if a is None or b is None: return None
    d = a[0][0] - b[0][0]; s = np.std(a[1][:, 0] - b[1][:, 0], ddof=1)
    return d, s, d / s if s > 0 else np.nan

S1 = []   # rows: dict
def run_s1(tests):
    for label, Pc, Ec in tests:
        for p in POPS:
            r = {g: syn_rows(Pc, Ec, MASKS[(p, g)]) for g in GROUPS}
            for g in GROUPS:
                w, a = r[g]["with"], r[g]["alone"]
                if w is None or a is None: S1.append(dict(test=label, pop=p, group=g, ok=False)); continue
                syn = w[0][0] - a[0][0]; sbs = w[1][:, 0] - a[1][:, 0]
                S1.append(dict(test=label, pop=p, group=g, ok=True, nd=w[2], with_=w[0][0], with_se=np.std(w[1][:, 0], ddof=1),
                               alone=a[0][0], syn=syn, syn_se=np.std(sbs, ddof=1), w_bs=w[1][:, 0], a_bs=a[1][:, 0]))
        print("S1 done", label, flush=True)

def s1_get(label, p, g):
    for r in S1:
        if r["test"] == label and r["pop"] == p and r["group"] == g and r["ok"]: return r
def s1_diff(label, p, key):
    a, b = s1_get(label, p, "good"), s1_get(label, p, "weak")
    if not a or not b: return None
    if key == "syn": d = a["syn"] - b["syn"]; bs = (a["w_bs"] - a["a_bs"]) - (b["w_bs"] - b["a_bs"])
    else: d = a["with_"] - b["with_"]; bs = a["w_bs"] - b["w_bs"]
    s = np.std(bs, ddof=1); return d, s, d / s

def s1_table(label_filter=None, pops=None):
    pops = pops or tuple(POPS)
    for p in pops:
        P(f"\n### S1 within-deck synergy (with payoff in hand minus alone), WR points per enabler, population {p}\n")
        P("| test | all-skill (orig) | good | average | weak | good minus weak (SE) z | with-payoff effect: good / weak, diff z |\n|---|---|---|---|---|---|---|")
        for label in dict.fromkeys(r["test"] for r in S1):
            if label_filter and not label_filter(label): continue
            cells = []
            for g in GROUPS:
                r = s1_get(label, p, g)
                cells.append("n/a" if not r else f"{r['syn']:+.2f} ({r['syn_se']:.2f}) [{r['nd']}]")
            d = s1_diff(label, p, "syn"); dw = s1_diff(label, p, "with")
            g_, w_ = s1_get(label, p, "good"), s1_get(label, p, "weak")
            P(f"| {label} | " + " | ".join(cells) + " | " + ("n/a" if not d else f"{d[0]:+.2f} ({d[1]:.2f}) z {d[2]:+.1f}") + " | "
              + ("n/a" if not dw else f"{g_['with_']:+.2f} / {w_['with_']:+.2f}, diff z {dw[2]:+.1f}") + " |")

def s1_with_table():
    for p in POPS:
        P(f"\n### S1b within-deck effect with the payoff in hand (the 'with payoff' column of docs/within-deck-FRA.md), WR points per enabler (SE), population {p}\n")
        P("| test | all-skill | good | average | weak | good minus weak z |\n|---|---|---|---|---|---|")
        for label in dict.fromkeys(r["test"] for r in S1):
            c = []
            for g in GROUPS:
                r = s1_get(label, p, g); c.append("n/a" if not r else f"{r['with_']:+.2f} ({r['with_se']:.2f})")
            d = s1_diff(label, p, "with"); P(f"| {label} | " + " | ".join(c) + f" | {'n/a' if not d else f'{d[0]:+.2f} z {d[2]:+.1f}'} |")

def s1_summary():
    for p in POPS:
        labels = list(dict.fromkeys(r["test"] for r in S1))
        zs = [s1_diff(l, p, "syn") for l in labels]; zs = [z for z in zs if z]
        zd = np.array([z[2] for z in zs])
        gz = {g: np.array([s1_get(l, p, g)["syn"] / s1_get(l, p, g)["syn_se"] for l in labels if s1_get(l, p, g)]) for g in GROUPS[1:]}
        n = len(zd)
        P(f"\n### S1 summary, population {p}: {n} tests with good and weak both estimable")
        P(f"good minus weak synergy: mean z {zd.mean():+.2f} (0 under no difference); |z|>1.96 in {(abs(zd)>1.96).sum()} (expected {0.05*n:.1f}); |z|>2.58 in {(abs(zd)>2.58).sum()} (expected {0.01*n:.1f}); "
          f"largest |z| {abs(zd).max():.2f}; sum z^2 {np.sum(zd**2):.0f} on {n} (chi2 mean {n}; tests share data and overlapping lists, so only indicative); Bonferroni z for {n} tests = {norm.isf(0.025/n):.2f}")
        for g in GROUPS[1:]:
            z = gz[g]; P(f"{g}: synergy z mean {z.mean():+.2f}, |z|>1.96 in {(abs(z)>1.96).sum()}/{len(z)} (expected {0.05*len(z):.1f}), largest {z[np.argmax(abs(z))]:+.2f}; Bonferroni (3 groups x {n}) z = {norm.isf(0.025/(3*n)):.2f}")
        top = sorted(((abs(d[2]), l, d) for l, d in zip(labels, [s1_diff(l, p, 'syn') for l in labels]) if d), reverse=True)[:6]
        P("largest good-minus-weak differences: " + "; ".join(f"{l} {d[0]:+.2f} (z {d[2]:+.1f})" for _, l, d in top))

# ------------------------------------------------------------------ S2 card_needs logistic
def logit(X, y):
    w = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-(X @ w)))
        Hh = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(Hh, X.T @ (y - p)); w += step
        if np.abs(step).max() < 1e-8: break
    return w, np.sqrt(np.diag(np.linalg.inv(Hh)))

def run_s2():
    from companion.advisor_v2 import config
    from companion.advisor_v2.needs import is_enabler
    cards = {}
    for c in ratings.values(): cards.setdefault(c["name"], c)
    wants = {}
    for k, v in (config.layer("FRA", "manual").get("card_wants") or {}).items():
        if k.startswith("_"): continue
        wants[k] = v
        for x in v.get("extra") or []: wants[f"{k}: {x['name']}"] = {**x, "card": k}
    gih = np.zeros(len(names))
    for n, c in cards.items():
        if n in nidx: gih[nidx[n]] = ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0
    Dg = D.astype(np.float32)
    quality = (Dg @ gih.astype(np.float32)) / np.maximum(1, Dg @ (gih > 0).astype(np.float32)); del Dg
    sk = np.where(np.isnan(skill), .55, skill).astype(float)
    sc = .55 * .45 * 100; out = []
    for x, spec in wants.items():
        base = spec.get("card", x)
        if base not in nidx: continue
        en = [nidx[n] for n, c in cards.items() if n != base and n in nidx and "Basic" not in (c.get("types") or []) and is_enabler(c, spec)]
        if not en: continue
        indeck = D[:, nidx[base]] > 0; hand = (H[:, nidx[base]] > 0).astype(float); nen = D[:, en].sum(1, dtype=np.int32).astype(float)
        for p in POPS:
            for g in GROUPS:
                m = np.where(indeck & MASKS[(p, g)])[0]
                if len(m) < 200 or hand[m].sum() < 30: continue
                q = quality[m]
                Xm = np.column_stack([np.ones(len(m)), nen[m], hand[m], nen[m] * hand[m], q - q.mean(), sk[m] - sk[m].mean(), play[m]])
                if Xm[:, 3].std() == 0: continue
                w, se = logit(Xm, won[m]); out.append(dict(test=x, pop=p, group=g, n=len(m), est=w[3] * sc, se=se[3] * sc))
    return out

def s2_table(S2):
    for p in POPS:
        P(f"\n### S2 card_needs logistic, WR points per enabler when the card is drawn (SE), population {p}\n")
        P("| card | all-skill | good | average | weak | good minus weak (SE) z |\n|---|---|---|---|---|---|")
        zs = []
        for t in dict.fromkeys(r["test"] for r in S2):
            g = {r["group"]: r for r in S2 if r["test"] == t and r["pop"] == p}
            c = [f"{g[k]['est']:+.2f} ({g[k]['se']:.2f})" if k in g else "n/a" for k in GROUPS]
            if "good" in g and "weak" in g:
                d = g["good"]["est"] - g["weak"]["est"]; s = np.hypot(g["good"]["se"], g["weak"]["se"]); zs.append(d / s); dd = f"{d:+.2f} ({s:.2f}) z {d/s:+.1f}"
            else: dd = "n/a"
            P(f"| {t} | " + " | ".join(c) + f" | {dd} |")
        zs = np.array(zs)
        P(f"S2 {p}: {len(zs)} tests; good minus weak |z|>1.96 in {(abs(zs)>1.96).sum()} (expected {0.05*len(zs):.1f}); mean z {zs.mean():+.2f}; Bonferroni z {norm.isf(0.025/len(zs)):.2f}; largest |z| {abs(zs).max():.2f}")

# ------------------------------------------------------------------ S3 empower / Way
def deck_tables():
    import statsmodels.api as sm, statsmodels.formula.api as smf
    by = {}
    for v in ratings.values():
        n = v["name"].split(" // ")[0]
        if n in by and not int(v.get("isprimarycard", 1)): continue
        by[n] = v
    gA = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
    isl = lambda v: "Land" in v["types"]
    EL = json.load(open(f"{R}/analysis/out/video_empower_lists.json")); WAYS, EMP = set(EL["ways"]), set(EL["empower"])
    rows = []
    for d in json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz", "rt")):
        if d["format"] != "PremierDraft": continue
        c = {}
        for n, k in d["counts"].items(): c[n.split(" // ")[0]] = c.get(n.split(" // ")[0], 0) + k
        ga = np.array([gA(by[n]) for n, k in c.items() if n in by and not isl(by[n]) for _ in range(k)], float); ga = ga[ga > 0]
        if len(ga) == 0: continue
        rows.append(dict(draft=d["draft"], lane=d["lane"], splash=int(d["splash"] != ""), skill=d["skill"], gb=d["games_bucket"], w=d["deck_wins"], g=d["deck_games"],
                         qA=ga.mean(), t10A=np.sort(ga)[-10:].mean(), E=sum(c.get(n, 0) for n in EMP), Wc=sum(c.get(n, 0) for n in WAYS)))
    df = pd.DataFrame(rows); df = df[df.g >= 1].copy(); df["y"] = df.w / df.g
    df["Ec"] = df.E - 4; df["W1"] = (df.Wc >= 1).astype(float); df["W1xEc"] = df.W1 * df.Ec
    for k_, cnd in (("Wn1", df.Wc == 1), ("Wn2", df.Wc == 2), ("Wn3", df.Wc == 3), ("Wn4", df.Wc >= 4)): df[k_] = cnd.astype(int)
    P(f"\ndeck file: {len(df)} Premier decks; games_bucket values {df.gb.value_counts().to_dict()} (the deck's own game count, no games term used)")
    gm = gmask(df.skill.values)
    out = []
    def fit(sub, terms):
        f = "y ~ " + " + ".join(terms) + " + qA + t10A + splash + skill + C(lane)"
        m = smf.glm(f, sub, family=sm.families.Binomial(), freq_weights=sub.g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.draft)[0]})
        return m
    def contrast(m, w):
        v = np.zeros(len(m.params)); ix = list(m.params.index)
        for t, c in w.items(): v[ix.index(t)] = c
        return K * float(v @ m.params.values), K * float(np.sqrt(v @ m.cov_params().values @ v))
    specs = [("empower x Way (extra pts per empower card with a Way present)", "W1xEc + Ec + W1", {"W1xEc": 1}),
             ("empower, no Way (pts per card)", "W1xEc + Ec + W1", {"Ec": 1}),
             ("Way count 2 minus 1", "Wn1 + Wn2 + Wn3 + Wn4 + Ec", {"Wn2": 1, "Wn1": -1}),
             ("Way count 3 minus 2 (post hoc lead)", "Wn1 + Wn2 + Wn3 + Wn4 + Ec", {"Wn3": 1, "Wn2": -1}),
             ("Way count 4+ minus 3", "Wn1 + Wn2 + Wn3 + Wn4 + Ec", {"Wn4": 1, "Wn3": -1})]
    for g in GROUPS:
        sub = df[gm[g]]
        fits = {}
        for lab, f, w in specs:
            if f not in fits: fits[f] = fit(sub, f.split(" + "))
            e, s = contrast(fits[f], w); out.append(dict(test=lab, group=g, est=e, se=s, n=len(sub)))
    return out, df

def s3_within():
    EL = json.load(open(f"{R}/analysis/out/video_empower_lists.json"))
    wi = [nidx[n] for n in EL["ways"] if n in nidx]; ei = [nidx[n] for n in EL["empower"] if n in nidx]
    nE = H[:, ei].sum(1, dtype=np.int32).astype(float); nW = H[:, wi].sum(1, dtype=np.int32)
    wh = (nW > 0).astype(float); dW = (D[:, wi] > 0).sum(1)
    anyway_deck = dW > 0
    out = []
    for p in POPS:
        for g in GROUPS:
            m = MASKS[(p, g)]
            r = wfit(np.where(m & anyway_deck)[0], np.column_stack([nE * wh, nE, wh, ncard, Lh]), kcode, 1)   # W2 interaction, decks with >= 1 Way
            if r: out.append(dict(test="W2 within-deck: empower in hand x Way in hand (pts per card)", pop=p, group=g, est=r[0][0], bs=r[1][:, 0], n=r[2]))
            r = wfit(np.where(m & (dW >= 3))[0], np.column_stack([nW == 1, nW == 2, nW >= 3, ncard, Lh]).astype(float), kcode, 3)   # W4, decks with >= 3 Ways
            if r:
                out.append(dict(test="W4 within-deck: 3+ Ways in hand minus 2", pop=p, group=g, est=r[0][2] - r[0][1], bs=r[1][:, 2] - r[1][:, 1], n=r[2]))
                out.append(dict(test="W4 within-deck: 2 Ways in hand minus 1", pop=p, group=g, est=r[0][1] - r[0][0], bs=r[1][:, 1] - r[1][:, 0], n=r[2]))
    return out

def gw_line(rows, label, p=None):
    """one row per group + good minus weak from independent estimates (SE) or joint bootstrap (bs)"""
    g = {r["group"]: r for r in rows if r["test"] == label and (p is None or r.get("pop") == p)}
    se = lambda r: r["se"] if "se" in r else np.std(r["bs"], ddof=1)
    cells = [f"{g[k]['est']:+.2f} ({se(g[k]):.2f}) [{g[k]['n']}]" if k in g else "n/a" for k in GROUPS]
    if "good" in g and "weak" in g:
        d = g["good"]["est"] - g["weak"]["est"]
        s = np.std(g["good"]["bs"] - g["weak"]["bs"], ddof=1) if "bs" in g["good"] else np.hypot(se(g["good"]), se(g["weak"]))
        return cells, f"{d:+.2f} [{d-1.96*s:+.2f}, {d+1.96*s:+.2f}] z {d/s:+.1f}"
    return cells, "n/a"

def s3_report(dk, wi):
    P("\n### S3 deck level (foundations rule 10 spec a: quality = mean + top-10 GIH, skill and lane and splash controlled, no games term), WR points; [n decks]\n")
    P("| test | all-skill | good | average | weak | good minus weak [95% CI] |\n|---|---|---|---|---|---|")
    for lab in dict.fromkeys(r["test"] for r in dk):
        c, d = gw_line(dk, lab); P(f"| {lab} | " + " | ".join(c) + f" | {d} |")
    for p in POPS:
        P(f"\n### S3 within-deck, population {p} [n decks (draft x build)]; joint draft bootstrap\n")
        P("| test | all-skill | good | average | weak | good minus weak [95% CI] |\n|---|---|---|---|---|---|")
        for lab in dict.fromkeys(r["test"] for r in wi):
            c, d = gw_line(wi, lab, p); P(f"| {lab} | " + " | ".join(c) + f" | {d} |")

# ------------------------------------------------------------------ S4 payoff / enabler (deck level, themes)
def run_s4(df_all, THEMES, cards):
    import statsmodels.api as sm, statsmodels.formula.api as smf
    from scipy.special import expit, logit as lg
    TESTED = [t for t, v in THEMES.items() if v.get("payoff") or v.get("payoff_names") or v.get("payoff_min_cmc")]
    ORDER = ["jace", "graveyard"] + [t for t in TESTED if t not in ("jace", "graveyard")]
    BASE = .55
    def fit(d, terms, gbterm):
        d = d.copy(); d["L"] = d.g - d.w
        f = "w + L ~ q + skill" + (" + C(gb)" if gbterm else "") + (" + C(lane)" if d.lane.nunique() > 1 else "") + " + " + " + ".join(terms)
        m = smf.glm(f, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
        pts = lambda x: 100 * (expit(lg(BASE) + x) - BASE)
        return {t: dict(pts=pts(m.params[t]), se=(pts(m.params[t] + m.bse[t]) - pts(m.params[t] - m.bse[t])) / 2, n=int(d[t].sum())) for t in terms}
    res = []
    for gname, sub0, gbterm in (("good", df_all[df_all.gm_good], False), ("average", df_all[df_all.gm_average], False), ("weak", df_all[df_all.gm_weak], False),
                                ("all-skill", df_all, False), ("good + C(gb) (original spec)", df_all[df_all.gm_good], True)):
        for t in ORDER:
            th = THEMES[t]; s = sub0[sub0.lane.isin(th["pairs"])]
            if len(s) < 300: s = sub0
            s = s.copy(); Pn, En = s[f"P_{t}"], s[f"E_{t}"]
            for k, lab in ((1, "P1"), (2, "P2"), (3, "P3p")): s[lab] = ((Pn == k) if k < 3 else (Pn >= 3)).astype(int)
            terms = [x for x in ("P1", "P2", "P3p") if s[x].sum() >= 30]
            s["E02"], s["E35"], s["E6"] = [((Pn >= 1) & m).astype(int) for m in (En <= 2, (En >= 3) & (En <= 5), En >= 6)]
            try:
                if len(s) >= 150 and terms and (Pn == 0).sum() >= 30:
                    for k, v in fit(s, terms, gbterm).items(): res.append(dict(group=gname, test=f"{t}: payoff count {k} vs 0", **v))
                    et = [x for x in ("E02", "E35", "E6") if s[x].sum() >= 30]
                    for k, v in fit(s, et, gbterm).items(): res.append(dict(group=gname, test=f"{t}: payoffs present, enablers {k[1:]} vs no payoff", **v))
                w = s[Pn >= 1].drop(columns="E02")
                wt = [x for x in ("E35", "E6") if w[x].sum() >= 30 and s["E02"].sum() >= 30]
                if wt and len(w) >= 150:
                    w2 = s[Pn >= 1].copy()
                    for k, v in fit(w2.drop(columns="E02"), wt, gbterm).items(): res.append(dict(group=gname, test=f"{t}: within payoffs>=1, enablers {k[1:]} vs 0-2", **v))
            except Exception as ex: print("s4 fail", gname, t, ex, flush=True)
        print("S4 done", gname, flush=True)
    return res

def s4_report(res):
    tests = list(dict.fromkeys(r["test"] for r in res))
    P("\n### S4 theme deck-level (payoff_enabler.py design), WR points (SE) by group, no games term; good minus weak with independent SEs\n")
    P("| term | good + C(gb) (original spec) | all-skill | good | average | weak | good minus weak z |\n|---|---|---|---|---|---|---|")
    zs, rowsout = [], []
    for t in tests:
        g = {r["group"]: r for r in res if r["test"] == t}
        c = [f"{g[k]['pts']:+.2f} ({g[k]['se']:.2f}) [{g[k]['n']}]" if k in g else "n/a" for k in ("good + C(gb) (original spec)", "all-skill", "good", "average", "weak")]
        if "good" in g and "weak" in g:
            d = g["good"]["pts"] - g["weak"]["pts"]; s = np.hypot(g["good"]["se"], g["weak"]["se"]); zs.append((d / s, t, d, s)); dd = f"{d:+.2f} ({s:.2f}) z {d/s:+.1f}"
        else: dd = "n/a"
        P(f"| {t} | " + " | ".join(c) + f" | {dd} |")
    z = np.array([x[0] for x in zs]); n = len(z)
    P(f"\nS4 summary: {n} terms with good and weak both estimable; good minus weak |z|>1.96 in {(abs(z)>1.96).sum()} (expected {0.05*n:.1f}), |z|>2.58 in {(abs(z)>2.58).sum()} (expected {0.01*n:.1f}); mean z {z.mean():+.2f}; Bonferroni z {norm.isf(0.025/n):.2f}")
    P("largest: " + "; ".join(f"{t} {d:+.2f} (z {zz:+.1f})" for zz, t, d, s in sorted(zs, key=lambda x: -abs(x[0]))[:6]))
    for gname in ("good", "average", "weak"):
        zz = np.array([r["pts"] / r["se"] for r in res if r["group"] == gname and r["se"] > 0 and re.search(r"vs 0$|vs no payoff|vs 0-2", r["test"])])
        P(f"{gname}: {len(zz)} terms, |z|>1.96 in {(abs(zz)>1.96).sum()} (expected {0.05*len(zz):.1f}), >=Bonferroni ({norm.isf(0.025/len(zz)):.2f}) in {(abs(zz)>norm.isf(0.025/len(zz))).sum()}")

def s4_prep():
    THEMES = json.loads(open(f"{R}/companion/advisor_v2/sets/FRA/manual.json").read())["themes"]
    BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
    TESTED = [t for t, v in THEMES.items() if v.get("payoff") or v.get("payoff_names") or v.get("payoff_min_cmc")]
    front = lambda n: n.split(" // ")[0]
    cards = {}
    for c in ratings.values():
        n = front(c["name"])
        if n not in cards or (c.get("isprimarycard") and c.get("oracle_text") and not cards[n].get("oracle_text")): cards[n] = c
    def roles(card, th):
        text = re.sub(r"\([^)]*\)", "", (card.get("oracle_text") or "").lower()); ty = card.get("types") or []
        e = bool(th.get("enabler") and re.search(th["enabler"], text)) or any(t in ty for t in th.get("enabler_types", ()))
        p = bool(th.get("payoff") and re.search(th["payoff"], text)) or card["name"] in th.get("payoff_names", ()) or front(card["name"]) in th.get("payoff_names", ())
        if th.get("payoff_min_cmc"): p = p or ((card.get("cmc") or 0) >= th["payoff_min_cmc"] and all(t in ty for t in th.get("payoff_types", ())))
        return e, p
    def gih(card, lane):
        d = card.get("deck_colors", {})
        for k in (lane, "All Decks"):
            v = (d.get(k) or {}).get("gihwr")
            if v: return v
        return np.nan
    rows = []
    for d in json.loads(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz").read()):
        if not d["deck_games"] or d["deck_games"] < 1 or sum(d["counts"].values()) != 40: continue
        r = dict(draft=d["draft"], lane=d["lane"], skill=d["skill"], gb=str(d["games_bucket"]), w=d["deck_wins"], g=d["deck_games"])
        q, ns, acc = 0.0, 0, {t: [0, 0] for t in TESTED}
        for n, k in d["counts"].items():
            c = cards.get(front(n))
            if n in BASICS or (c and "Land" in (c.get("types") or [])): continue
            if not c: continue
            q += gih(c, d["lane"]) * k; ns += k
            for t in TESTED:
                e, p = roles(c, THEMES[t])
                if p: acc[t][0] += k
                elif e: acc[t][1] += k
        r["q"] = q / max(ns, 1)
        for t in TESTED: r[f"P_{t}"], r[f"E_{t}"] = acc[t]
        rows.append(r)
    df = pd.DataFrame(rows).dropna(subset=["q"])
    gm = gmask(df.skill.values)
    for g in ("good", "average", "weak"): df[f"gm_{g}"] = gm[g]
    return df, THEMES, cards

# ------------------------------------------------------------------ run
CLOSE = ["Kiora of Salt and Sand", "Way of the Cryomancer", "Way of the Cryomancer (list 2)"]
tests = build_tests()
P(f"within-deck tests available: {len(tests)}")
if MODE == "heldout":
    run_s1([t for t in tests if t[0] in CLOSE + ["Rescue Girl, First Responder", "Ghalta the Unstoppable"]])
    s1_table(); s1_with_table()
elif MODE == "close":
    run_s1([t for t in tests if t[0] in CLOSE])
    s1_table(); s1_with_table()
    dk, df = deck_tables()
    wi = s3_within()
    s3_report(dk, wi)
else:
    run_s1(tests)
    s1_table(); s1_with_table(); s1_summary()
    dk, df = deck_tables(); wi = s3_within(); s3_report(dk, wi)
    S2 = run_s2(); s2_table(S2)
    dfp, THEMES, cards = s4_prep()
    P(f"\npayoff/enabler deck table: {len(dfp)} decks; good {dfp.gm_good.sum()}, average {dfp.gm_average.sum()}, weak {dfp.gm_weak.sum()}")
    s4_report(run_s4(dfp, THEMES, cards))
open(OUT, "w").write("\n".join(lines))
P("done")
