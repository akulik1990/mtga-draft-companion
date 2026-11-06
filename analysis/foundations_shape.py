"""Workstream A, shape foundations: do the builder inputs fitted on "60%+ players' decks" move when the sample is restricted to the
leak-free games buckets (7-9) instead of 1-6? (Skill bucket includes the deck's own games, docs/review-tests-2026-11-06.md 1.1.)
Items: (1) tools/deck_targets.py per-pair targets, replicated exactly (nothing written); (2) TEMPO_CREATURE / WALKER_CREATURE
(analysis/tempo_substitution.py, walker_substitution.py); (3) splash and three-colour rules (tools/fit_splash.py), scratch refit and the shipped
rule's calibration; (4) share of deckmodel_top3's x3-weighted games from buckets 1-6.
games_bucket (1-9) exists only in analysis/out/decks_FRA.json.gz; the game CSV's user_n_games_bucket is 100/500 for every deck there. CSV decks are
joined to it by draft id. Groups for items 1, 3, 4: B7-9, B1-6 (both 100+ games players), LOW (CSV players with < 100 games, the leakiest, not in
decks_FRA), UNM (100+ games, draft not in decks_FRA). 0.5 card is a reporting flag only, not a decision rule.
Output: analysis/out/foundations_shape.txt. Run (minutes; refits are bootstrapped over drafts):
nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy p.pydantic p.requests p.fastapi])' -c python3 analysis/foundations_shape.py"""
import csv, gzip, hashlib, json, os, re, sys, time, warnings
from multiprocessing import get_context
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MTGA_COMPANION_DATA", f"{ROOT}/data")
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
import companion.advisor_v2.engine  # noqa: E402,F401  (registers the settings activate checks, as tools/deck_targets.py)
from companion.advisor_v2.composition import curve_slot, roles  # noqa: E402
from companion.advisor_v2.deckbuilder import SPLASH_INPUTS, THREE_INPUTS, splash_features, three_features  # noqa: E402
from companion.advisor_v2.lanes import LANES, castable  # noqa: E402

OUTF = f"{ROOT}/analysis/out/foundations_shape.txt"
FLAG = 0.5  # reporting flag for a shift, in cards; not a decision rule
BOOT = 200
lines = []
def P(s=""): print(s, flush=True); lines.append(s)
def groups(g): return {"B7-9": g == "B7-9", "B1-6": g == "B1-6", "LOW": g == "LOW", "UNM": g == "UNM"}

config.activate("FRA")
fitted = json.load(open(f"{ROOT}/companion/advisor_v2/sets/FRA/fitted.json"))
data = json.loads((DATA_DIR / "Sets" / "FRA_PremierDraft_All_Data.json").read_text())["card_ratings"]
decks_all = json.load(gzip.open(f"{ROOT}/analysis/out/decks_FRA.json.gz", "rt"))
GB = {}
for d in decks_all: GB[d["draft"]] = d["games_bucket"]


def grp(draft, nb):
    if nb < 100: return "LOW"
    if draft not in GB: return "UNM"
    return "B7-9" if GB[draft] >= 7 else "B1-6"


# ---- one pass over the game CSV ----
wrs = [((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0 for c in data.values()]
wrs = [w for w in wrs if w > 0]
by_role, slot, by = {}, {}, {}
for c in data.values():
    for n in (c["name"], c["name"].split(" // ")[0]):
        by_role.setdefault(n, roles(c)); slot.setdefault(n, curve_slot(c)); by.setdefault(n, c)
CURVE = ("1", "2", "3", "4", "5", "6+")
FIELDS = ("creatures", "removal", "two_drops", "five_plus", "lands") + CURVE
SKILLS = [0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68]  # tools/deck_model.py


def top_flag(sk):  # tools/deck_model.context_row: any SKILLS[1:] bucket >= 0.60 within 0.02
    return any(abs((sk or 0) - b) < 0.02 for b in SKILLS[1:] if b >= 0.60 - 1e-9)


t0 = time.time()
decks, seen, game_rows = [], set(), []
with gzip.open(DATA_DIR / "Temp" / "synergy" / "game_data_public.FRA.PremierDraft.csv.gz", "rt", newline="", encoding="utf-8") as fh:
    rd = csv.reader(fh); header = next(rd); col = {h: i for i, h in enumerate(header)}
    deck_cols = [(i, h[5:]) for i, h in enumerate(header) if h.startswith("deck_")]
    side_cols = {h[10:]: i for i, h in enumerate(header) if h.startswith("sideboard_")}
    for row in rd:
        sk_s, pair = row[col["user_game_win_rate_bucket"]], row[col["main_colors"]]
        nb_s = row[col["user_n_games_bucket"]]
        sk = float(sk_s) if sk_s not in ("", "None") else None
        nb = float(nb_s) if nb_s not in ("", "None") else 0
        did = row[col["draft_id"]]
        game_rows.append((did, nb, top_flag(sk), sk))
        if len(pair) not in (2, 3) or sk is None or sk < 0.60: continue
        key = (did, row[col["build_index"]])
        if key in seen: continue
        seen.add(key)
        vec = dict.fromkeys(FIELDS, 0.0); deck, pool = {}, {}
        for i, name in deck_cols:
            v = row[i]
            d = int(float(v)) if v and v != "0" else 0
            s = int(float(row[side_cols[name]] or 0)) if name in side_cols else 0
            if d:
                deck[name] = d
                for r in by_role.get(name, ()):
                    if r in vec: vec[r] += d
                if slot.get(name): vec[slot[name]] += d
            if d + s: pool[name] = d + s
        decks.append(dict(draft=did, lane="".join(sorted(pair, key="WUBRG".index)), splash=row[col["splash_colors"]], nb=nb, skill=sk,
                          h=int(hashlib.md5(did.encode()).hexdigest(), 16) % 1000, g=grp(did, nb), vec=vec, pool=pool, deck=deck))
D = pd.DataFrame(decks)
P(f"CSV pass {time.time() - t0:.0f}s: {len(game_rows)} games, {len(D)} distinct 60%+ builds (2-3 main colours)")
P("builds by group (all 60%+ players, 2-3 colours): " + ", ".join(f"{k} {v}" for k, v in D.g.value_counts().items()))
P("(B7-9 / B1-6 = 100+ games players split by decks_FRA games_bucket; LOW = < 100 games, the leakiest; UNM = 100+ games, draft not in decks_FRA)\n")


# ---- cluster-robust mean and difference ----
def cmean(x, cl):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return np.nan, np.nan, 0
    m = x.mean(); s = pd.Series(x - m).groupby(np.asarray(cl)).sum()
    return m, np.sqrt((s ** 2).sum()) / n, n


# ====================== ITEM 1 ======================
P("=" * 100); P("ITEM 1. Deck targets per pair (tools/deck_targets.py, min_skill 0.60), by games-bucket group\n");
shipped = fitted["decks"]["pairs"]
def ship(lane, f):
    s = shipped[lane]
    return s["curve"][f] if f in CURVE else s[f]
V = pd.DataFrame(list(D.vec));
for f in FIELDS: D[f] = V[f].values
D2 = D[D.lane.str.len() == 2].copy()
# reproduction check: all builds, per pair, vs shipped (rounded to 0.1)
bad = 0; tot = 0
for lane in sorted(D2.lane.unique()):
    if lane not in shipped: continue
    for f in FIELDS:
        tot += 1; bad += abs(round(D2[D2.lane == lane][f].mean(), 1) - ship(lane, f)) > 1e-9
P(f"Reproduction of fitted.json 'decks' from this loader (all 60%+ builds, per pair, rounded to 0.1): {tot - bad}/{tot} cells equal\n")

lanes = [l for l in sorted(D2.lane.unique()) if len(D2[D2.lane == l]) >= 30]
res1 = []
for lane in lanes:
    s = D2[D2.lane == lane]
    for f in FIELDS:
        a = s[s.g == "B7-9"]; b = s[s.g == "B1-6"]
        ma, sa, na = cmean(a[f], a.draft); mb, sb, nb_ = cmean(b[f], b.draft)
        sh = ship(lane, f)
        diff = ma - mb; sd = np.hypot(sa, sb)
        res1.append(dict(lane=lane, f=f, shipped=sh, n79=na, m79=ma, se79=sa, n16=nb_, m16=mb, se16=sb, diff=diff, sed=sd,
                         inside=(ma - 1.96 * sa - .05 <= sh <= ma + 1.96 * sa + .05),
                         low=D2[(D2.lane == lane) & (D2.g == "LOW")][f].mean(), nlow=int(((D2.lane == lane) & (D2.g == "LOW")).sum())))
R1 = pd.DataFrame(res1)
P("Per pair, key fields. shipped = fitted.json (0.1 rounding; 'inside' allows +-0.05). m79/m16 = mean per deck, SE clustered by draft.")
P("diff = B7-9 minus B1-6; z = diff/SE; FLAG = |diff| > 0.5 card (reporting only); LOW = mean of < 100 games players (n in last column)\n")
P("| pair | field | shipped | B7-9 mean (SE) n | B1-6 mean (SE) n | diff [95% CI] | z | in B7-9 CI | LOW mean (n) | flag |\n|---|---|---|---|---|---|---|---|---|---|")
for _, r in R1[R1.f.isin(["creatures", "removal", "two_drops", "five_plus", "lands"])].iterrows():
    z = r["diff"] / r.sed
    P(f"| {r.lane} | {r.f} | {r.shipped:.1f} | {r.m79:.2f} ({r.se79:.2f}) {r.n79} | {r.m16:.2f} ({r.se16:.2f}) {r.n16} | {r['diff']:+.2f} [{r['diff'] - 1.96 * r.sed:+.2f}, {r['diff'] + 1.96 * r.sed:+.2f}] | {z:+.1f} | {'yes' if r.inside else 'NO'} | {r.low:.2f} ({r.nlow}) | {'FLAG' if abs(r['diff']) > FLAG else ''} |")
P("\nCurve slots (cards per deck), same columns, compact:\n")
P("| pair | slot | shipped | B7-9 (SE) | B1-6 (SE) | diff | z | in CI | flag |\n|---|---|---|---|---|---|---|---|---|")
for _, r in R1[R1.f.isin(CURVE)].iterrows():
    z = r["diff"] / r.sed
    P(f"| {r.lane} | {r.f} | {r.shipped:.1f} | {r.m79:.2f} ({r.se79:.2f}) | {r.m16:.2f} ({r.se16:.2f}) | {r['diff']:+.2f} | {z:+.1f} | {'yes' if r.inside else 'NO'} | {'FLAG' if abs(r['diff']) > FLAG else ''} |")

P("\nSummary of item 1 (two-colour pairs with >= 30 builds in the table above)")
P(f"cells {len(R1)}; |diff| > {FLAG}: {int((R1['diff'].abs() > FLAG).sum())}; |z| > 1.96: {int((np.abs(R1['diff'] / R1.sed) > 1.96).sum())} (about {0.05 * len(R1):.0f} expected by chance, cells are correlated); shipped outside B7-9 95% CI: {int((~R1.inside).sum())}")
P("Flagged cells (|diff| > 0.5): " + "; ".join(f"{r.lane} {r.f} {r['diff']:+.2f} (z {r['diff'] / r.sed:+.1f}, n79 {r.n79})" for _, r in R1[R1['diff'].abs() > FLAG].iterrows()))
P("\nPair-adjusted shift (OLS field ~ B7-9 + C(pair), SE clustered by draft; B7-9 vs B1-6 builds of 100+ games players; sample in the fit):\n")
P("| field | shift B7-9 vs B1-6 [95% CI] | p | LOW vs B1-6 [95% CI] | p |\n|---|---|---|---|---|")
for f in FIELDS:
    d = D2[D2.g.isin(["B7-9", "B1-6", "LOW"])].copy(); d["x79"] = (d.g == "B7-9") * 1.0; d["xlow"] = (d.g == "LOW") * 1.0
    m = smf.ols(f"Q('{f}') ~ x79 + xlow + C(lane)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
    c = m.conf_int()
    P(f"| {f} | {m.params['x79']:+.2f} [{c.loc['x79', 0]:+.2f}, {c.loc['x79', 1]:+.2f}] | {m.pvalues['x79']:.3f} | {m.params['xlow']:+.2f} [{c.loc['xlow', 0]:+.2f}, {c.loc['xlow', 1]:+.2f}] | {m.pvalues['xlow']:.3f} |")

# Removal targets and REMOVAL_SLACK note
P("\nRemoval target by pair (the number REMOVAL_SLACK is measured against): shipped / B7-9 / B1-6 / LOW\n")
P("| pair | shipped | B7-9 | B1-6 | LOW | diff 7-9 minus 1-6 |\n|---|---|---|---|---|---|")
for lane in lanes:
    r = R1[(R1.lane == lane) & (R1.f == "removal")].iloc[0]
    P(f"| {lane} | {r.shipped:.1f} | {r.m79:.2f} | {r.m16:.2f} | {r.low:.2f} | {r['diff']:+.2f} |")
# power: smallest detectable shift on pooled field
P("")


# ====================== ITEM 2 ======================
P("=" * 100); P("ITEM 2. TEMPO_CREATURE 0.57 and WALKER_CREATURE 0.62: same regressions as analysis/tempo_substitution.py and walker_substitution.py\n"
                "(decks_FRA, skill >= 0.60, two-colour, no splash, 22-24 spells; creatures ~ count + C(lane) + spells, SE clustered by draft), by games bucket\n")
cr = data
byf = {c["name"].split(" // ")[0]: c for c in cr.values()}
TEMPO = {"Unsummon", "Icy Reception", "Countersculpt", "Fatehold Charm"}
WALKER = {n for n, c in byf.items() if "Planeswalker" in c["types"] or re.search(r"empower jace", c.get("oracle_text") or "", re.I)}
WALKER = {n for n in WALKER if "Creature" not in byf[n]["types"]}
REMOVAL = {n for n, c in byf.items() if "removal" in (c.get("tags") or [])}
rows = []
for d in decks_all:
    if d["skill"] < 0.60 or len(d["lane"]) != 2 or d["splash"]: continue
    cnt = dict(spells=0, creatures=0, tempo=0, removal=0, walkers=0)
    for n, k in d["counts"].items():
        f = n.split(" // ")[0]; c = byf.get(f)
        if not c or "Land" in c["types"]: continue
        cnt["spells"] += k; cnt["creatures"] += k * ("Creature" in c["types"]); cnt["tempo"] += k * (f in TEMPO)
        cnt["removal"] += k * (f in REMOVAL); cnt["walkers"] += k * (f in WALKER)
    if 22 <= cnt["spells"] <= 24:
        rows.append(dict(draft=d["draft"], lane=d["lane"], gb=d["games_bucket"], **cnt))
T = pd.DataFrame(rows)
def reg(df, y, x):
    m = smf.ols(f"{y} ~ {x} + C(lane) + spells", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})
    return m.params[x], *m.conf_int().loc[x], m
P("| rule | sample | decks | cards per deck (mean, share 1+) | coefficient [95% CI] | credit by the rule (CI excludes 0) | shipped in CI |\n|---|---|---|---|---|---|---|")
out2 = {}
for name, y, x, ship_v, two_sided in (("TEMPO creatures", "creatures", "tempo", 0.57, True), ("TEMPO removal", "removal", "tempo", None, True), ("WALKER creatures", "creatures", "walkers", 0.62, False)):
    for lab, sel in (("all (as shipped)", T.gb > 0), ("B7-9", T.gb >= 7), ("B1-6", T.gb <= 6)):
        d = T[sel]; b, lo, hi, m = reg(d, y, x)
        credit = (-b if (hi < 0 or lo > 0) else 0.0) if two_sided else (-b if hi < 0 else 0.0)
        out2[(name, lab)] = (b, lo, hi)
        ins = "" if ship_v is None else ("yes" if lo <= -ship_v <= hi else "NO")
        P(f"| {name} | {lab} | {len(d)} | {d[x].mean():.2f}, {100 * (d[x] > 0).mean():.0f}% | {b:+.3f} [{lo:+.3f}, {hi:+.3f}] | {credit:.2f} | {ins} |")
    d = T[T.gb <= 9].copy(); d["h"] = (d.gb >= 7) * 1.0
    m = smf.ols(f"{y} ~ {x}*h + C(lane) + spells", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
    c = m.conf_int().loc[f"{x}:h"]
    P(f"|   interaction {name} | B7-9 minus B1-6 | | | {m.params[f'{x}:h']:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] (p {m.pvalues[f'{x}:h']:.2f}) | | |")
P("\n(shipped credit = minus the coefficient: TEMPO 0.57 -> coefficient -0.57; WALKER 0.62 -> -0.62.)\n")


# ====================== ITEM 3 ======================
P("=" * 100); P("ITEM 3. Splash and three-colour rules (tools/fit_splash.py): sample = 60%+ builds in draft-hash buckets 300-999 (as the tool), no 8000 cap here\n")
S = D[D.h >= 300].copy()
S["core"] = None
LANE_OF = lambda r: r.lane
def feats(item):
    lane, splash, deck, pool = item
    cards = [by[n] for n, k in pool.items() if n in by for _ in range(k)]
    spells = [by[n] for n in deck if n in by and "Land" not in (by[n].get("types") or [])]
    core = lane if len(lane) == 2 else max(LANES, key=lambda pair: sum(castable(c, pair) for c in spells))
    x3 = three_features(cards, core)
    x = splash_features(cards, lane) if len(lane) == 2 else None
    return x, x3
t0 = time.time()
with get_context("fork").Pool(12) as pl:
    FX = pl.map(feats, list(zip(S.lane, S.splash, S.deck, S.pool)), chunksize=200)
P(f"features for {len(S)} builds in {time.time() - t0:.0f}s")
S["x"] = [f[0] for f in FX]; S["x3"] = [f[1] for f in FX]
S["y3"] = (S.lane.str.len() == 3) * 1.0
S2 = S[S.lane.str.len() == 2].copy(); S2["y"] = (S2.splash != "") * 1.0
rule, rule3 = fitted["splash"], fitted["three"]
def mat(rows, inputs, names):
    ix = [names.index(k) for k in inputs]; return np.array([[r[i] for i in ix] for r in rows], float)
X = mat(S2.x, rule["inputs"], list(SPLASH_INPUTS)); X3 = mat(S.x3, rule3["inputs"], list(THREE_INPUTS))
def sig(z): return 1 / (1 + np.exp(-z))
def predict(rule, X): return sig(np.c_[np.ones(len(X)), (X - np.array(rule["mu"])) / np.array(rule["sd"])] @ np.array(rule["w"]))
def fit_std(X, y, rule):  # same ridge logistic as fit_splash.fit, standardised with the SHIPPED mu/sd so coefficients are comparable
    Z = np.c_[np.ones(len(X)), (X - np.array(rule["mu"])) / np.array(rule["sd"])]
    def nll(w):
        q = np.clip(sig(Z @ w), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)) + 1e-3 * w[1:] @ w[1:]
    return minimize(nll, np.array(rule["w"]), method="L-BFGS-B").x
def acc(rule, X, y): return float(((predict(rule, X) > .5) == (y > .5)).mean())

def boot(X, y, cl, rule, n=BOOT, seed=0):
    rng = np.random.default_rng(seed); u, inv = np.unique(cl, return_inverse=True)
    ix = [np.where(inv == k)[0] for k in range(len(u))]; ws = []
    for _ in range(n):
        s = np.concatenate([ix[k] for k in rng.integers(0, len(u), len(u))])
        ws.append(fit_std(X[s], y[s], rule))
    return np.array(ws)

def block(title, Sx, X, y, rule, names, extra_by=None):
    P(f"\n--- {title} ---")
    g = Sx.g.values; cl = Sx.draft.values
    pr = predict(rule, X)
    P("Shipped rule's calibration on these builds (predicted mean vs observed rate; accuracy at 0.5):\n")
    P("| group | builds | observed rate | shipped rule predicts | obs - pred [95% CI, clustered] | accuracy |\n|---|---|---|---|---|---|")
    for gname in ("B7-9", "B1-6", "LOW", "UNM"):
        m = g == gname
        if m.sum() < 30: continue
        e, se, n = cmean(y[m] - pr[m], cl[m])
        P(f"| {gname} | {n} | {y[m].mean():.3f} | {pr[m].mean():.3f} | {e:+.3f} [{e - 1.96 * se:+.3f}, {e + 1.96 * se:+.3f}] | {acc(rule, X[m], y[m]):.3f} |")
    P("\nScratch refit (fixed spec = shipped inputs, standardised with the shipped mu/sd; ridge 1e-3 as the tool), per-SD coefficients, bootstrap over drafts (%d):\n" % BOOT)
    P("| group | n | " + " | ".join(["intercept"] + names) + " |\n|---|---|" + "---|" * (len(names) + 1))
    P("| shipped | %d | " % rule["n"] + " | ".join(f"{v:+.2f}" for v in rule["w"]) + " |")
    ws = {}
    for gname in ("ALL", "B7-9", "B1-6"):
        m = np.ones(len(y), bool) if gname == "ALL" else g == gname
        w = fit_std(X[m], y[m], rule); ws[gname] = (w, boot(X[m], y[m], cl[m], rule, seed=1))
        lo, hi = np.percentile(ws[gname][1], [2.5, 97.5], axis=0)
        P(f"| {gname} | {m.sum()} | " + " | ".join(f"{a:+.2f} [{l:+.2f}, {h:+.2f}]" for a, l, h in zip(w, lo, hi)) + " |")
    lo, hi = np.percentile(ws["B7-9"][1], [2.5, 97.5], axis=0)
    inside = [(l <= s <= h) for s, l, h in zip(rule["w"], lo, hi)]
    P("\nshipped weight inside the B7-9 bootstrap 95% CI: " + ", ".join(f"{n} {'yes' if i else 'NO'}" for n, i in zip(["intercept"] + names, inside)))
    # difference B7-9 vs B1-6 bootstrap (independent groups)
    d = ws["B7-9"][0] - ws["B1-6"][0]; dsd = np.sqrt(ws["B7-9"][1].var(0) + ws["B1-6"][1].var(0))
    P("B7-9 minus B1-6 (per SD): " + ", ".join(f"{n} {a:+.2f} (z {a / s:+.1f})" for n, a, s in zip(["intercept"] + names, d, dsd)))
    # swapped calibration: each group's own fit applied to the other
    for a, b in (("B7-9", "B1-6"), ("B1-6", "B7-9")):
        pa = sig(np.c_[np.ones((g == b).sum()), (X[g == b] - np.array(rule["mu"])) / np.array(rule["sd"])] @ ws[a][0])
        P(f"{a}-fitted rule on {b} builds: accuracy {float(((pa > .5) == (y[g == b] > .5)).mean()):.3f} vs shipped rule {acc(rule, X[g == b], y[g == b]):.3f} vs {b}-fitted in-sample {float(((sig(np.c_[np.ones((g == b).sum()), (X[g == b] - np.array(rule['mu'])) / np.array(rule['sd'])] @ ws[b][0]) > .5) == (y[g == b] > .5)).mean()):.3f}")
    return ws

P("Splash rule (two-colour builds; y = splashed). Shipped: n %d, rate %.3f, held-out accuracy %.3f, inputs %s" % (rule["n"], rule["rate"], rule["heldout_accuracy"], rule["inputs"]))
ws_s = block("splash", S2, X, S2.y.values, rule, rule["inputs"])
P("\nObserved splash rate by pair and group (60%+ two-colour builds, hash >= 300; pair splash rate = share with a splash colour; SE clustered by draft):\n")
P("| pair | B7-9 rate (n) | B1-6 rate (n) | diff [95% CI] | LOW rate (n) | shipped rule predicts B7-9 / B1-6 |\n|---|---|---|---|---|---|")
pr_all = predict(rule, X)
for lane in sorted(S2.lane.unique()):
    row = S2.lane == lane
    cells = {}
    for gname in ("B7-9", "B1-6", "LOW"):
        m = (row & (S2.g == gname)).values; cells[gname] = cmean(S2.y[m], S2.draft[m])
    if cells["B7-9"][2] < 15: continue
    a, b = cells["B7-9"], cells["B1-6"]; dd = a[0] - b[0]; ds = np.hypot(a[1], b[1])
    pa = pr_all[(row & (S2.g == "B7-9")).values].mean(); pb = pr_all[(row & (S2.g == "B1-6")).values].mean()
    P(f"| {lane} | {a[0]:.3f} ({a[2]}) | {b[0]:.3f} ({b[2]}) | {dd:+.3f} [{dd - 1.96 * ds:+.3f}, {dd + 1.96 * ds:+.3f}] | {cells['LOW'][0]:.3f} ({cells['LOW'][2]}) | {pa:.3f} / {pb:.3f} |")
m79, m16 = (S2.g == "B7-9").values, (S2.g == "B1-6").values
a, b = cmean(S2.y[m79], S2.draft[m79]), cmean(S2.y[m16], S2.draft[m16])
P(f"pooled: B7-9 {a[0]:.3f}, B1-6 {b[0]:.3f}, diff {a[0] - b[0]:+.3f} (SE {np.hypot(a[1], b[1]):.3f}); LOW {S2.y[(S2.g == 'LOW').values].mean():.3f}")
# splash-rate shift pair-adjusted
d = S2[S2.g.isin(["B7-9", "B1-6"])].copy(); d["x79"] = (d.g == "B7-9") * 1.0
m = smf.ols("y ~ x79 + C(lane)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.draft)[0]})
P(f"pair-adjusted splash-rate shift B7-9 minus B1-6 (linear prob.): {m.params['x79']:+.3f} [{m.conf_int().loc['x79', 0]:+.3f}, {m.conf_int().loc['x79', 1]:+.3f}]")

P("\nThree-colour rule (builds with 2 or 3 main colours; y = three main colours). Shipped: n %d, rate %.3f, held-out accuracy %.3f, inputs %s" % (rule3["n"], rule3["rate"], rule3["heldout_accuracy"], rule3["inputs"]))
ws_t = block("three-colour", S, X3, S.y3.values, rule3, rule3["inputs"])
P("\nObserved three-colour share by group: " + ", ".join(f"{g_} {S.y3[S.g == g_].mean():.3f} (n {int((S.g == g_).sum())})" for g_ in ("B7-9", "B1-6", "LOW", "UNM")))


# ====================== ITEM 4 ======================
P("\n" + "=" * 100); P("ITEM 4. deckmodel_top3 (x3 weight for 60%+ games): where do the weighted games come from?\n")
G = pd.DataFrame(game_rows, columns=["draft", "nb", "top", "sk"])
G["g"] = [grp(d, nb) for d, nb in zip(G.draft, G.nb)]
G["w"] = np.where(G.top, 3.0, 1.0)
tab = G.groupby("g").agg(games=("w", "size"), top_games=("top", "sum"), weight=("w", "sum"))
tab["top weight (x3)"] = 3 * tab.top_games
P(f"all CSV games {len(G)}; top-flagged (context-dummy rule of tools/deck_model.py: skill within 0.02 of 0.60, 0.64 or 0.68) {int(G.top.sum())}")
P("| group | games | top games | share of top games | share of top weight (3x) | share of total weight |\n|---|---|---|---|---|---|")
for k, r in tab.iterrows():
    P(f"| {k} | {int(r.games)} | {int(r.top_games)} | {r.top_games / tab.top_games.sum():.3f} | {r['top weight (x3)'] / tab['top weight (x3)'].sum():.3f} | {r.weight / tab.weight.sum():.3f} |")
t100 = tab.loc[[x for x in ("B7-9", "B1-6", "UNM") if x in tab.index], "top_games"]
P(f"\nTop-weighted games from players with 100+ games: B1-6 share {tab.loc['B1-6', 'top_games'] / t100.sum():.3f}, B7-9 {tab.loc['B7-9', 'top_games'] / t100.sum():.3f} (UNM included in denominator)")
P(f"Games with skill >= 0.70 (the dummy rule gives them no top flag, so no x3): {int((G.sk >= 0.70 - 1e-9).sum())} of {len(G)}, {int(((G.sk >= 0.70 - 1e-9) & (G.sk.notna())).sum())} counted; shown for the doc")
D6 = D.copy()
P(f"builds at skill >= 0.70 among 60%+ builds: {int((D6.skill >= 0.70 - 1e-9).sum())} of {len(D6)}")
open(OUTF, "w").write("\n".join(lines) + "\n")
P(f"\nwrote {OUTF}")
