"""Video splash tests H3 (mqZ-40) and H2 (mqZ-44), deck level, FRA. Output analysis/out/video_splash_deck.txt.
Design fixed before any outcome was looked at (docs/video-test-splash-FRA.md):
  Population: two-colour-lane decks (len(lane)==2), >=1 game, >=20 priced spells (video_counters.load rule), all formats; decks_FRA.json.gz (100+ games players).
  Splash cards of a deck: non-land, non-landcycler spells with a plain pip (front face, hybrid not counted) of a flagged splash colour (B5 rule). Distinct = names, copies = sum of counts.
  Sources: lands in the 40 that make the splash colour (basics incl. Snow-Covered, any land with the colour, Room of Refuge any), summed over splash colours = "lands only" (primary, B5 conclusion);
           effective = lands + all landcyclers + Murmuring Volume (video_dpsplash rule; sensitivity).
  Model (binomial GLM on wins, games as weights, SEs clustered by draft, effects in WR points = 24.75 x logit coef; no game-count term):
     y ~ C(cat) + q_base + sg + src + skill + C(lane) + C(fmt)   [H3 adds nothing; H2 adds clipped total splash copies dummies]
     q_base = mean lane GIH WR of the non-splash priced spells; sg = mean All-Decks GIH WR of the splash cards; src = splash land sources (sg and src centred on splash decks, 0 for clean pairs).
  H3: single-colour-splash decks; cat = clean (no flag) / s0 (flag, no card) / s1 / s2 / s3+ distinct splash cards. Primary: s2 - s1. Copies version: c1/c2/c3+ copies.
  H2: cat = clean / one splash colour / 2+ splash colours. Primary: 2+ minus 1, total splash copies held fixed.
  Groups by the player's skill bucket (good >=0.60, avg 0.52-0.58, weak <0.52); good - weak independent groups. Halves: drafts by md5(draft) parity. Bonferroni over 3 primaries: |z| 2.39.
Run: systemd-run --user --scope -q -p MemoryMax=4G nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_splash_deck.py"""
import gzip, hashlib, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"
sys.path.insert(0, f"{R}/analysis")
from video_counters import cards
K = 100 * .55 * .45
lines = []
def P(s=""): print(s, flush=True); lines.append(s)

by = cards(); COL = "WUBRG"; BASIC = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
gA = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
info = {}
for n, v in by.items():
    front = (v.get("mana_cost") or "").split("//")[0]; t = v.get("oracle_text") or ""
    isl = "Land" in v["types"]
    mk = set()
    if isl:
        b = n.replace("Snow-Covered ", "")
        mk = set(COL) if n == "Room of Refuge" else {BASIC[b]} if b in BASIC else {c for c in (v.get("colors") or []) if c in COL}
    info[n] = dict(land=isl, pips={c: len(re.findall(r"\{%s\}" % c, front)) for c in COL}, cyc=bool(re.search(r"landcycling", t, re.I)), makes=mk, ga=gA(v))

rows = []
for d in json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt")):
    lane, S = d["lane"], d["splash"]
    if len(lane) != 2 or d["deck_games"] < 1 or set(S) & set(lane): continue
    q, sgl, nd, nc, srcl, srce = [], [], 0, 0, 0, 0
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; i = info.get(n); v = by.get(n)
        if i is None: continue
        if i["land"]:
            srcl += k * sum(1 for c in S if c in i["makes"]); srce += k * sum(1 for c in S if c in i["makes"])
            continue
        if i["cyc"] or n == "Murmuring Volume": srce += k * len(S) if S else 0
        isp = (not i["cyc"]) and any(i["pips"][c] >= 1 for c in S)
        if isp: nd += 1; nc += k; sgl += [i["ga"]] * k
        g = (v["deck_colors"].get(lane) or {}).get("gihwr") or i["ga"]
        if g and not isp: q += [g] * k
    if len(q) + nc < 20 or len(q) == 0: continue
    rows.append(dict(draft=d["draft"], lane=lane, fmt=d["format"], S=S, ncol=len(S), skill=d["skill"], y=d["deck_wins"] / d["deck_games"], g=d["deck_games"],
                     qb=np.mean(q), qall=np.mean(q + [x for x in sgl if x]), sg=np.mean([x for x in sgl if x]) if any(sgl) else np.nan, nd=nd, nc=nc, srcl=srcl, srce=srce))
DF = pd.DataFrame(rows)
DF["half"] = DF.draft.map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 2)
def grp(df, k):
    s = df.skill
    return df if k == "all" else df[s >= .6 - 1e-9] if k == "good" else df[(s >= .52 - 1e-9) & (s <= .58 + 1e-9)] if k == "avg" else df[s < .52 - 1e-9]
GR = ["all", "good", "avg", "weak"]
P(f"decks {len(DF)} (two-colour lane): splash colours 0/1/2/3 = {[(DF.ncol == i).sum() for i in range(4)]}; formats {DF.fmt.value_counts().to_dict()}")
one = DF[DF.ncol == 1]
P(f"one-colour splash decks by distinct splash cards 0/1/2/3+: {[int((one.nd == i).sum()) if i < 3 else int((one.nd >= 3).sum()) for i in range(4)]}; by copies 0/1/2/3+: {[int((one.nc == i).sum()) if i < 3 else int((one.nc >= 3).sum()) for i in range(4)]}")
P(f"skill groups (all 2-colour decks) good/avg/weak: {[len(grp(DF, k)) for k in GR[1:]]}; other skill values (between groups, in 'all' only): {int(len(DF) - sum(len(grp(DF, k)) for k in GR[1:]))}")

def fit(df, f):
    G = pd.factorize(df.draft)[0]
    return smf.glm("y ~ " + f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": G})
def con(m, w):
    names = list(m.params.index)
    try:
        v = np.zeros(len(names))
        for t, c in w.items(): v[names.index(t)] = c
        b = float(v @ m.params.values); se = float(np.sqrt(v @ m.cov_params().values @ v)); return K * b, K * se
    except ValueError: return np.nan, np.nan
def fmt(e): return "n/a" if np.isnan(e[0]) else f"{e[0]:+.2f} ({e[1]:.2f}) [{e[0]-1.96*e[1]:+.2f}, {e[0]+1.96*e[1]:+.2f}] z {e[0]/e[1]:+.2f}"
T = lambda c: f"C(cat)[T.{c}]"

def prep(df, kind):
    df = df.copy()
    if kind == "H3":
        d = df[df.ncol <= 1].copy()
        d["cat"] = np.where(d.ncol == 0, "clean", np.select([d.nd == 0, d.nd == 1, d.nd == 2], ["s0", "s1", "s2"], "s3p"))
    elif kind == "H3c":
        d = df[df.ncol <= 1].copy()
        d["cat"] = np.where(d.ncol == 0, "clean", np.select([d.nc == 0, d.nc == 1, d.nc == 2], ["s0", "c1", "c2"], "c3p"))
    else:
        d = df.copy(); d["cat"] = np.where(d.ncol == 0, "clean", np.where(d.ncol == 1, "one", "two"))
        d["ncl"] = d.nc.clip(0, 5)
    on = d.ncol > 0
    d["sgc"] = np.where(on & d.sg.notna(), d.sg - d.sg[on & d.sg.notna()].mean(), 0.0)
    for c in ("srcl", "srce"): d[c + "c"] = np.where(on, d[c] - d[c][on].mean(), 0.0)
    order = {"H3": ["clean", "s0", "s1", "s2", "s3p"], "H3c": ["clean", "s0", "c1", "c2", "c3p"], "H2": ["clean", "one", "two"]}[kind]
    d["cat"] = pd.Categorical(d.cat, categories=order); d["cat"] = d["cat"].cat.remove_unused_categories()
    return d
def formula(kind, qual="base", src="srclc"):
    q = "qb + sgc" if qual == "base" else "qall"
    f = f"C(cat) + {q} + {src} + skill + C(lane) + C(fmt)"
    return f + (" + C(ncl)" if kind == "H2" else "")

def run(kind, terms, label, qual="base", src="srclc", fmts=None, tag=""):
    """terms: [(label, {term: coef})]; returns {group: [est..]} printing a table."""
    base = DF if fmts is None else DF[DF.fmt.isin(fmts)]
    out = {}
    cells = {k: grp(base, k) for k in GR}
    cells["half 0"] = base[base.half == 0]; cells["half 1"] = base[base.half == 1]
    for k, sub in cells.items():
        d = prep(sub, kind); m = None
        try: m = fit(d, formula(kind, qual, src))
        except Exception as e: print("fail", k, e)
        out[k] = ([con(m, w) if m is not None else (np.nan, np.nan) for _, w in terms], d)
    P(f"\n**{label}**{tag} (WR points, SE, 95% CI, z)\n")
    P("| group | decks (drafts) | " + " | ".join(l for l, _ in terms) + " |\n|---|---|" + "---|" * len(terms))
    for k, (es, d) in out.items():
        P(f"| {k} | {len(d)} ({d.draft.nunique()}) | " + " | ".join(fmt(e) for e in es) + " |")
    g, w = out["good"][0], out["weak"][0]
    P("| good minus weak | | " + " | ".join(fmt(((a[0] - b[0]), np.hypot(a[1], b[1]))) for a, b in zip(g, w)) + " |")
    return out

def counts(kind):
    d = prep(DF, kind)
    P("cell sizes (decks) per skill group: " + "; ".join(f"{k}: " + ", ".join(f"{c} {int((grp(d, k).cat == c).sum())}" for c in d.cat.cat.categories) for k in GR))

P("\n# H3 (mqZ-40): one-colour splash, distinct splash cards 2 vs 1\n")
counts("H3")
terms3 = [("**s2 - s1 (PRIMARY)**", {T("s2"): 1, T("s1"): -1}), ("s1 - clean", {T("s1"): 1}), ("s2 - clean", {T("s2"): 1}), ("s3+ - s1", {T("s3p"): 1, T("s1"): -1}), ("s0 (flag, no card) - clean", {T("s0"): 1})]
r3 = run("H3", terms3, "H3 distinct cards, lands-only sources")
run("H3", terms3[:2] + terms3[3:4], "sensitivity: effective sources", src="srcec")
run("H3", terms3[:2], "sensitivity: whole-deck quality only (no separate splash GIH)", qual="all")
run("H3", terms3[:2], "sensitivity: Premier Draft only", fmts=["PremierDraft"])
P("\nCopies instead of distinct cards:")
counts("H3c")
run("H3c", [("**c2 - c1 copies**", {T("c2"): 1, T("c1"): -1}), ("c1 - clean", {T("c1"): 1}), ("c3+ - c1", {T("c3p"): 1, T("c1"): -1})], "H3 copies")

P("\n# H2 (mqZ-44): base pair + 1 splash colour vs 2+ splash colours (total splash copies controlled)\n")
counts("H2")
terms2 = [("**2+ colours - 1 colour (PRIMARY)**", {T("two"): 1, T("one"): -1}), ("1 colour - clean", {T("one"): 1}), ("2+ colours - clean", {T("two"): 1})]
r2 = run("H2", terms2, "H2 total copies dummies, lands-only sources")
run("H2", terms2[:1], "sensitivity: effective sources", src="srcec")
run("H2", terms2[:1], "sensitivity: whole-deck quality only", qual="all")
run("H2", terms2[:1], "sensitivity: Premier Draft only", fmts=["PremierDraft"])
d2 = prep(DF, "H2"); sp = d2[d2.ncol > 0]
P("\nH2 descriptives (splash decks): " + "; ".join(f"{c}: n {int((sp.cat == c).sum())}, copies {sp[sp.cat == c].nc.mean():.2f}, distinct {sp[sp.cat == c].nd.mean():.2f}, lands-only sources {sp[sp.cat == c].srcl.mean():.2f}, base GIH {sp[sp.cat == c].qb.mean():.2f}, skill {sp[sp.cat == c].skill.mean():.3f}" for c in ("one", "two")))
d3 = prep(DF, "H3"); sp = d3[d3.ncol > 0]
P("H3 descriptives (one-colour splash): " + "; ".join(f"{c}: n {int((sp.cat == c).sum())}, copies {sp[sp.cat == c].nc.mean():.2f}, sources {sp[sp.cat == c].srcl.mean():.2f}, splash-card GIH {sp[sp.cat == c].sg.mean():.2f}, base GIH {sp[sp.cat == c].qb.mean():.2f}, skill {sp[sp.cat == c].skill.mean():.3f}, raw WR {np.average(sp[sp.cat == c].y, weights=sp[sp.cat == c].g)*100:.2f}" for c in ("s1", "s2", "s3p")))
open(f"{OUT}/video_splash_deck.txt", "w").write("\n".join(lines) + "\n")
