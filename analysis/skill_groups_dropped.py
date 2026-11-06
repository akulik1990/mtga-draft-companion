"""Dropped ideas re-checked by player skill group (FRA). Doc: docs/skill-groups-dropped-FRA.md.
Groups by the player's skill bucket (never by the deck's result): good >= 0.60, average [0.52, 0.60), weak < 0.52, plus 'all' (every 100+ games player, the corrected
all-player estimate). Deck level: binomial GLM on wins/losses (or proportion with deck-game freq weights, as each original), SEs clustered by draft, skill as a control inside the
group, NO game-count term (games_bucket is the deck's own game count). Effects in win-rate points = 24.75 x logit coefficient (linear at 55%). Good-minus-weak = difference of two
independent fits (disjoint players, hence disjoint drafts), SE = sqrt(se1^2 + se2^2).
Originals are re-used by executing their source with two textual patches (skill filter off, ' + C(gb)' off); every patch is asserted, so a changed original fails loudly.
Pre-registered verdict rules (fixed before any output): nominal p < 0.05 per group; Bonferroni lines 0.05/N (good-minus-weak) and 0.05/(3N) (a group), N = number of rows.

  python3 analysis/skill_groups_dropped.py cache          # b3b8 game cache (own npz), streams the game CSV in 20k-row chunks
  python3 analysis/skill_groups_dropped.py misc           # deck-level tests, all groups
  python3 analysis/skill_groups_dropped.py b3b8 <group>   # B3 / B8 via the patched b3b8 script, group in all|good|avg|weak
  python3 analysis/skill_groups_dropped.py older          # older (11-05) extremes and finishers, simplified deck-level GLM
  python3 analysis/skill_groups_dropped.py report         # merges parts -> analysis/out/skill_groups_dropped.txt
Run each as: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 ...
"""
import contextlib, csv, gzip, io, json, os, re, sys, warnings, zlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); A = f"{R}/analysis"; OUT = f"{A}/out"; PARTS = f"{OUT}/skill_groups_parts"
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
K = 24.75
GROUPS = {"all": (0.0, 9.0), "good": (0.60, 9.0), "avg": (0.52, 0.60), "weak": (0.0, 0.52)}
os.makedirs(PARTS, exist_ok=True); sys.path.insert(0, R); sys.path.insert(0, A)

def sel(df, g, col="skill"):
    lo, hi = GROUPS[g]; return df[(df[col] >= lo - 1e-9) & (df[col] < hi - 1e-9)]

@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()): yield

def run_src(path, cut, subs, extra=None):
    """exec the original script's source up to the line starting with `cut`, after asserted textual patches; returns its namespace"""
    src = open(path).read()
    for a, b in subs:
        assert a in src, (path, a); src = src.replace(a, b)
    assert cut in src, (path, cut); src = src[:src.index(cut)]
    ns = {"__name__": "x", "__file__": path, **(extra or {})}
    with quiet(): exec(compile(src, path, "exec"), ns)
    return ns

def contrast(m, w):
    n = list(m.params.index); v = np.zeros(len(n))
    for t, c in w.items(): v[n.index(t)] = c
    return K * float(v @ m.params.values), K * float(np.sqrt(v @ m.cov_params().values @ v))

def glm_prop(df, f, wcol="g"):
    return smf.glm(f, df, family=sm.families.Binomial(), freq_weights=df[wcol]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})
def glm_cb(df, f):
    return smf.glm(f, df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})

def by_group(store, tid, df, fn, col="skill"):
    """fn(sub) -> {row: (est, se)}; store[tid|row][group] = [est, se, n]"""
    for g in GROUPS:
        sub = sel(df, g, col)
        try: out = fn(sub)
        except Exception as e: print("FAIL", tid, g, repr(e)[:120], flush=True); out = {}
        for row, v in out.items():
            store.setdefault(f"{tid}|{row}", {})[g] = [None if v is None else float(v[0]), None if v is None else float(v[1]), int(len(sub))]
        print(tid, g, len(sub), {k: (None if v is None else round(v[0], 2)) for k, v in out.items()}, flush=True)

def save(name, d): json.dump(d, open(f"{PARTS}/{name}.json", "w"), indent=0)

# ---------------------------------------------------------------------------------------------------------------- misc
def misc():
    import video_counters as vc, video_sinks as vs
    S = {}
    # ---- B2 top end (podcast_topend.py design; own loader because the original has no import-safe split)
    raw = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
    cards = {}
    for v in raw.values():
        n = v["name"].split(" // ")[0]; a = v["deck_colors"]["All Decks"]
        if n not in cards or a["samples"] > cards[n]["s"]: cards[n] = dict(s=a["samples"], land="Land" in v["types"], cmc=v["cmc"], g=a["gihwr"])
    rows = []
    for d in json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt")):
        sp = [(cards[n], k) for n, k in d["counts"].items() if n in cards and not cards[n]["land"] and cards[n]["g"] > 0]
        if not sp or d["deck_games"] < 1: continue
        g = np.repeat([c["g"] for c, k in sp], [k for c, k in sp])
        rows.append(dict(draft=d["draft"], lane=d["lane"], splash=int(d["splash"] != ""), skill=d["skill"], w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"],
                         mean_g=g.mean(), top10=np.sort(g)[-10:].mean(), big=sum(k for c, k in sp if c["cmc"] >= 5), c7=sum(k for c, k in sp if c["cmc"] >= 7)))
    TE = pd.DataFrame(rows); TE["bucket"] = np.select([TE.big <= 1, TE.big == 2, TE.big == 3, TE.big == 4], ["b01", "b2", "b3", "b4"], "b5p")
    def f_top(s):
        m = glm_cb(s, "I(w)+I(l) ~ C(bucket, Treatment('b2')) + c7 + mean_g + top10 + skill + splash + C(lane)")
        nm = lambda b: f"C(bucket, Treatment('b2'))[T.{b}]"
        return {"B2 cmc>=5 count 0-1 vs 2": contrast(m, {nm("b01"): 1}), "B2 cmc>=5 count 5+ vs 2": contrast(m, {nm("b5p"): 1}), "B2 per cmc>=7 card (beyond bucket)": contrast(m, {"c7": 1})}
    by_group(S, "topend", TE, f_top); del TE
    # ---- B4 non-board (podcast_nonboard.py source, skill filter off)
    ns = run_src(f"{A}/podcast_nonboard.py", 'df = pd.DataFrame(rows)', [('d["skill"] < 0.60 or ', '')])
    NB = pd.DataFrame(ns["rows"]); NB["nb"] = NB.draw + NB.life + NB.aura + NB.trick; NB["remc"] = NB.rem - NB.rem.mean(); del ns
    def f_nb(s):
        a = glm_prop(s, "y ~ nb + quality + cre + rem + splash + skill + C(lane)")
        b = glm_prop(s, "y ~ trick*remc + draw + life + aura + quality + cre + splash + skill + C(lane)")
        return {"B4 non-board spells, per spell": contrast(a, {"nb": 1}), "B4 trick x removal (per trick per removal card)": contrast(b, {"trick:remc": 1})}
    by_group(S, "nonboard", NB, f_nb); del NB
    # ---- B7 cantrips / 18 lands (podcast_lands18.py deck-level source: build())
    src = open(f"{A}/podcast_lands18.py").read(); i0 = src.index("# ---------- (a)"); i1 = src.index("by = {c["); i2 = src.index("for smin in")
    ns = {"__name__": "x", "__file__": f"{A}/podcast_lands18.py"}
    with quiet(): exec(compile(src[:i0] + src[i1:i2], "lands18", "exec"), ns)
    LD = ns["build"](0.0); del ns
    def lf(s, var):
        s = s.copy(); s["x"] = (s.lands == var).astype(int); s["y"] = s.w / s.g
        nx, nr = int(s.x.sum()), int((s.lands == 17).sum())
        if nx < 20 or nr < 20: return None
        return contrast(glm_prop(s, "y ~ x + quality + skill + C(lane)"), {"x": 1})
    def f_ld(s):
        c2 = s[(s.ct >= 2) & (s.lands != 18)]; c01 = s[(s.ct <= 1) & (s.lands != 18)]
        return {"B7 16 vs 17 lands, 2+ cantrips": lf(c2, 16), "B7 16 vs 17 lands, 0-1 cantrips": lf(c01, 16), "B1 18 vs 17 lands (deck GLM, all)": lf(s[s.lands != 16], 18)}
    by_group(S, "lands", LD, f_ld)
    for g in GROUPS:
        sub = sel(LD, g); S.setdefault("lands|_n", {})[g] = [int(((sub.ct >= 2) & (sub.lands == 16)).sum()), int(((sub.ct >= 2) & (sub.lands == 17)).sum()), int(len(sub))]
    del LD
    # ---- B9 splash by speed (podcast_splash_speed.py)
    ns = run_src(f"{A}/podcast_splash_speed.py", "\nres = {}", [(" + C(gb)", "")])
    SS = ns["df"]
    def f_ss(s):
        r = ns["run"]("ext", s, False); se = lambda t: (r[t][2] - r[t][1]) / 3.92
        return {"B9 extended vs clean, fast tercile (T1)": (r["T1"][0], se("T1")), "B9 extended-vs-clean cost, fast minus slow tercile (T1-T3)": (r["T1-T3"][0], se("T1-T3"))}
    by_group(S, "splash_speed", SS, f_ss); del ns, SS
    # ---- B9 weak pool (podcast_weakpool.py)
    ns = run_src(f"{A}/podcast_weakpool.py", "\nout = {}", [(" + C(gb)", "")])
    full = ns["decks"]
    def f_wp(s):
        ns["decks"] = s; m = ns["fit"](False)
        return {"B9 extended vs clean, weakest pool quartile (Q1)": contrast(m, {"extended": 1}), "B9 extended-vs-clean cost, Q1 minus Q4": contrast(m, {"extended:C(q)[T.Q4]": -1})}
    by_group(S, "weakpool", full, f_wp); del ns, full
    # ---- sinks and counters (video_counters.load rows)
    by = vc.cards(); T = vs.tag(by)
    ex = lambda c, by_, d: dict(sinks=sum(k for n, k in c.items() if n in T))
    SK = vc.load(ex)
    by_group(S, "sinks", SK, lambda s: {"Mana sinks, per sink (ACT+X+FB+MAN)": contrast(glm_prop(s, "y ~ sinks + quality + cre + splash + skill + C(lane)"), {"sinks": 1})}); del SK
    from companion.advisor_v2 import config
    eng = {n for n, v in by.items() if "Land" not in v["types"] and "removal" in config.tags(v, "FRA")}
    ex2 = lambda c, by_, d: dict(remeng=sum(k for n, k in c.items() if n in eng), ctr=sum(k for n, k in c.items() if n in vc.CORE))
    CT = vc.load(ex2); CT["sE_c"] = ((CT.remeng + CT.ctr) < 6).astype(int)
    def f_ct(s):
        m = glm_prop(s, "y ~ remeng + ctr + quality + cre + splash + skill + C(lane)"); m2 = glm_prop(s, "y ~ sE_c + quality + cre + splash + skill + C(lane)")
        return {"Counterspells (core 4), per card beyond removal count": contrast(m, {"ctr": 1}), "Counterspell vs removal card, removal minus counter": contrast(m, {"remeng": 1, "ctr": -1}),
                "Short of 6 removal+counters (pts for being short)": contrast(m2, {"sE_c": 1})}
    by_group(S, "counters", CT, f_ct); del CT
    # ---- flyers (flyer_check.py)
    ns = run_src(f"{A}/flyer_check.py", "df = pd.DataFrame(rows)", [('d["skill"] < 0.60 or ', "")])
    FL = pd.DataFrame(ns["rows"]); del ns
    by_group(S, "flyers", FL, lambda s: {"Flyers beyond rating, per flying creature": contrast(glm_prop(s, "y ~ flyers + creatures + quality + skill + C(lane)"), {"flyers": 1})})
    # ---- two-colour split (podcast_split.py)
    ns = run_src(f"{A}/podcast_split.py", "m = fit(None)", [('df[["lane", "gb"]]', 'df[["lane"]]')])
    SP = ns["df"]
    def f_sp(s):
        ns["df"] = s; ns["y"] = s[["w", "l"]]; m = ns["fit"](None); k = 0.1 * 24.75
        return {"Two-colour split, per +0.1 main-colour share (quality controlled)": (k * m.params["ms"], k * m.bse["ms"])}
    by_group(S, "split", SP, f_sp)
    save("misc", S)

# ---------------------------------------------------------------------------------------------------------------- b3b8
def b3b8(g):
    path = f"{A}/b3b8_speed_tapped.py"; src = open(path).read()
    npz = f"{SCR}/skill_b3b8_games.npz"
    subs = [('NPZ = f"{SCR}/b3b8_games.npz"', f'NPZ = "{npz}"'),
            ("DF = pd.DataFrame(rows)\n", "DF = pd.DataFrame(rows); DF = DF[(DF.skill >= LO - 1e-9) & (DF.skill < HI - 1e-9)].copy()\n"),
            ("clean = (np.char.str_len(main) == 2) & (np.char.str_len(splash) == 0)", "clean = (np.char.str_len(main) == 2) & (np.char.str_len(splash) == 0) & (skill >= LO - 1e-9) & (skill < HI - 1e-9)"),
            ('open(f"{OUT}/b3b8_speed_tapped.txt", "w").write("\\n".join(lines)); P("done")', 'P("done")')]
    for a, b in subs:
        assert a in src, a; src = src.replace(a, b)
    lo, hi = GROUPS[g]; ns = {"__name__": "x", "__file__": path, "LO": lo, "HI": hi}
    sys.argv = ["x", "run"]
    with quiet(): exec(compile(src, path, "exec"), ns)
    res, lines = ns["res"], ns["lines"]
    json.dump({"res": [[list(k), v] for k, v in res.items()], "lines": lines}, open(f"{PARTS}/raw_b3b8_{g}.json.raw", "w"), default=float)
    def find(prefix, split="all"):
        ks = [k for k in res if k[0] == prefix and k[1] == split] or [k for k in res if k[0].startswith(prefix) and "all formats" not in k[0] and k[1] == split]
        assert len(ks) == 1, (prefix, ks); return res[ks[0]]
    S = {}
    def put(row, e):
        if e is None or np.isnan(e[0]): S[row] = None; return
        S[row] = [float(e[0]), float((e[2] - e[1]) / 3.92), int(len(ns["B3" if row.startswith("B3") else "B8"]))]
    put("B3 slowness premium, linear per +1 SD mean cmc, Q(b) (post hoc)", find("mc linear, Q(b) n="))
    put("B3 P1 mean cmc ends-minus-middle, Q(a)", find("mc ends-minus-middle, Q(a)"))
    put("B3 P2 two-drop share ends-minus-middle, Q(a)", find("two ends-minus-middle, Q(a)"))
    put("B3 P3 aggressive cards x mean cmc, Q(a)", find("AGG x mc, Q(a)"))
    put("B8 P4 off-colour dual, per card, Q(a)", find("off-colour dual (P4), Q(a) n="))
    put("B8 P5 on-colour tapped lands pooled, per card, Q(a)", find("tap_on pooled (P5), Q(a)"))
    put("B8 P6 PW dual x walker count, Q(a)", find("PW x WALK (P6), Q(a) n="))
    put("B8 P7 on-colour tapped x deck speed, Q(a)", find("tap_on x mc (P7), Q(a) n="))
    ln = [x for x in lines if x.strip().startswith("P8 PW + SLOW, all games")]
    mm = re.search(r"decks\s+(\d+) games\s+(\d+).*adjusted ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", ln[0]) if ln else None
    S["B8 P8 tapped dual in opening hand (placebo-adjusted), games"] = None if mm is None else [float(mm.group(3)), (float(mm.group(5)) - float(mm.group(4))) / 3.92, int(mm.group(2))]
    d = {f"b3b8|{k}": {g: v} for k, v in S.items()}
    d["b3b8|_log"] = {g: [x for x in lines if x.startswith("B3 population") or x.startswith("Population:") or x.startswith("games ") or x.startswith("Decks:")][:8]}
    save(f"b3b8_{g}", d)
    print("\n".join(f"{k}: {v}" for k, v in S.items()))

def cache():
    path = f"{A}/b3b8_speed_tapped.py"; src = open(path).read()
    a = 'NPZ = f"{SCR}/b3b8_games.npz"'; assert a in src; src = src.replace(a, f'NPZ = "{SCR}/skill_b3b8_games.npz"')
    sys.argv = ["x", "cache"]
    try: exec(compile(src, path, "exec"), {"__name__": "x", "__file__": path})
    except SystemExit: pass

# ---------------------------------------------------------------------------------------------------------------- older
def older():
    rows = list(csv.DictReader(open(f"{R}/data/tables/FRA/2_cards.csv")))
    num = lambda r, f: float(r.get(f) or 0)
    T = {r["name"]: r for r in rows}
    sp = [r for r in rows if num(r, "land") == 0]; gw = np.array([num(r, "gihwr") for r in sp]); gw = gw[gw > 0]; gm, gs = gw.mean(), gw.std()
    top40 = set(r["name"] for r in sorted((r for r in sp if num(r, "gihwr") > 0), key=lambda r: -num(r, "gihwr"))[:40])
    print("gihwr mean/sd", round(gm, 2), round(gs, 2), "finisher cut", round(gm + gs, 2), "top40 cut", min(num(T[n], "gihwr") for n in top40))
    out = []
    for d in json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt")):
        if d["format"] != "PremierDraft" or d["deck_games"] < 1 or not 2 <= len(d["lane"]) <= 3: continue
        cr = rm = fin = lands = two = top = 0
        for n, k in d["counts"].items():
            r = T.get(n) or T.get(n.split(" // ")[0])
            if r is None: continue
            if num(r, "land") == 1: lands += k; continue
            cr += k * (num(r, "role_creatures") > 0); rm += k * (num(r, "role_removal") > 0); two += k * (num(r, "cmc") == 2)
            fin += k * (num(r, "cmc") >= 5 and num(r, "gihwr") >= gm + gs); top += k * (r["name"] in top40)
        out.append(dict(draft=d["draft"], lane=d["lane"], splash=int(d["splash"] != ""), skill=d["skill"], w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"],
                        cr9=int(cr <= 9), rm2=int(rm <= 2), notop=int(top == 0), landoff=int(lands not in (16, 17)), two4=int(two <= 4), fin=min(fin, 3), fin3=int(fin >= 3), fin1=int(fin >= 1), finraw=fin))
    DF = pd.DataFrame(out); print(len(DF), DF[["cr9", "rm2", "notop", "landoff", "two4", "fin1", "fin3"]].mean().round(3).to_dict(), flush=True)
    S = {}
    def fx(col, name): return lambda s: {name: contrast(glm_cb(s, f"I(w)+I(l) ~ {col} + skill + splash + C(lane)"), {col: 1})}
    for col, name in (("cr9", "older: creatures <= 9"), ("rm2", "older: removal <= 2"), ("notop", "older: no top-40 card"), ("landoff", "older: lands not 16-17"), ("two4", "older: two-drops <= 4"),
                      ("fin", "older: finishers, per card (capped at 3)"), ("fin3", "older: finishers 3+ vs 0-2")):
        by_group(S, col, DF, fx(col, name))
    # earlier bands (low < 0.45, mid 0.45-0.59, high 0.60+) on the 100+ games decks, finishers per card capped at 3
    for nm, (lo, hi) in {"band low <0.45": (0, .45), "band mid 0.45-0.59": (.45, .60), "band high 0.60+": (.60, 9)}.items():
        s = DF[(DF.skill >= lo - 1e-9) & (DF.skill < hi - 1e-9)]; e, se = contrast(glm_cb(s, "I(w)+I(l) ~ fin + skill + splash + C(lane)"), {"fin": 1})
        S.setdefault("finbands|" + nm, {})["all"] = [e, se, len(s)]; print(nm, len(s), round(e, 2), round(se, 2))
    save("older", S)

# ---------------------------------------------------------------------------------------------------------------- report
def report():
    M = {}
    for fn in sorted(os.listdir(PARTS)):
        if fn.endswith('.raw'): continue
        for k, v in json.load(open(f"{PARTS}/{fn}")).items(): M.setdefault(k, {}).update(v)
    ORIG = {"B7 16 vs 17 lands, 2+ cantrips": "+0.74 [-0.02, +1.51] (60%+)", "Two-colour split, per +0.1 main-colour share (quality controlled)": "+0.37, p 0.094 (all)",
            "B3 slowness premium, linear per +1 SD mean cmc, Q(b) (post hoc)": "+0.80 [+0.47, +1.14]", "B8 P7 on-colour tapped x deck speed, Q(a)": "+0.58 [+0.04, +1.13] p .037",
            "B8 P8 tapped dual in opening hand (placebo-adjusted), games": "+1.47 [+0.26, +2.69]", "Mana sinks, per sink (ACT+X+FB+MAN)": "-0.23 [-0.39, -0.06]",
            "older: creatures <= 9": "+2.0 (SE 0.4)", "older: removal <= 2": "-2.1 (SE 0.4)", "older: no top-40 card": "-2.5", "older: lands not 16-17": "no effect", "older: two-drops <= 4": "no effect",
            "older: finishers, per card (capped at 3)": "large low/high bands, flat mid"}
    rowsk = [k for k in M if not k.endswith("_n") and not k.endswith("_log") and not k.startswith("finbands|")]
    N = len(rowsk); bd, bg = 0.05 / N, 0.05 / (3 * N)
    L = []; P = L.append
    f = lambda v: "n/a" if v is None else f"{v[0]:+.2f} [{v[0]-1.96*v[1]:+.2f}, {v[0]+1.96*v[1]:+.2f}]"
    pz = lambda v: np.nan if v is None or not v[1] else 2 * norm.sf(abs(v[0] / v[1]))
    P(f"Rows (tests): {N}. Bonferroni: good-minus-weak p < {bd:.4f}; a single group p < {bg:.5f}. Nominal flag: p < 0.05.\n")
    P("| test | original | all 100+ | good (>=0.60) | average (0.52-0.59) | weak (<0.52) | good minus weak | p(diff) | verdict |"); P("|---|---|---|---|---|---|---|---|---|")
    for k in rowsk:
        v = M[k]; t = k.split("|", 1)[1]; gd, av, wk, al = v.get("good"), v.get("avg"), v.get("weak"), v.get("all")
        diff = None if gd is None or wk is None else [gd[0] - wk[0], float(np.hypot(gd[1], wk[1]))]
        ps = {n: pz(x) for n, x in (("good", gd), ("avg", av), ("weak", wk))}; pdif = pz(diff)
        nom = [n for n, p in ps.items() if p < 0.05]; surv = [n for n in nom if ps[n] < bg]; pall = pz(al)
        sg = lambda x: np.sign(x[0]) if x else 0
        same = gd and av and wk and len({sg(gd), sg(av), sg(wk)}) == 1
        opp = gd and wk and sg(gd) != sg(wk) and ps["good"] < 0.05 and ps["weak"] < 0.05
        if diff and pdif < bd: ver = f"groups differ (good-weak survives, p {pdif:.4f})"
        elif opp: ver = "opposite effects (good and weak nominal, opposite signs)" + ("" if pdif < bd else "; does not survive")
        elif pall < bd and same and not (diff and pdif < 0.05): ver = f"same effect in every group (pooled p {pall:.1e}); no group-specific effect"
        elif surv: ver = f"effect in {','.join(surv)} survives; nominal in {','.join(n for n in nom if n not in surv) or 'none else'}"
        elif nom or (diff and pdif < 0.05) or pall < 0.05: ver = f"nominal only ({','.join(nom) or 'no single group'}{'; diff p<.05' if diff and pdif < 0.05 else ''}{'; pooled p<.05' if pall < 0.05 else ''}), none survives"
        else: ver = "null in every group"
        P(f"| {t} | {ORIG.get(t, '-')} | {f(al)} | {f(gd)} | {f(av)} | {f(wk)} | {f(diff)} | {pdif:.3f} | {ver} |")
    P("\nOlder finisher bands (100+ games decks, per finisher capped at 3, no game-count term):")
    for k in M:
        if k.startswith("finbands|"): P(f"  {k.split('|')[1]}: {f(M[k]['all'])}  decks {M[k]['all'][2]}")
    P("\nSample sizes (decks; games for P8):")
    for k in rowsk:
        v = M[k]; P(f"  {k.split('|')[1][:70]:72} " + "  ".join(f"{g}:{v[g][2] if v.get(g) else '-'}" for g in GROUPS))
    if "lands|_n" in M: P("\nB7 cell sizes (16-land cantrip-2+ decks / 17-land cantrip-2+ decks / all decks): " + str(M["lands|_n"]))
    P("\nb3b8 logs:"); [P(f"  {g}: {x}") for g, v in M.get("b3b8|_log", {}).items() for x in v]
    open(f"{OUT}/skill_groups_dropped.txt", "w").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    {"misc": misc, "cache": cache, "older": older, "report": report, "b3b8": lambda: b3b8(sys.argv[2])}[sys.argv[1]]()
