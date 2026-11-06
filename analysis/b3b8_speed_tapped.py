"""B3 (deck speed inside a pair) and B8 (enters-tapped lands), FRA Premier. Test plan: docs/test-plan-2026-11-07.md. Doc: docs/b3b8-speed-tapped-FRA.md.
  python3 analysis/b3b8_speed_tapped.py cache   # stream the Premier game CSV (20k-row chunks) into a compact npz in SCR (lands in opening hand, deck speed)
  python3 analysis/b3b8_speed_tapped.py run     # -> analysis/out/b3b8_speed_tapped.txt
Run as: systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 ...

PRE-REGISTRATION (written before any outcome of these tests was computed; the design is repeated in the doc).
Deck level (analysis/out/decks_FRA.json.gz; GLM binomial on wins/games, freq weights = deck games, SE clustered by draft, controls skill + lane FE, NO game-count term;
effects in win-rate points at 55% = 24.75 x logit coefficient; halves = crc32(draft) % 2).
Quality controls, both reported for every model: (a) mean All-Decks GIH WR of all spells in the deck plus top-10 mean; (b) the same outside the tested class
  (B3 mean cmc: mean GIH of the cmc 3-4 spells, which do not move the speed tails; B3 two-drops: spells outside the 2-drops; B3 aggressive cards: outside AGG;
   B8 lands: spells are already outside the class, (b) = (a) + nothing, so reported once as (a); B8 walkers: outside WALK).
B3 population: PremierDraft, two-colour lane (splash allowed, dummy control). Speed = mean cmc of non-land cards (mc); two-drop share = 2-cmc non-land cards / non-land cards.
  Both centred on the lane (pair) mean and scaled by the pooled within-pair SD, so a squared term is within-pair. Convex (ends beat the middle) = positive squared term.
  P1: mc squared term (pts at +-1 SD above the middle). P2: same for two-drop share. P3: AGG x mc interaction (pts per AGG card per +1 SD of mc; the trap claim predicts negative).
  AGG (fixed from oracle/Scryfall before outcomes): non-land creature with (cmc == 2 and power >= 2) or (cmc <= 3 and haste or menace).
  Secondary: within-pair terciles of mc (middle vs mean of the outer two), all formats, halves.
B8 lands (from oracle text, listed in the output): PW = tapped unless you control a planeswalker (10 duals); SLOW = tapped unless two or more other lands (5 duals, untapped from
  the third land drop); ALW = Room of Refuge, Roiling Canopy, Theorist's Sanctum (tapped, Sanctum unless a Jace is beheld). On-colour = colour set within the lane (Room: any lane).
  Off-colour dual = dual whose colour pair differs from the lane. WALK = planeswalker cards + Ways + empower cards (analysis/out/video_empower_lists.json).
  Deck level population: PremierDraft, two-colour lane, no splash. Controls: Q(a), skill, lane FE, land count (<=16 / 17 / >=18).
  P4: off-colour duals (per card). P5: on-colour tapped lands, PW + SLOW + ALW (per card). P6: PW-on x WALK count (centred) interaction, plus PW-on slope in WALK >= 4 vs < 4.
  P7: on-colour tapped x mc (within-pair SD units).
  Within deck (games): P8: a PW or SLOW dual of the lane's colours in the opening hand vs another land (lands in opening hand and mulligans held), deck FE, decks with >= 1 such dual,
  SE clustered by draft, adjusted by a placebo (one random lane basic in the opening hand, 40 draws). Splits fixed in advance: play / draw; deck-speed terciles (cut at the tercile of games' mc);
  WALK >= 4; PW vs SLOW; halves.
Multiplicity: 8 primaries (P1-P8): Bonferroni alpha 0.00625; everything else is secondary or robustness and labelled so."""
import json, gzip, os, re, sys, zlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{R}/analysis/out"
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
NPZ = f"{SCR}/b3b8_games.npz"
GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
K = 100 * .55 * .45

def cards():  # as b1_within_removal.cards
    by = {}
    for v in json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"].values():
        n = v["name"].split(" // ")[0]
        if n in by and not v.get("isprimarycard", 1): continue
        by[n] = v
    return by
by = cards()
SC = {c["name"].split(" // ")[0]: c for c in json.load(open(f"{R}/data/Temp/scryfall_cards_FRA.json")).values()}
isl = lambda v: "Land" in v["types"]
gA = lambda v: (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
EL = json.load(open(f"{OUT}/video_empower_lists.json"))
nl = {n: v for n, v in by.items() if not isl(v)}
WALK = {n for n, v in nl.items() if "Planeswalker" in v["types"]} | set(EL["ways"]) | set(EL["empower"])
def agg(n):
    v, s = nl[n], SC.get(n, {})
    if "Creature" not in v["types"]: return False
    c = v["cmc"] or 0
    try: p = float(s.get("power"))
    except (TypeError, ValueError): p = np.nan
    kw = set(s.get("keywords") or []) | set(v.get("keywords") or [])
    return (c == 2 and p >= 2) or (c <= 3 and bool(kw & {"Haste", "Menace"}))
AGG = {n for n in nl if agg(n)}
# lands
LAND = {}   # name -> (class, colour set)
for n, v in by.items():
    if not isl(v) or n in ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes") or n.startswith("Snow-Covered"): continue
    t = v.get("oracle_text") or ""; cs = frozenset(v["colors"])
    if "enters tapped unless you control a planeswalker" in t: c = "PW"
    elif "enters tapped unless you control two or more other lands" in t: c = "SLOW"
    elif "this land enters tapped" in t.lower() or "enters tapped." in t: c = "ALW"
    elif "behold a Jace. If you don't, this land enters tapped" in t: c = "ALW"
    else: c = "UNT"
    LAND[n] = (c, cs)
BASICS = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
TAPPED = {n: x for n, x in LAND.items() if x[0] != "UNT"}
ROOM = "Room of Refuge"
def on_lane(n, lane):  # on-colour: colour set within the lane (Room of Refuge, colourless chosen colour: any lane)
    cs = LAND[n][1]
    return n == ROOM or (len(cs) > 0 and cs <= set(lane))
lines = []
def P(s=""): print(s, flush=True); lines.append(s)

# ================================================================ cache
if sys.argv[1] == "cache":
    hdr = pd.read_csv(GD, nrows=0).columns.tolist()
    spells = sorted(n for n in nl if f"deck_{n}" in hdr)
    landn = [n for n in by if isl(by[n]) and f"opening_hand_{n}" in hdr]
    keep = list(TAPPED) + list(LAND) + BASICS; keep = [n for n in dict.fromkeys(keep) if f"opening_hand_{n}" in hdr]
    meta = ["draft_id", "build_index", "won", "on_play", "num_mulligans", "main_colors", "splash_colors", "user_game_win_rate_bucket"]
    use = meta + [f"opening_hand_{n}" for n in landn] + [f"deck_{n}" for n in spells] + [f"deck_{n}" for n in keep]
    dt = {c: np.float32 for c in use if c not in meta and c not in ("main_colors", "splash_colors")}
    cmc = np.array([nl[n]["cmc"] or 0 for n in spells], float); two = (cmc == 2).astype(float)
    wk = np.array([n in WALK for n in spells], float); ag = np.array([n in AGG for n in spells], float)
    parts = []
    for i, ch in enumerate(pd.read_csv(GD, usecols=use, chunksize=20000, dtype=dt)):
        D = ch[[f"deck_{n}" for n in spells]].fillna(0).values.astype(np.float32); ns = D.sum(1)
        m = ch[meta].copy()
        m["nsp"] = ns; m["mc"] = D @ cmc / np.maximum(ns, 1); m["two"] = D @ two; m["walk"] = D @ wk; m["nagg"] = D @ ag
        m["ohl"] = ch[[f"opening_hand_{n}" for n in landn]].fillna(0).values.sum(1)
        O = ch[[f"opening_hand_{n}" for n in keep]].fillna(0).values.astype(np.int8)
        Dl = ch[[f"deck_{n}" for n in keep]].fillna(0).values.astype(np.int8)
        parts.append((m, O, Dl)); print("chunk", i, flush=True)
    M = pd.concat([p[0] for p in parts], ignore_index=True)
    np.savez_compressed(NPZ, O=np.vstack([p[1] for p in parts]), Dl=np.vstack([p[2] for p in parts]), keep=np.array(keep),
                        draft=M.draft_id.values.astype(str), build=M.build_index.values.astype(np.int16), won=M.won.values.astype(np.int8),
                        play=M.on_play.values.astype(np.int8), mull=M.num_mulligans.fillna(0).values.astype(np.int8),
                        main=M.main_colors.fillna("").values.astype(str), splash=M.splash_colors.fillna("").values.astype(str),
                        skill=M.user_game_win_rate_bucket.values.astype(np.float32), mc=M.mc.values.astype(np.float32), nsp=M.nsp.values.astype(np.int16),
                        two=M.two.values.astype(np.int8), walk=M.walk.values.astype(np.int8), agg=M["nagg"].values.astype(np.int8), ohl=M.ohl.values.astype(np.int8))
    print("cached"); sys.exit()

# ================================================================ run
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
P("# B3 / B8: deck speed and enters-tapped lands (FRA Premier)\n")
P("## Card sets (fixed from oracle text / Scryfall before outcomes)\n")
P("Lands that enter tapped (class, colours):")
for n, (c, cs) in sorted(TAPPED.items(), key=lambda x: (x[1][0], x[0])): P(f"  {c:5} {n:24} {''.join(sorted(cs)) or '-'}   | " + (by[n].get("oracle_text") or "").split("\n")[0][:90])
P("Non-basic lands that do not enter tapped: " + ", ".join(sorted(n for n, x in LAND.items() if x[0] == "UNT")))
P(f"\nAGG ({len(AGG)}): " + ", ".join(sorted(AGG))); P(f"WALK ({len(WALK)}): " + ", ".join(sorted(WALK)))

def pts(m, w):
    names = list(m.params.index); v = np.zeros(len(names))
    for t, c in w.items(): v[names.index(t)] = c
    b = float(v @ m.params.values); se = float(np.sqrt(v @ m.cov_params().values @ v))
    return K * b, K * (b - 1.96 * se), K * (b + 1.96 * se), float(2 * norm.sf(abs(b / se)))
def fit(df, f):
    G = pd.factorize(df.draft)[0]
    return smf.glm("y ~ " + f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": G})
def fmt(e): return "n/a" if e is None or np.isnan(e[0]) else f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}] p={e[3]:.3f}"

# ---------------------------------------------------------------- deck table
DECKS = json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt"))
rows = []
for d in DECKS:
    lane = d["lane"]; its = []; lc = {}; nland = 0
    for n, k in d["counts"].items():
        n = n.split(" // ")[0]; v = by.get(n)
        if v is None: continue
        if isl(v):
            nland += k
            if n in LAND: lc[n] = lc.get(n, 0) + k
        else: its.append((n, k, v["cmc"] or 0, gA(v)))
    if not its: continue
    nm = np.array([x[0] for x in its]); k = np.array([x[1] for x in its], float); c = np.array([x[2] for x in its], float); g = np.array([x[3] for x in its], float)
    def qm(mask): s = mask & (g > 0); return np.average(g[s], weights=k[s]) if s.any() and k[s].sum() > 0 else np.nan
    def t10(mask):
        s = np.repeat(g[mask & (g > 0)], k[mask & (g > 0)].astype(int)); return np.sort(s)[-10:].mean() if len(s) else np.nan
    ia = np.array([n in AGG for n in nm]); iw = np.array([n in WALK for n in nm]); i2 = c == 2; al = np.ones(len(nm), bool)
    r = dict(draft=d["draft"], fmt=d["format"], lane=lane, nlane=len(lane), splash=int(d["splash"] != ""), skill=d["skill"], w=d["deck_wins"], g=d["deck_games"], y=d["deck_wins"] / d["deck_games"],
             nsp=k.sum(), mc=(k * c).sum() / k.sum(), two=k[i2].sum() / k.sum(), nag=k[ia].sum(), walk=k[iw].sum(), nland=nland,
             qA=qm(al), t10=t10(al), q34=qm((c >= 3) & (c <= 4)), qA_x2=qm(~i2), t10_x2=t10(~i2), qA_xag=qm(~ia), t10_xag=t10(~ia), qA_xwk=qm(~iw), t10_xwk=t10(~iw))
    pw = sl = al_ = off = 0
    for n, kk in lc.items():
        cl, cs = LAND[n]
        if cl in ("PW", "SLOW"):
            if cs == set(lane): pw += kk if cl == "PW" else 0; sl += kk if cl == "SLOW" else 0
            else: off += kk
        elif cl == "ALW" and on_lane(n, lane): al_ += kk
    r.update(pw_on=pw, slow_on=sl, alw_on=al_, off=off, tap_on=pw + sl + al_, dual_any=pw + sl + off)
    rows.append(r)
DF = pd.DataFrame(rows)
DF["half"] = DF.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
DF["nl3"] = pd.cut(DF.nland, [0, 16, 17, 99], labels=["le16", "l17", "ge18"]).astype(str)
P(f"\nDecks: {len(DF)}; Premier {int((DF.fmt=='PremierDraft').sum())}; two-colour lane {int((DF.nlane==2).sum())}")

# ================================================================ B3
P("\n# B3 deck speed inside a pair\n")
B3 = DF[(DF.fmt == "PremierDraft") & (DF.nlane == 2) & DF.qA.notna() & DF.t10.notna() & DF.q34.notna()].copy()
def prep(d):
    d = d.copy()
    for v in ("mc", "two"):
        dev = d[v] - d.groupby("lane")[v].transform("mean"); d[v + "w"] = dev / dev.std(); d[v + "w2"] = d[v + "w"] ** 2
        r = d.groupby("lane")[v].rank(pct=True, method="first"); d[v + "T"] = np.where(r <= 1 / 3, "lo", np.where(r <= 2 / 3, "mid", "hi"))
    d["aggc"] = d["nag"] - d["nag"].mean()
    return d
B3 = prep(B3)
P(f"B3 population: {len(B3)} decks (Premier, two-colour lane), {B3.draft.nunique()} drafts; mc mean {B3.mc.mean():.2f} (pooled within-pair SD {(B3.mc-B3.groupby('lane').mc.transform('mean')).std():.3f}); "
  f"two-drop share mean {B3.two.mean():.3f} (within-pair SD {(B3.two-B3.groupby('lane').two.transform('mean')).std():.3f}); AGG per deck mean {B3["nag"].mean():.2f}, SD {B3["nag"].std():.2f}")
P(f"corr(mc, qA) {B3.mc.corr(B3.qA):+.2f}; corr(mc, two) {B3.mc.corr(B3.two):+.2f}; corr(mc, agg) {B3.mc.corr(B3["nag"]):+.2f}")
CTRL = "skill + splash + C(lane)"
QS = {"mc": {"a": "qA + t10", "b": "q34", "none": ""}, "two": {"a": "qA + t10", "b": "qA_x2 + t10_x2", "none": ""}, "agg": {"a": "qA + t10", "b": "qA_xag + t10_xag", "none": ""}}
def one(d, term, kind, q, w):
    f = " + ".join(x for x in (term, QS[kind][q], CTRL) if x)
    try: return pts(fit(d, f), w)
    except Exception as e: print("fail", f, e); return None
res = {}
def rec(key, e): res[key] = e; P(f"  {key[0]:34} {key[1]:12} {fmt(e)}")
SPL = [("all", B3), ("half 0", B3[B3.half == 0]), ("half 1", B3[B3.half == 1])]
for kind, v in (("mc", "mc"), ("two", "two")):
    P(f"\n## {'P1' if kind=='mc' else 'P2'}: {'mean cmc' if kind=='mc' else 'two-drop share'} within pair, linear + squared (pts: linear = per +1 SD; ends-minus-middle = squared term, i.e. WR at +-1 SD minus WR at the pair mean)")
    for q in ("a", "b", "none"):
        for sn, d in SPL:
            m = one(d, f"{v}w + {v}w2", kind, q, {f"{v}w": 1});
            mm = one(d, f"{v}w + {v}w2", kind, q, {f"{v}w2": 1})
            rec((f"{kind} linear, Q({q}) n={len(d)}", sn), m); rec((f"{kind} ends-minus-middle, Q({q})", sn), mm)
    P(f"  terciles within pair ({kind}), middle = reference; ends minus middle = mean of the two outer terciles minus middle")
    for q in ("a", "b"):
        for sn, d in SPL[:1]:
            f = " + ".join((f"C({v}T, Treatment('mid'))", QS[kind][q], CTRL))
            try:
                m = fit(d, f); nm_ = list(m.params.index); lo = [x for x in nm_ if "[T.lo]" in x][0]; hi = [x for x in nm_ if "[T.hi]" in x][0]
                rec((f"{kind} terciles Q({q})  ends-mid", sn), pts(m, {lo: .5, hi: .5})); rec((f"{kind} terciles Q({q})  hi-lo", sn), pts(m, {hi: 1, lo: -1}))
            except Exception as e: print("fail", e)
# P3
P("\n## P3: AGG x mean cmc (within-pair SD units). AGG main = pts per AGG card at the average speed; interaction = change per +1 SD of mc (trap claim: negative)")
for q in ("a", "b", "none"):
    for sn, d in SPL:
        f = " + ".join(x for x in ("aggc + mcw + aggc:mcw", QS["agg"][q], CTRL) if x)
        try:
            m = fit(d, f); rec((f"AGG main, Q({q}) n={len(d)}", sn), pts(m, {"aggc": 1})); rec((f"AGG x mc, Q({q})", sn), pts(m, {"aggc:mcw": 1}))
        except Exception as e: print("fail", e)
P("  AGG slope by mc tercile (secondary)")
for q in ("a", "b"):
    d = B3
    f = " + ".join(("C(mcT):aggc + C(mcT)", QS["agg"][q], CTRL))
    try:
        m = fit(d, f)
        for t in ("lo", "mid", "hi"): rec((f"AGG per card in {t}-speed tercile, Q({q})", "all"), pts(m, {f"C(mcT)[{t}]:aggc": 1}))
    except Exception as e: print("fail", e)
# all-formats sensitivity
Ball = DF[(DF.nlane == 2) & DF.qA.notna() & DF.t10.notna() & DF.q34.notna()].copy(); Ball = prep(Ball)
P(f"\nSensitivity, all formats ({len(Ball)} decks), format dummy added:")
CTRL0 = CTRL; CTRL = CTRL + " + C(fmt)"
for kind, v in (("mc", "mc"), ("two", "two")):
    rec((f"{kind} ends-minus-middle, Q(a), all formats", "all"), one(Ball, f"{v}w + {v}w2", kind, "a", {f"{v}w2": 1}))
    rec((f"{kind} linear, Q(a), all formats", "all"), one(Ball, f"{v}w + {v}w2", kind, "a", {f"{v}w": 1}))
f = " + ".join(("aggc + mcw + aggc:mcw", QS["agg"]["a"], CTRL))
rec(("AGG x mc, Q(a), all formats", "all"), pts(fit(Ball, f), {"aggc:mcw": 1}))
CTRL = CTRL0

# ================================================================ B8 deck level
P("\n# B8 (b)(c) deck level: enters-tapped lands in clean two-colour decks (Premier, no splash)\n")
B8 = DF[(DF.fmt == "PremierDraft") & (DF.nlane == 2) & (DF.splash == 0) & DF.qA.notna() & DF.t10.notna() & DF.qA_xwk.notna()].copy()
B8 = prep(B8); B8["walkc"] = B8.walk - B8.walk.mean(); B8["wk4"] = (B8.walk >= 4).astype(int)
P(f"Population: {len(B8)} decks, {B8.draft.nunique()} drafts. Per deck means: PW-on {B8.pw_on.mean():.2f}, SLOW-on {B8.slow_on.mean():.2f}, ALW-on {B8.alw_on.mean():.2f}, off-colour duals {B8.off.mean():.2f}; WALK {B8.walk.mean():.2f} (SD {B8.walk.std():.2f}); WALK>=4 {B8.wk4.mean():.2%}")
P("Distribution of counts (decks): " + "; ".join(f"{c}: " + ", ".join(f"{int(k)}:{int(v)}" for k, v in B8[c].value_counts().sort_index().items()) for c in ("pw_on", "slow_on", "alw_on", "off", "tap_on")))
SPL8 = [("all", B8), ("half 0", B8[B8.half == 0]), ("half 1", B8[B8.half == 1])]
C8 = "skill + C(lane) + C(nl3)"
def one8(d, term, q, w):
    qq = {"a": "qA + t10", "none": "", "qonly": "qA", "b": "qA_xwk + t10_xwk"}[q]
    f = " + ".join(x for x in (term, qq, C8) if x)
    try: return pts(fit(d, f), w)
    except Exception as e: print("fail", f, e); return None
P("\n## P4 / P5: per-card effect (pts) of land classes, all entered together (replaces a basic; land count controlled)")
for q in ("a", "none"):
    for sn, d in SPL8:
        f = "pw_on + slow_on + alw_on + off"
        for t, lab in (("pw_on", "on-colour PW dual"), ("slow_on", "on-colour SLOW dual"), ("alw_on", "on-colour ALW tapped"), ("off", "off-colour dual (P4)")):
            rec((f"{lab}, Q({q}) n={len(d)}", sn), one8(d, f, q, {t: 1}))
P("\n  P5: on-colour tapped lands pooled (PW + SLOW + ALW) with off-colour duals separate")
for q in ("a", "none"):
    for sn, d in SPL8:
        rec((f"tap_on pooled (P5), Q({q})", sn), one8(d, "tap_on + off", q, {"tap_on": 1}))
P("\n## P6: PW-on duals x WALK (centred walker count): interaction per extra walker card (positive = walkers offset the cost); and PW-on slope by WALK >= 4")
for q in ("a", "b"):
    for sn, d in SPL8:
        dd = d.copy(); dd["walkc"] = dd.walk - dd.walk.mean()
        rec((f"PW x WALK (P6), Q({q}) n={len(d)}", sn), one8(dd, "pw_on*walkc + slow_on + alw_on + off", q, {"pw_on:walkc": 1}))
    d = B8
    rec((f"PW slope, WALK<4, Q({q})", "all"), one8(d, "pw_on:C(wk4) + C(wk4) + slow_on + alw_on + off", q, {"pw_on:C(wk4)[0]": 1}))
    rec((f"PW slope, WALK>=4, Q({q})", "all"), one8(d, "pw_on:C(wk4) + C(wk4) + slow_on + alw_on + off", q, {"pw_on:C(wk4)[1]": 1}))
    rec((f"PW slope diff (>=4 minus <4), Q({q})", "all"), one8(d, "pw_on:C(wk4) + C(wk4) + slow_on + alw_on + off", q, {"pw_on:C(wk4)[1]": 1, "pw_on:C(wk4)[0]": -1}))
P("\n## P7: on-colour tapped lands x deck speed (mcw, within-pair SD): interaction per card per +1 SD of mc (cost grows in fast decks = positive on slow)")
for q in ("a", "none"):
    for sn, d in SPL8:
        rec((f"tap_on x mc (P7), Q({q}) n={len(d)}", sn), one8(d, "tap_on*mcw + off", q, {"tap_on:mcw": 1}))
    rec((f"tap_on main at average speed, Q({q})", "all"), one8(B8, "tap_on*mcw + off", q, {"tap_on": 1}))
P("  tap_on slope by mc tercile (secondary)")
for t in ("lo", "mid", "hi"):
    rec((f"tap_on per card, {t}-speed tercile", "all"), one8(B8, "C(mcT):tap_on + C(mcT) + off", "a", {f"C(mcT)[{t}]:tap_on": 1}))
P("\n  Secondary: the 4th+ tapped land (tap_on >= 4 vs <= 3), unit change")
B8["t4"] = (B8.tap_on >= 4).astype(int); P(f"  decks with tap_on >= 4: {int(B8.t4.sum())}")
rec(("tap_on >= 4 dummy, Q(a)", "all"), one8(B8, "tap_on + t4 + off", "a", {"t4": 1}))
rec(("tap_on >= 4 dummy x WALK>=4", "all"), one8(B8, "t4*wk4 + tap_on + off", "a", {"t4:wk4": 1}))

# ================================================================ B8 (a) within deck
P("\n# B8 (a) within deck: lane-colour PW / SLOW dual in the opening hand vs another land (Premier games)\n")
Z = np.load(NPZ, allow_pickle=True)
O, Dl, keep = Z["O"], Z["Dl"], list(Z["keep"]); kix = {n: i for i, n in enumerate(keep)}
draft = np.asarray(Z["draft"], dtype=str); build = Z["build"]; won = Z["won"].astype(float); play = Z["play"]; mull = Z["mull"].astype(float)
main = np.asarray(Z["main"], dtype=str); splash = np.asarray(Z["splash"], dtype=str); skill = Z["skill"]; mc = Z["mc"].astype(float); walk = Z["walk"]; ohl = Z["ohl"].astype(float)
dcode, _ = pd.factorize(draft); deck = pd.factorize(pd.Series(draft) + "|" + pd.Series(build.astype(str)))[0]
half = pd.Series(draft).map(lambda x: zlib.crc32(x.encode()) % 2).values
rng = np.random.default_rng(20261107)
clean = (np.char.str_len(main) == 2) & (np.char.str_len(splash) == 0)
mainset = np.array(["".join(sorted(x)) for x in main])
def classcount(arr, classes):
    out = np.zeros(len(main), np.int16)
    for n, (c, cs) in LAND.items():
        if c in classes and n in kix and len(cs) == 2: out += arr[:, kix[n]].astype(np.int16) * (mainset == "".join(sorted(cs)))
    return out
OH_PW, OH_SL = classcount(O, {"PW"}), classcount(O, {"SLOW"}); DK_PW, DK_SL = classcount(Dl, {"PW"}), classcount(Dl, {"SLOW"})
P(f"games {len(won)}, decks {deck.max()+1}, drafts {dcode.max()+1}; clean two-colour games {int(clean.sum())}; on play {play.mean():.3f}")
P(f"on-colour dual in opening hand: PW {OH_PW[clean].mean():.3f} per game, SLOW {OH_SL[clean].mean():.3f}; decks with a lane PW dual {len(np.unique(deck[clean & (DK_PW>0)]))}, SLOW {len(np.unique(deck[clean & (DK_SL>0)]))}")
def demean(X, g):
    cnt = np.bincount(g).astype(float); o = X.astype(float).copy()
    for j in range(X.shape[1]): o[:, j] -= (np.bincount(g, X[:, j], minlength=len(cnt)) / np.maximum(cnt, 1))[g]
    return o
def wfit(x, mask, extra=()):
    """deck-demeaned OLS of won on [x, lands in opening hand, mulligans, extra]; returns win-rate points per card (est, lo, hi, p); SE clustered by draft"""
    m = np.where(mask)[0]; g = pd.factorize(deck[m])[0]; ok = np.bincount(g)[g] >= 2; m = m[ok]; g = pd.factorize(g[ok])[0]
    if len(m) < 2000 or x[m].std() == 0: return None
    X = np.column_stack([x[m], ohl[m], mull[m]] + [e[m] for e in extra]); A = demean(np.column_stack([X, won[m]]), g)
    mod = sm.OLS(A[:, -1], A[:, :-1]).fit(cov_type="cluster", cov_kwds={"groups": dcode[m]})
    b, se = 100 * mod.params[0], 100 * mod.bse[0]
    return b, b - 1.96 * se, b + 1.96 * se, float(mod.pvalues[0]), len(m), len(np.unique(g)), se
BM = np.column_stack([O[:, kix[b]] for b in BASICS]).astype(float)   # opening-hand basics, order Plains Island Swamp Mountain Forest
CIX = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4}
LC = np.array([[CIX.get(c, 0) for c in (x + "WW")[:2]] for x in mainset])
def placebo(mask, n=40):
    """one random lane basic per deck in the opening hand instead of the dual; same mask, same deck FE"""
    idx = np.where(mask)[0]; out = []
    for _ in range(n):
        pick = rng.integers(0, 2, deck.max() + 1)[deck[idx]]
        x = np.zeros(len(main)); x[idx] = BM[idx, LC[idx, pick]]
        r = wfit(x, mask)
        if r: out.append(r[0])
    return float(np.mean(out)), float(np.std(out, ddof=1))
def wrec(lab, x, mask, plac=True):
    r = wfit(x, mask)
    if r is None: P(f"  {lab:46} too few"); return
    if plac:
        pm, ps = placebo(mask); adj = r[0] - pm; ase = np.hypot(r[6], ps); h = 1.96 * ase
        P(f"  {lab:46} decks {r[5]:5} games {r[4]:6}  {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] p={r[3]:.3f}   placebo {pm:+.2f} ({ps:.2f})  adjusted {adj:+.2f} [{adj-h:+.2f}, {adj+h:+.2f}] z {adj/ase:+.1f}")
    else: P(f"  {lab:46} decks {r[5]:5} games {r[4]:6}  {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] p={r[3]:.3f}")
    return r
P("(pts of win rate per tapped dual in the opening hand instead of another land; lands in the opening hand and mulligans held; deck FE)")
x_all = (OH_PW + OH_SL).astype(float); has = clean & ((DK_PW + DK_SL) > 0)
P("\n## P8 primary and splits")
wrec("P8 PW + SLOW, all games", x_all, has)
wrec("PW only (decks with a PW dual)", OH_PW.astype(float), clean & (DK_PW > 0))
wrec("SLOW only (decks with a SLOW dual)", OH_SL.astype(float), clean & (DK_SL > 0))
for lab, mk in (("on the play", play == 1), ("on the draw", play == 0), ("half 0", half == 0), ("half 1", half == 1), ("skill >= 0.60", skill >= .6)):
    wrec(f"PW + SLOW, {lab}", x_all, has & mk)
cut = np.percentile(mc[has], [100 / 3, 200 / 3]); P(f"deck-speed tercile cuts (mean cmc of games' decks with the class): {cut[0]:.2f}, {cut[1]:.2f}")
for lab, mk in (("fast tercile", mc <= cut[0]), ("middle tercile", (mc > cut[0]) & (mc <= cut[1])), ("slow tercile", mc > cut[1])):
    wrec(f"PW + SLOW, {lab}", x_all, has & mk)
    wrec(f"PW + SLOW, {lab}, on the play", x_all, has & mk & (play == 1))
for lab, mk in (("WALK < 4", walk < 4), ("WALK >= 4", walk >= 4)):
    wrec(f"PW only, {lab}", OH_PW.astype(float), clean & (DK_PW > 0) & mk)
    wrec(f"PW + SLOW, {lab}", x_all, has & mk)
P("\n  Room of Refuge (always tapped, any lane) in the opening hand vs another land, clean two-colour decks containing it (secondary)")
if ROOM in kix:
    xr = O[:, kix[ROOM]].astype(float); hr = clean & (Dl[:, kix[ROOM]] > 0)
    wrec("Room of Refuge, all games", xr, hr, plac=False); wrec("Room of Refuge, on the play", xr, hr & (play == 1), plac=False); wrec("Room of Refuge, on the draw", xr, hr & (play == 0), plac=False)
open(f"{OUT}/b3b8_speed_tapped.txt", "w").write("\n".join(lines)); P("done")
