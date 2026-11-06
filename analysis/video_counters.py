"""Video test 1: do counterspells count as removal? Deck GLM as analysis/podcast_removal.py (binomial on wins, SEs clustered by draft, two-colour decks,
60%+ skill primary, all-skills sensitivity, drafts split by parity for replication). Controls: mean lane GIH WR of spells, creatures, splash, lane, skill, games bucket.
Counter sets (card-by-card review in docs/video-test-counters-sinks-splash-FRA.md): CORE = Countersculpt, Icy Reception, Precise Redaction, Sublime Epiphany (a mode counters
a spell); WIDE = CORE + Fatehold Charm, Unsummon, Consign to Memory (bounce / narrow counters). Theorix Charm is already in the removal tag (-2/-2 mode) and stays there.
Removal: REM48 = the 48 reviewed cards of B5 (removal_classes_FRA.json); REMENG = engine tags after tag_fixes (config.tags; roles.json is not used: it files counters under removal for the overlay).
Effects: win-rate points at 55% (per card; for 0/1 terms 100*.2475*b). Run:
nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_counters.py"""
import gzip, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
K, K1 = 100 * .55 * .45, 100 * .2475
OUT = f"{R}/analysis/out"
CORE = {"Countersculpt", "Icy Reception", "Precise Redaction", "Sublime Epiphany"}
WIDE = CORE | {"Fatehold Charm", "Unsummon", "Consign to Memory"}

def cards():
    raw = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
    by = {}
    for v in raw.values():
        n = v["name"].split(" // ")[0]
        if n in by and not v.get("isprimarycard", 1): continue
        by[n] = v
    return by

def load(extra=None, split_decks=False):
    """One row per two-colour deck (>=1 game, >=20 priced spells) as in podcast_removal.py; extra(counts, by) -> dict of added columns."""
    by = cards()
    rows = []
    for d in json.load(gzip.open(f"{OUT}/decks_FRA.json.gz", "rt")):
        if len(d["lane"]) != 2 or d["deck_games"] < 1: continue
        q, cre = [], 0
        for n, k in d["counts"].items():
            n = n.split(" // ")[0]; v = by.get(n)
            if not v or "Land" in v["types"]: continue
            g = (v["deck_colors"].get(d["lane"]) or {}).get("gihwr") or (v["deck_colors"].get("All Decks") or {}).get("gihwr") or 0
            if g: q += [g] * k
            cre += k * ("Creature" in v["types"])
        if len(q) < 20: continue
        r = dict(draft=d["draft"], lane=d["lane"], splash_c=d["splash"], skill=d["skill"], gb=d["games_bucket"], splash=int(d["splash"] != ""),
                 y=d["deck_wins"] / d["deck_games"], g=d["deck_games"], quality=np.mean(q), cre=cre, fmt=d["format"])
        if extra: r.update(extra({n.split(" // ")[0]: k for n, k in d["counts"].items()}, by, d))
        rows.append(r)
    return pd.DataFrame(rows)

def fit(df, f, drop_lane=False):
    if drop_lane: f = f.replace(" + C(lane)", "")
    G = pd.factorize(df.draft)[0]
    return smf.glm(f, df, family=sm.families.Binomial(), freq_weights=df.g).fit(cov_type="cluster", cov_kwds={"groups": G})

def row(m, x, k=K):
    b, se = m.params[x], m.bse[x]
    return f"| {x} | {k*b:+.2f} | [{k*(b-1.96*se):+.2f}, {k*(b+1.96*se):+.2f}] | {m.pvalues[x]:.3g} |"

def diff(m, a, b, k=K):
    A = np.zeros((1, len(m.params))); A[0, list(m.params.index).index(a)] = 1; A[0, list(m.params.index).index(b)] = -1
    t = m.t_test(A); e, se = float(np.squeeze(t.effect)), float(np.squeeze(t.sd))
    return f"{k*e:+.2f} [{k*(e-1.96*se):+.2f}, {k*(e+1.96*se):+.2f}] p {float(np.squeeze(t.pvalue)):.3g}"

CTRL = "quality + cre + splash + skill + C(gb) + C(lane)"
lines = []
def P(s=""): print(s); lines.append(s)

def main():
    RC = json.load(open(f"{OUT}/removal_classes_FRA.json"))
    from companion.advisor_v2 import config
    by = cards()
    eng = {n for n, v in by.items() if "Land" not in v["types"] and "removal" in config.tags(v, "FRA")}
    assert CORE | WIDE <= set(by), WIDE - set(by)
    P("Counter cards in the engine removal tag: " + (", ".join(sorted(eng & WIDE)) or "none") + "; in REM48: " + (", ".join(sorted(set(RC) & WIDE)) or "none"))
    P(f"REM48 {len(RC)} cards, REMENG {len(eng)} cards (REMENG includes Theorix Charm: {'Theorix Charm' in eng})")
    def extra(c, by, d):
        return dict(rem48=sum(k for n, k in c.items() if n in RC), remeng=sum(k for n, k in c.items() if n in eng),
                    ctr=sum(k for n, k in c.items() if n in CORE), ctrw=sum(k for n, k in c.items() if n in WIDE),
                    **{f"c_{n.split()[0]}": c.get(n, 0) for n in sorted(WIDE)})
    ALL = load(extra)
    ALL["s48"] = (ALL.rem48 < 6).astype(int); ALL["sE"] = (ALL.remeng < 6).astype(int)
    ALL["sE_c"] = ((ALL.remeng + ALL.ctr) < 6).astype(int); ALL["sE_cw"] = ((ALL.remeng + ALL.ctrw) < 6).astype(int)
    ALL["s48_c"] = ((ALL.rem48 + ALL.ctr) < 6).astype(int)
    ALL["ctr1"] = (ALL.ctr > 0).astype(int)
    def run(df, tag, lane_split=False):
        P(f"\n### {tag}: {len(df)} decks, {df.draft.nunique()} drafts, {int(df.g.sum())} games")
        P(f"mean per deck: rem48 {df.rem48.mean():.2f}, remeng {df.remeng.mean():.2f}, ctr(core) {df.ctr.mean():.3f}, ctr(wide) {df.ctrw.mean():.3f}; share with >=1 core counter {df.ctr1.mean():.3f}; "
          f"remeng<6 {df.sE.mean():.3f}, remeng+ctr<6 {df.sE_c.mean():.3f}; rem48<6 {df.s48.mean():.3f}")
        dl = lane_split
        for rem, nm in (("rem48", "REM48"), ("remeng", "REMENG")):
            for c, cn in (("ctr", "core"), ("ctrw", "wide")):
                m = fit(df, f"y ~ {rem} + {c} + {CTRL}", dl)
                P(f"\n(A) {nm} + counter({cn}) as own term\n\n| term | pts/card | 95% CI | p |\n|---|---|---|---|\n{row(m, rem)}\n{row(m, c)}\n\nremoval minus counter: {diff(m, rem, c)}")
        P("\n(B) shortfall (below 6 vs 6+), pts for being short, REMENG count\n\n| definition | pts | 95% CI | p |\n|---|---|---|---|")
        for s, lab in (("sE", "removal only"), ("sE_c", "removal + core counters"), ("sE_cw", "removal + wide counters")):
            m = fit(df, f"y ~ {s} + {CTRL}", dl); P(row(m, s, K1).replace(f"| {s} |", f"| {lab} |"))
        P("\n(B') same, REM48 count\n\n| definition | pts | 95% CI | p |\n|---|---|---|---|")
        for s, lab in (("s48", "REM48 only"), ("s48_c", "REM48 + core counters")):
            m = fit(df, f"y ~ {s} + {CTRL}", dl); P(row(m, s, K1).replace(f"| {s} |", f"| {lab} |"))
    for lbl, df in (("PRIMARY skill>=0.60", ALL[ALL.skill >= .60]), ("SENSITIVITY all skills", ALL)):
        P(f"\n## {lbl}"); run(df, "all drafts")
        if lbl.startswith("PRIMARY"):
            o = {d: i for i, d in enumerate(sorted(df.draft.unique()))}; h = df.draft.map(o) % 2
            for k in (0, 1): run(df[h == k], f"replication half {k + 1}")
            for ln in ("WU", "UB", "UR"):
                sub = df[df.lane.apply(lambda x: set(x) == set(ln))]
                run(sub, f"lane {ln} (lane control dropped)", True)
    open(f"{OUT}/video_css_counters.txt", "w").write("\n".join(lines))

if __name__ == "__main__":
    main()
