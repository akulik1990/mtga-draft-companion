"""Review check for the double-pip splash test (docs/video-test-counters-sinks-splash-FRA.md, part 3).
Three things the original did not do: (1) an ordinal model over source count inside DP-splash decks instead of post-hoc cuts;
(2) a negative control: the same source-count model inside single-pip splash decks (group B), where castability of a double pip
cannot be the mechanism, so any 8+ jump there is a "drafted its fixing / better deck" proxy; (3) a held-out look by format:
the 8 cut was chosen on all formats pooled, so Premier vs Trad+PickTwo gives two separate looks (not a clean pre-registered split,
but independent decks). Output analysis/out/review_dpsplash.txt.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/review_dpsplash.py"""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_counters import R, OUT, K, K1, cards, load, fit, row, diff

BASIC = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
lines = []
def P(s=""): print(s); lines.append(s)

by = cards()
dbl, lands, cyc = {}, {}, set()
for n, v in by.items():
    front = (v.get("mana_cost") or "").split("//")[0]; s = re.findall(r"\{([WUBRG])\}", front)
    dbl[n] = next((c for c in "WUBRG" if s.count(c) >= 2), None)
    t = (v.get("oracle_text") or "")
    if "Land" in v["types"]:
        base = n.replace("Snow-Covered ", "")
        lands[n] = set("WUBRG") if n == "Room of Refuge" else {BASIC[base]} if base in BASIC else {c for c in (v.get("colors") or []) if c in "WUBRG"}
    elif re.search(r"landcycling", t, re.I): cyc.add(n)

def extra(c, by, d):
    sp = d["splash"]
    if len(sp) != 1: return dict(sp=sp, dp=0, src=0, ok=0, spq=np.nan)
    src = sum(k for n, k in c.items() if (n in lands and sp in lands[n]) or n in cyc or n == "Murmuring Volume")
    dp = sum(k for n, k in c.items() if dbl.get(n) == sp and n not in cyc)
    # mean All Decks GIH of the splashed (off-lane) spells: the "bomb" selection the groups differ on
    spq = [(by[n]["deck_colors"].get("All Decks") or {}).get("gihwr") or np.nan for n in c
           if n in by and "Land" not in by[n]["types"] and sp in (by[n].get("colors") or []) and not set(by[n]["colors"]) <= set(d["lane"])]
    return dict(sp=sp, dp=dp, src=src, ok=1, spq=np.nanmean(spq) if spq else np.nan)

A = load(extra)
A = A[(A.splash_c == "") | (A.ok == 1)].copy()
A["grp"] = np.where(A.splash == 0, "A", np.where(A.dp == 0, "B", np.where(A.src < 7, "C", "D")))
ctrl = "quality + cre + skill + C(gb) + C(lane)"
S = A[A.splash == 1].copy()
S["srcc"] = S.src.clip(3, 10)
S["hasdp"] = (S.dp > 0).astype(int)
S["premier"] = (S.fmt == "PremierDraft").astype(int)

P("# Review: double-pip splash and source count\n")
P(f"splash decks (single-colour splash): {len(S)}; with DP copy {int(S.hasdp.sum())}; formats {S.fmt.value_counts().to_dict()}")
P("\nsources distribution, DP decks vs single-pip decks (all skills):")
P("DP:  " + str(S[S.hasdp == 1].srcc.value_counts().sort_index().to_dict()))
P("noDP:" + str(S[S.hasdp == 0].srcc.value_counts().sort_index().to_dict()))

def ordinal(df, tag):
    """WR by source bucket (3-4 ref) within a set of splash decks, quality controlled; plus linear slope per source."""
    d = df.copy(); d["sb"] = pd.cut(d.srcc, [0, 4, 5, 6, 7, 8, 99], labels=["<=4", "5", "6", "7", "8", "9+"]).astype(str)
    if d.sb.nunique() < 3 or len(d) < 60: P(f"{tag}: too few ({len(d)})"); return
    m = fit(d, f"y ~ C(sb, Treatment('<=4')) + {ctrl}")
    P(f"\n{tag}: {len(d)} decks; by sources vs <=4 (pts)\n\n| sources | n | pts | 95% CI | p |\n|---|---|---|---|---|")
    for b in ["5", "6", "7", "8", "9+"]:
        x = f"C(sb, Treatment('<=4'))[T.{b}]"
        if x in m.params: P(row(m, x, K1).replace(x, f"{b} | {int((d.sb == b).sum())}"))
    m2 = fit(d, f"y ~ srcc + {ctrl}"); P(f"linear, per source (3..10 clipped): {row(m2, 'srcc', K1)}")
    m3 = fit(d, f"y ~ srcc + spq + {ctrl}"); P(f"linear, per source, splashed-card quality (spq) also controlled: {row(m3, 'srcc', K1)}  spq: {row(m3, 'spq', K1)}")

for lbl, df in (("ALL SKILLS", S), ("skill>=0.60", S[S.skill >= .6])):
    P(f"\n## {lbl}")
    ordinal(df[df.hasdp == 1], "DP-splash decks (the claim)")
    ordinal(df[df.hasdp == 0], "single-pip splash decks (negative control: no DP castability mechanism)")
    # interaction: does the source slope differ between DP and single-pip decks?
    d = df.copy()
    m = fit(d, f"y ~ srcc * hasdp + {ctrl}")
    P(f"\ninteraction srcc x hasdp (extra per-source slope in DP decks): {row(m, 'srcc:hasdp', K1)}; main srcc {row(m, 'srcc', K1)}; hasdp {row(m, 'hasdp', K1)}")
    d["s8"] = (d.srcc >= 8).astype(int)
    m = fit(d, f"y ~ s8 * hasdp + {ctrl}")
    P(f"8+ cut: s8 in single-pip decks {row(m, 's8', K1)}; extra in DP decks (s8:hasdp) {row(m, 's8:hasdp', K1)}")

P("\n## Held-out by format (the 8 cut was chosen on all formats pooled)")
for f_, df in (("PremierDraft", S[S.fmt == "PremierDraft"]), ("Trad+PickTwo", S[S.fmt != "PremierDraft"])):
    d = df[df.hasdp == 1].copy(); d["s8"] = (d.srcc >= 8).astype(int); d["s7"] = (d.srcc >= 7).astype(int)
    if d.s8.nunique() < 2: continue
    m8 = fit(d, f"y ~ s8 + {ctrl}"); m7 = fit(d, f"y ~ s7 + {ctrl}"); ml = fit(d, f"y ~ srcc + {ctrl}")
    P(f"{f_}: DP decks {len(d)} (8+: {int(d.s8.sum())}); 8+ vs <8 {row(m8, 's8', K1)}; 7+ vs <7 {row(m7, 's7', K1)}; per source {row(ml, 'srcc', K1)}")
    d = df[df.hasdp == 0].copy(); d["s8"] = (d.srcc >= 8).astype(int)
    m8 = fit(d, f"y ~ s8 + {ctrl}"); P(f"{f_}: single-pip control {len(d)} (8+: {int(d.s8.sum())}); 8+ vs <8 {row(m8, 's8', K1)}")

P("\n## Selection: splashed-card quality and skill by sources (DP decks, all skills)")
g = S[S.hasdp == 1].groupby("srcc").agg(n=("y", "size"), skill=("skill", "mean"), quality=("quality", "mean"), spq=("spq", "mean"), wr=("y", "mean"))
P(g.round(3).to_string())
open(f"{OUT}/review_dpsplash.txt", "w").write("\n".join(lines))
