"""Video test 3: is a double-pip splash fine with 7+ real sources? (-2.03 per copy pooled, tools/deck_features.py dp_splash.)
Population: two-colour decks (as video_counters.py) whose dataset splash flag is ONE colour s (multi-colour splashes excluded: 3% of splash decks).
Double pip: front-face mana cost has >= 2 plain {C} pips of the same colour (hybrid pips not counted; same rule as tools/deck_features.py). DP copies = copies of cards whose double colour is s, landcyclers excluded (they are splashed to be cycled for a land, not cast: Hexhaven Battalion, Apex Witchstalker, Vinelasher Adept).
Effective sources of s (each card counted once, whatever it does): basic lands of s (Snow-Covered included) + non-basic lands that make s (card colours; Room of Refuge counts for any colour)
  + basic landcyclers (4 cyclers + Vinelasher Adept = 5 cards with "landcycling", each 1 whatever the colour) + Murmuring Volume. Gideon's Memorial (restricted mana) and heartwood tokens are not counted.
Groups: A clean two-colour (no splash); B splash with no double-pip card of s; C splash with DP copies and < 7 sources; D the same with 7+ sources.
Model: binomial GLM on wins, SEs clustered by draft; controls mean lane GIH WR of spells, creatures, skill, games bucket, lane (no splash term: the groups hold it). Effects in WR points at 55% (24.75 x coefficient for group dummies).
Per-copy form (comparable with the -2.03): within splash decks, DP copies split by sources (<7, 7+) with sources as a covariate. Run:
nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_dpsplash.py"""
import os, re, sys, warnings
warnings.filterwarnings("ignore")
import json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_counters import R, OUT, K, K1, cards, load, fit, row, diff

BASIC = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
lines = []
def P(s=""): print(s); lines.append(s)

def main():
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
    P(f"Landcyclers counted as sources: {sorted(cyc)}; Murmuring Volume; double-pip cards in set: {sum(1 for x in dbl.values() if x)}")
    RC = json.load(open(f"{OUT}/removal_classes_FRA.json"))
    def extra(c, by, d):
        sp = d["splash"]
        if len(sp) != 1: return dict(sp=sp, dp=0, src=0, nsp=0, ok=0, dpcards="", rem_sp=0, rem_base=0, dpq=np.nan)
        src = sum(k for n, k in c.items() if (n in lands and sp in lands[n]) or n in cyc or n == "Murmuring Volume")
        dp = sum(k for n, k in c.items() if dbl.get(n) == sp and n not in cyc)
        nsp = sum(k for n, k in c.items() if n in by and "Land" not in by[n]["types"] and sp in (by[n].get("colors") or []) and sp not in d["lane"])
        return dict(sp=sp, dp=dp, src=src, nsp=nsp, ok=1,
                    dpcards="|".join(n for n in c if dbl.get(n) == sp and n not in cyc),
                    rem_sp=sum(k for n, k in c.items() if n in RC and sp in (by[n].get("colors") or []) and not set(by[n]["colors"]) & set(d["lane"])),
                    rem_base=sum(k for n, k in c.items() if n in RC and (not (by[n].get("colors") or []) or set(by[n]["colors"]) <= set(d["lane"]))),
                    dpq=np.mean([(by[n]["deck_colors"].get("All Decks") or {}).get("gihwr") or np.nan for n in c if dbl.get(n) == sp and n not in cyc] or [np.nan]))
    A = load(extra)
    A = A[(A.splash_c == "") | (A.ok == 1)].copy()
    A["grp"] = np.where(A.splash == 0, "A", np.where(A.dp == 0, "B", np.where(A.src < 7, "C", "D")))
    A["hi"] = (A.src >= 7).astype(int); A["dp_lo"] = A.dp * (1 - A.hi); A["dp_hi"] = A.dp * A.hi
    A["dp_sink"] = np.where(A.dp > 0, A.dp, 0)
    ctrl = "quality + cre + skill + C(gb) + C(lane)"
    def run(df, tag):
        P(f"\n### {tag}: {len(df)} decks, {df.draft.nunique()} drafts, {int(df.g.sum())} games")
        P("groups (decks, mean WR %): " + "; ".join(f"{g} {int((df.grp==g).sum())} ({100*np.average(df[df.grp==g].y, weights=df[df.grp==g].g):.1f})" if (df.grp==g).any() else f"{g} 0" for g in "ABCD"))
        if min((df.grp == g).sum() for g in "CD") < 10: P("too few decks in C or D for a fit"); return
        m = fit(df, f"y ~ C(grp, Treatment('A')) + {ctrl}")
        P("\nvs clean two-colour (A), WR pts\n\n| group | pts | 95% CI | p |\n|---|---|---|---|")
        for g in "BCD":
            x = f"C(grp, Treatment('A'))[T.{g}]"; P(row(m, x, K1).replace(x, {"B": "B splash, no DP", "C": "C DP splash, <7 sources", "D": "D DP splash, 7+ sources"}[g]))
        t = lambda g: f"C(grp, Treatment('A'))[T.{g}]"
        P(f"\nD minus C: {diff(m, t('D'), t('C'), K1)}\nC minus B: {diff(m, t('C'), t('B'), K1)}; D minus B: {diff(m, t('D'), t('B'), K1)}")
        S = df[df.splash == 1]
        m2 = fit(S, f"y ~ dp_lo + dp_hi + src + {ctrl}")
        P("\nper DP copy within splash decks (reference: splash decks with no DP copy), WR pts per copy; src = per extra source\n\n| term | pts | 95% CI | p |\n|---|---|---|---|")
        P(row(m2, "dp_lo")); P(row(m2, "dp_hi")); P(row(m2, "src", K1)); P(f"\ndp_hi minus dp_lo: {diff(m2, 'dp_hi', 'dp_lo')}")
        m3 = fit(S, f"y ~ dp + src + {ctrl}"); P(f"pooled per DP copy (comparable to -2.03): {row(m3, 'dp')}")
        S2 = S[S.dp > 0].copy(); S2["s8"] = (S2.src >= 8).astype(int)
        for c in (6, 8):
            S2["h"] = (S2.src >= c).astype(int)
            if S2.h.nunique() == 2: m4 = fit(S2, f"y ~ h + {ctrl}"); P(f"within DP-splash decks, {c}+ vs <{c} sources: {row(m4, 'h', K1).replace('| h |', '| ' + str(c) + '+ |')} (n {int(S2.h.sum())} / {int((1-S2.h).sum())})")
    for lbl, df in (("PRIMARY skill>=0.60", A[A.skill >= .60]), ("SENSITIVITY all skills", A)):
        P(f"\n## {lbl}"); run(df, "all drafts")
        if lbl.startswith("PRIMARY"):
            o = {d: i for i, d in enumerate(sorted(df.draft.unique()))}; h = df.draft.map(o) % 2
            for k in (0, 1): run(df[h == k], f"replication half {k + 1}")
    # selection and descriptives (all skills, all two-colour decks with a single-colour or no splash)
    P("\n## Selection: who splashes a double pip (all skills)\n\n| group | decks | skill | mean GIH of spells | creatures | splash-colour cards | sources of s | removal (REM48) in splash colour only / base |\n|---|---|---|---|---|---|---|---|")
    for g in "ABCD":
        x = A[A.grp == g]; P(f"| {g} | {len(x)} | {x.skill.mean():.3f} | {x.quality.mean():.2f} | {x.cre.mean():.1f} | {x.nsp.mean():.1f} | {x.src.mean():.1f} | {x.rem_sp.mean():.2f} / {x.rem_base.mean():.2f} |" if g != "A" else f"| A | {len(x)} | {x.skill.mean():.3f} | {x.quality.mean():.2f} | {x.cre.mean():.1f} | - | - | {x.rem_base.mean():.2f} base |")
    sp = A[A.splash == 1]
    P(f"\nsplash decks with a DP copy: {(sp.dp>0).mean():.3f} of {len(sp)}; sources distribution of those (decks): " + str(sp[sp.dp > 0].src.value_counts().sort_index().to_dict()))
    P("splash decks without DP, sources: " + str(sp[sp.dp == 0].src.value_counts().sort_index().to_dict()))
    P("splash colour in DP decks: " + str(sp[sp.dp > 0].sp.value_counts().to_dict()) + "; in other splash decks: " + str(sp[sp.dp == 0].sp.value_counts().to_dict()))
    P(f"mean GIH WR of the double-pip card(s) in C+D decks {A[A.grp.isin(['C','D'])].dpq.mean():.2f} vs mean GIH of the same decks' spells {A[A.grp.isin(['C','D'])].quality.mean():.2f}")
    cnt = {}
    for s in sp[sp.dp > 0].dpcards:
        for n in s.split("|"): cnt[n] = cnt.get(n, 0) + 1
    P("most splashed double-pip cards (decks, GIH WR %): " + ", ".join(f"{n} ({k}, {(by[n]['deck_colors'].get('All Decks') or {}).get('gihwr')})" for n, k in sorted(cnt.items(), key=lambda t: -t[1])[:12]))
    open(f"{OUT}/video_css_dpsplash.txt", "w").write("\n".join(lines))

if __name__ == "__main__":
    main()
