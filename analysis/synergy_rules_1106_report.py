"""Tables from analysis/synergy_rules_1106.py output: python3 analysis/synergy_rules_1106_report.py results.json > tables.md"""
import json, sys
import numpy as np
from scipy.stats import norm
r0 = json.load(open(sys.argv[1]))
EMPMAIN = {"Way of the Deathbringer", "Way of the Healer", "Way of the Mentor", "Way of the Mind Sculptor", "Way of the Necromancer", "Way of the Warlord", "Way of the Wildspeaker"}
def rename(k):
    c, _, l = k.partition(" | ")
    l = "empower Jace cards" if (c in EMPMAIN and l == "main") else l
    return f"{c}: {l}"
r = {rename(k): v for k, v in r0.items()}
NEW = lambda k: (k.startswith(("Ghalta", "Danitha")) or (("empower Jace cards" in k) and not k.startswith("POOLED")) or k.endswith(("fodder", "lifegain")) or k == "Way of the Paradox: main")
new = [k for k in r if NEW(k)]; zB = norm.isf(0.05 / 2 / len(new))
print(f"New lists in the family: {len(new)}; Bonferroni two-sided 0.05 line |z| >= {zB:.2f}\n")
def zpl(x):
    if not x or x["pl_mean"] is None: return None
    d = x["syn"] - x["pl_mean"]; s = np.hypot(x["syn_se"], x["pl_sd"]); return d, s, d / s
def row(k):
    v = r[k]; a, p = v["res"].get("ALL|all"), v["res"].get("P100|all")
    za, zp = zpl(a), zpl(p)
    if not za: return f"| {k} | {v['enablers']} | n/a |" + " |" * 8
    f = lambda z: "n/a" if not z else f"{z[0]:+.2f} ({z[1]:.2f}) z {z[2]:+.1f}"
    ci = f"[{za[0]-1.96*za[1]:+.1f}, {za[0]+1.96*za[1]:+.1f}]"
    cb = f"{za[0]+zB*za[1]:+.1f}"
    sg = "same" if zp and np.sign(za[0]) == np.sign(zp[0]) else "opposite"
    if k in new:
        ps = "PASS" if za[2] >= zB and zp and zp[0] > 0 else ("worse (sig)" if za[2] <= -zB else "no")
    else: ps = "(not in family)"
    return (f"| {k} | {v['enablers']} | {a['decks_with']} | {a['with_']:+.2f} / {a['alone']:+.2f} | {a['syn']:+.2f} ({a['syn_se']:.2f}) | {a['pl_mean']:+.2f} ± {a['pl_sd']:.2f} | {f(za)} {ci} | {cb} | {f(zp)} ({sg}) | {ps} |")
print("### Table 1. Synergy against placebo, all skill. WR points per enabler. Columns: lists (enablers), decks with the payoff in hand, effect with payoff in hand / payoff in deck not drawn, synergy (bootstrap SE), placebo mean ± SD (30 sets), synergy minus placebo (SE) z [95% CI], same at P100, verdict\n")
print("| list | n | decks | with / alone | synergy | placebo | ALL: synergy - placebo [95% CI] | Bonferroni upper bound | P100: synergy - placebo | verdict |\n|---|---|---|---|---|---|---|---|---|---|")
for k in r: print(row(k))
def grp(k, pop):
    v = r[k]["res"]; cells = []
    for g in ("good", "average", "weak"):
        x = v.get(f"{pop}|{g}"); cells.append("n/a" if not x else f"{x['syn']:+.2f} ({x['syn_se']:.2f}) [{x['decks_with']}]")
    a, b = v.get(f"{pop}|good"), v.get(f"{pop}|weak")
    if a and b and a["sb"] and b["sb"]:
        d = a["syn"] - b["syn"]; s = np.std(np.array(a["sb"]) - np.array(b["sb"]), ddof=1); dd = f"{d:+.2f} [{d-1.96*s:+.1f}, {d+1.96*s:+.1f}] z {d/s:+.1f}"
    else: dd = "n/a"
    return cells, dd
for pop in ("ALL", "P100"):
    print(f"\n### Table 2{'ab'[pop=='P100']}. Synergy by skill group, population {pop} (WR points per enabler, SE, [decks with payoff in hand]); good minus weak with 95% CI from one joint draft bootstrap\n")
    print("| list | good | average | weak | good minus weak [95% CI] |\n|---|---|---|---|---|")
    for k in r:
        c, d = grp(k, pop); print(f"| {k} | " + " | ".join(c) + f" | {d} |")
OFF = json.load(open(f"{__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))}/companion/advisor_v2/sets/FRA/fitted.json"))["card_needs"]
print("\n### Table 3. card_needs.py design (logistic product term: extra WR points per deck enabler when the payoff is drawn, SE). Official tool run (ALL players, deck quality, skill and play/draw controlled) and our port by population (skill and play/draw controlled, no deck-quality term, so ALL differs slightly from the tool)\n")
print("| list | tool, ALL | port, ALL | port, P100 | P100 good | P100 average | P100 weak | sign vs within-deck synergy minus placebo (ALL) |\n|---|---|---|---|---|---|---|---|")
for k0, v in r0.items():
    k = rename(k0); c, _, l = k0.partition(" | "); ok = c if l == "main" else f"{c}: {l}"
    o = OFF.get(ok); n = v["needs"]; f = lambda key: "n/a" if key not in n else f"{n[key]['est']:+.2f} ({n[key]['se']:.2f})"
    z = zpl(r[k]["res"].get("ALL|all"))
    sg = ("same" if np.sign(o["per_enabler_in_hand"]) == np.sign(z[0]) else "DIFFERS") if (o and z) else "n/a"
    print(f"| {k} | {'n/a' if not o else '%+.2f (%.2f) z %+.1f' % (o['per_enabler_in_hand'], o['se'], o['z'])} | {f('ALL|all')} | {f('P100|all')} | {f('P100|good')} | {f('P100|average')} | {f('P100|weak')} | {sg} |")
# group-difference noise count
for pop in ("ALL", "P100"):
    zs = []
    for k in r:
        a, b = r[k]["res"].get(f"{pop}|good"), r[k]["res"].get(f"{pop}|weak")
        if a and b and a["sb"] and b["sb"]:
            zs.append((a["syn"] - b["syn"]) / np.std(np.array(a["sb"]) - np.array(b["sb"]), ddof=1))
    zs = np.array(zs); print(f"\nGood minus weak, {pop}: {len(zs)} lists, |z| > 1.96 in {(abs(zs) > 1.96).sum()} (expected {0.05*len(zs):.1f}), |z| > {norm.isf(0.025/len(zs)):.2f} (Bonferroni) in {(abs(zs) > norm.isf(0.025/len(zs))).sum()}; largest {zs[np.argmax(abs(zs))]:+.1f}")
