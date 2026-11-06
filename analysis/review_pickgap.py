"""Translate a WR-point effect into picks: from the per-pick replay log (data/Temp/replay/logs/head-78e1953.json, v2 at 78e1953 on the
600 train drafts) the gap between v2's top-1 and top-2 scores, overall and when the two cards differ on a tested trait (REM48 removal,
cmc >= 5, gold, counter, Way). A per-card bonus of b points can only change picks where the gap is under b and the trait differs.
Output analysis/out/review_pickgap.txt."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_counters import R, OUT, cards
by = cards(); RC = set(json.load(open(f"{OUT}/removal_classes_FRA.json")))
CORE = {"Countersculpt", "Icy Reception", "Precise Redaction", "Sublime Epiphany"}
L = json.load(open(f"{R}/data/Temp/replay/logs/head-78e1953.json"))
traits = {"removal (REM48)": lambda n: n in RC, "cmc >= 5": lambda n: n in by and "Land" not in by[n]["types"] and (by[n]["cmc"] or 0) >= 5,
          "gold": lambda n: n in by and len(by[n]["colors"]) >= 2, "counter (CORE)": lambda n: n in CORE, "Way": lambda n: n.startswith("Way of the"),
          "land": lambda n: n in by and "Land" in by[n]["types"]}
gaps, trait_gaps = [], {k: [] for k in traits}
for d, pk, pi, ranked, _ in L:
    if len(ranked) < 2: continue
    (a, sa), (b, sb) = ranked[0], ranked[1]; g = sa - sb; gaps.append(g)
    for k, f in traits.items():
        if f(a) != f(b): trait_gaps[k].append(g)
gaps = np.array(gaps); out = []
def P(s): print(s); out.append(s)
P(f"picks {len(gaps)}; top-1 minus top-2 score gap: median {np.median(gaps):.2f}, quartiles {np.quantile(gaps, .25):.2f} / {np.quantile(gaps, .75):.2f}")
P("\n| bonus b (pts) | share of all picks with gap < b | " + " | ".join(f"gap < b and {k} differs" for k in traits) + " |\n|---|---|" + "---|" * len(traits))
for b in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
    P(f"| {b} | {(gaps < b).mean():.3f} | " + " | ".join(f"{(np.array(v) < b).sum() / len(gaps):.3f}" for v in trait_gaps.values()) + " |")
P("\nShare of picks where the trait differs between top-1 and top-2 at all: " + ", ".join(f"{k} {len(v) / len(gaps):.3f}" for k, v in trait_gaps.items()))
P("One pick on the 600 train drafts (about 25,000 picks) is 0.004 top-1 points; the replay noise floor on a 100-draft slice is about 0.3 top-1 points (docs/spec.md).")
open(f"{OUT}/review_pickgap.txt", "w").write("\n".join(out))
