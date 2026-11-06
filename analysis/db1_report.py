import json
import numpy as np
from scipy import stats
O = "/home/morris/Projects/mtga-draft-companion/data/Temp/sim/db1/"
b, n = json.load(open(O + "mock_base.json")), json.load(open(O + "mock_new.json"))
a = np.array([r["v2"]["deck_wr"] for r in b]); c = np.array([r["v2"]["deck_wr"] for r in n])
d = c - a; m = d.mean(); se = d.std(ddof=1) / len(d) ** 0.5
t = stats.ttest_rel(c, a); h = stats.t.ppf(.975, len(d) - 1) * se
print(f"n={len(d)} base {a.mean():.2f} new {c.mean():.2f} diff {m:+.2f} CI [{m-h:+.2f},{m+h:+.2f}] t={t.statistic:.2f} p={t.pvalue:.3g}; better {sum(d>0)} worse {sum(d<0)} same {sum(d==0)}; sd {d.std(ddof=1):.2f}")
print("picks identical:", all(x["v2"]["picks"] == y["v2"]["picks"] for x, y in zip(b, n)))
def splashn(lbl): return lbl.count("+") if lbl else 0
for nm, rows in (("base", b), ("new", n)):
    L = [r["v2"]["label"] or "" for r in rows]
    three = sum(len(l.split(" ")[0]) == 3 for l in L)
    print(nm, "three-col", three, "splash0", sum(len(l.split(" ")[0]) == 2 and "+" not in l for l in L), "splash1", sum(len(l.split(" ")[0]) == 2 and l.count("+") == 1 for l in L), "splash2+", sum(l.count("+") >= 2 for l in L), "three+splash", sum(len(l.split(" ")[0]) == 3 and "+" in l for l in L))
for i in np.argsort(-abs(d))[:3]:
    print("seed", i + 1, f"{a[i]:.1f} -> {c[i]:.1f}", b[i]["v2"]["label"], "->", n[i]["v2"]["label"])
