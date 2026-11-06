"""Describe how the new builder's mock decks differ from the base's (v2 side). python3 analysis/dm2_deck_diff.py <base.json> <new.json>"""
import json, sys
from statistics import mean
from tools.deck_model import card_facts

a, b = (json.load(open(p)) for p in sys.argv[1:3])
facts = card_facts("FRA")
B = {r["seed"]: r for r in b}


def shape(deck):
    sp = [(n, k) for n, k in deck.items() if n in facts and not facts[n]["land"]]
    ls = [(n, k) for n, k in deck.items() if n in facts and facts[n]["land"]]
    ns = sum(k for n, k in ls if not facts[n]["basic"] and n not in ("Plains", "Island", "Swamp", "Mountain", "Forest"))
    cm = [facts[n]["cmc"] for n, k in sp for _ in range(k)]
    return {"spells": sum(k for _, k in sp), "lands": sum(k for _, k in ls), "nonbasic": ns,
            "creatures": sum(k for n, k in sp if facts[n]["creature"]), "removal": sum(k for n, k in sp if facts[n]["removal"]),
            "avg_cmc": mean(cm), "cmc<=2": sum(c <= 2 for c in cm), "cmc>=5": sum(c >= 5 for c in cm)}


rows = []
for x in a:
    y = B[x["seed"]]
    dx, dy = x["v2"], y["v2"]
    rows.append((dy["deck_wr"] - dx["deck_wr"], x["seed"], dx, dy))
diff = [r for r in rows if r[2]["deck"] != r[3]["deck"]]
print(f"differing decks {len(diff)}/100; deck_wr change among them mean {mean(r[0] for r in diff):+.2f}; up {sum(r[0]>0 for r in diff)} down {sum(r[0]<0 for r in diff)}")
def splash(d): return len(d["label"] or "") > 2 if d.get("label") else False
for nm, i in (("base", 2), ("new", 3)):
    S = [shape(r[i]["deck"]) for r in rows]
    cols = [len(r[i]["label"] or "") for r in rows]
    print(nm, {k: round(mean(s[k] for s in S), 2) for k in S[0]}, "label-len mean", round(mean(cols), 2), "labels", {l: sum((r[i]["label"] or "") == l for r in rows) for l in sorted({r[i]["label"] for r in rows if r[i]["label"]}, key=str)} if False else "")
import collections
print("label examples", [(r[2]["label"], r[3]["label"]) for r in diff[:5]])
print("splash flag (label has >2 colours): base", sum(len((r[2]["label"] or "").replace("/", "").replace(" ", ""))>2 for r in rows), "new", sum(len((r[3]["label"] or "").replace("/", "").replace(" ", ""))>2 for r in rows))
for dwr, seed, dx, dy in sorted(rows, key=lambda r: -abs(r[0]))[:5]:
    print(f"\n=== seed {seed}: deck_wr {dx['deck_wr']:.2f} -> {dy['deck_wr']:.2f} ({dwr:+.2f}); base {dx['label']} / new {dy['label']}")
    sx, sy = shape(dx["deck"]), shape(dy["deck"])
    print("  base", {k: round(v, 2) for k, v in sx.items()}); print("  new ", {k: round(v, 2) for k, v in sy.items()})
    out = {n: k - dy["deck"].get(n, 0) for n, k in dx["deck"].items() if k > dy["deck"].get(n, 0)}
    inn = {n: k - dx["deck"].get(n, 0) for n, k in dy["deck"].items() if k > dx["deck"].get(n, 0)}
    f = lambda d: "; ".join(f"{k}x {n} (cmc{facts[n]['cmc'] if n in facts else '?'}{',C' if n in facts and facts[n]['creature'] else ''}{',R' if n in facts and facts[n]['removal'] else ''}{',L' if n in facts and facts[n]['land'] else ''})" for n, k in d.items())
    print("  out:", f(out)); print("  in: ", f(inn))
