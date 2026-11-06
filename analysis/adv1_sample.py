"""Fix the adv1 sample before any outcome is read: FRA top600 TEST drafts (skill>=0.60, not tuned on) joined to decks_FRA.json.gz,
exactly one registered deck per draft, PremierDraft. splash = 2-colour base + 1-2 splash colours; three = 3-colour base (+/- splash);
control = plain two-colour (no splash). seed 20261106: 60 splash + 40 three (fewer if short), 100 control."""
import gzip, json, random, collections as C
R = "/home/morris/Projects/mtga-draft-companion/"
decks = C.defaultdict(list)
for x in json.load(gzip.open(R + "analysis/out/decks_FRA.json.gz")): decks[x["draft"]].append(x)
dr = json.load(gzip.open(R + "data/Temp/replay/FRA.top600.test.drafts.json.gz"))
grp = C.defaultdict(list)
for d in dr:
    L = decks.get(d["id"])
    if not L or len(L) != 1 or L[0]["format"] != "PremierDraft" or d["skill"] < 0.6: continue
    k = L[0]
    kind = "three" if len(k["lane"]) == 3 else "splash" if len(k["lane"]) == 2 and k["splash"] else "two" if len(k["lane"]) == 2 else None
    if kind: grp[kind].append((d, k))
print({k: len(v) for k, v in grp.items()})
rng = random.Random(20261106)
pick = {"splash": rng.sample(grp["splash"], min(60, len(grp["splash"]))), "three": rng.sample(grp["three"], min(40, len(grp["three"]))),
        "two": rng.sample(grp["two"], min(100, len(grp["two"])))}
meta = {}
for kind, v in pick.items():
    for d, k in v: meta[d["id"]] = dict(kind=kind, lane=k["lane"], splash=k["splash"], skill=d["skill"], counts=k["counts"])
print({k: len(v) for k, v in pick.items()})
json.dump(meta, open(R + "data/Temp/adv1/meta.json", "w"))
for name, kinds in [("sp3", ["splash", "three"]), ("two", ["two"])]:
    out = [d for kk in kinds for d, _ in pick[kk]]
    json.dump(out, gzip.open(R + f"data/Temp/adv1/{name}.drafts.json.gz", "wt"))
    print(name, len(out))
