"""Why do some builds have 38-39 cards? Per build of the player's pool: spells/nonbasics/basics counts and the variant. Also v2 duplicate check."""
import gzip, json, sys, collections as C
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.replay import run
from companion import session
from companion.advisor_v2 import config, deckbuilder
app = run.replay.App(str(run.dataset_path("FRA"))); run.use_advisor("v2"); config.activate("FRA")
drafts = json.load(gzip.open("/home/morris/Projects/mtga-draft-companion/data/Temp/adv1/sp3.drafts.json.gz", "rt"))[:int(sys.argv[1])]
res = json.load(open("/home/morris/Projects/mtga-draft-companion/data/Temp/adv1/deck_full.json"))
tot = C.Counter(); ex = []
targets = config.fitted("FRA", "decks").get("pairs"); model = session._deck_model("FRA", deckbuilder.MODEL); yard = session._deck_model("FRA")
for d in drafts:
    pool = app.cards([p[2] for p in d["picks"]])
    for b in deckbuilder.build(pool, model, targets, rule=config.fitted("FRA", "splash"), three=config.fitted("FRA", "three"), yardstick=yard):
        n_sp, n_nb, n_ba = len(b["spells"]), len(b["nonbasics"]), sum(b["basics"].values()); t = n_sp + n_nb + n_ba
        tot[(b["variant"], t)] += 1
        if t != 40 and len(ex) < 6: ex.append((d["id"][:8], b["lane"], b.get("splash"), b["variant"], n_sp, n_nb, n_ba, t, dict(b["basics"])))
print(sorted(tot.items(), key=str)); [print(e) for e in ex]

bad = 0; n = 0
for d in drafts:
    pool = app.cards([p[2] for p in d["picks"]])
    for b in session.own_builds("FRA", pool)[:1]:
        lst = sum(c["count"] for c in b["deck"]); dct = len({c["name"] for c in b["deck"]}); names = [c["name"] for c in b["deck"]]
        dup = [k for k, v in C.Counter(names).items() if v > 1]; n += 1
        if lst != 40 or dup: bad += 1; print("OWN", d["id"][:8], b["label"], "list total", lst, "entries", len(names), "dup names", dup, "pool basics", [c["name"] for c in pool if "Basic" in c["types"]])
print("own_builds[0]: list total != 40 or duplicate names in", bad, "of", n)

for d in drafts:
    pool = app.cards([p[2] for p in d["picks"]])
    b = session.own_builds("FRA", pool)[0]; names = [c["name"] for c in b["deck"]]
    if len(set(names)) != len(names):
        print(d["id"][:8], b["label"], [(c["name"], c["count"], c["types"]) for c in b["deck"] if c["name"] in deckbuilder.BASIC.values()]); break
