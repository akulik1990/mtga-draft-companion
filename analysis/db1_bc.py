"""builder_check variant that passes the yardstick (builder_check.build omits it, so DEFAULT_BEST has no effect there).
uv run python analysis/db1_bc.py FRA OUT.json [--v2 K=V ...]"""
import argparse, json, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import builder_check as bc
from companion.advisor_v2 import deckbuilder, config
from companion import session

Y = {}

def init(set_code, ov):
    bc.init(set_code, ov)
    Y["y"] = session._deck_model(set_code)

def one(item):
    lane, splash, deck, pool = item
    A = bc.ARGS
    cards = [A["by_name"][n] for n, k in pool.items() if n in A["by_name"] for _ in range(k)]
    t = time.time()
    builds = deckbuilder.build(cards, A["model"], A["targets"], top=1, rule=A["rule"], three=A["three"], yardstick=Y["y"])
    dt = time.time() - t
    if not builds:
        return None
    b = builds[0]
    ours = Counter(c["name"].split(" // ")[0] for c in b["spells"])
    theirs = Counter({n: k for n, k in deck.items() if n in A["by_name"] and "Land" not in (A["by_name"][n].get("types") or [])})
    return {"lane": lane, "their_splash": splash, "our_lane": b["lane"], "our_splash": b["splash"] or "",
            "overlap": sum((ours & theirs).values()) / max(1, sum(theirs.values())),
            "yardstick": b.get("yardstick") or deckbuilder.yardstick_wr(b, Y["y"]), "variant": b.get("variant"), "secs": dt}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("set"); p.add_argument("out"); p.add_argument("--n", type=int, default=300)
    p.add_argument("--jobs", type=int, default=3); p.add_argument("--v2", action="append", default=[])
    a = p.parse_args()
    init(a.set, a.v2)
    items = bc.sample(a.set, a.n, colours=(2, 3), buckets=bc.CHECK)
    with Pool(a.jobs, initializer=init, initargs=(a.set, a.v2)) as pool:
        rows = [r for r in pool.map(one, items, chunksize=2) if r]
    Path(a.out).write_text(json.dumps(rows))
    n = len(rows)
    print(a.v2, n, "pools; mean secs/pool", sum(r["secs"] for r in rows) / n)
