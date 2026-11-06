"""Part 2(a,b): rebuild registered FRA pools with our builder and compare the deck model's rating of their 40 with ours.

    uv run python analysis/builder_gap_pools.py read      # one pass over the game CSV -> data/Temp/gap1106/pools_FRA.json.gz  (a few minutes)
    uv run python analysis/builder_gap_pools.py draw      # sample per cell -> pool_sample.json.gz
    uv run python analysis/builder_gap_pools.py build     # sample per cell, build in parallel -> data/Temp/gap1106/pool_gaps.jsonl
    uv run python analysis/builder_gap_pools.py report    # -> analysis/out/builder_gap_pools.txt

FIXED BEFORE LOOKING AT ANY GAP:
  Population: builds (draft, build_index) of drafts in analysis/out/decks_FRA.json.gz (100+ games players), 2- or 3-colour main lane.
  Classes: 2+spl = 2-colour lane with 2+ splash colours; 1spl = 2-lane, 1 splash colour; clean2 = 2-lane, none; c45 = 3-lane with a splash (4-5 colours);
           tri = 3-lane, none. Skill groups by the player's bucket: good >= 0.60, average 0.52-0.58, weak < 0.52.
  Sample: up to CAPS (2+spl 200, 1spl 200, c45 150, tri 100, clean2 100) builds per class x group, taken in md5(draft|build) order (all of a cell when it has fewer).
  Gap = deck model "deckmodel" rating of their registered 40 minus the rating of our build from the same pool (session.deck_win_rate rule:
        counts incl. basics, main = lane, splash = colours outside it), two builder variants: A as configured (OBJECTIVE gihwr, MODEL deckmodel_top3),
        B maximising the yardstick itself (OBJECTIVE model, MODEL deckmodel; the splash/three rules and all limits unchanged).
  Primary contrast: gap(2+spl) - gap(1spl), variant A, all skill groups pooled; then by group and good - weak. Secondary: c45 vs tri; variant B.
  CIs: bootstrap over drafts (2000 resamples). Halves: md5 parity of the draft id.
"""

import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

from builder_gap_common import *  # noqa: F401,F403

TMP = REPO / "data" / "Temp" / "gap1106"
CAPS = {"2+spl": 200, "1spl": 200, "c45": 150, "tri": 100, "clean2": 100}  # per skill group
CSV = DATA_DIR / "Temp" / "synergy" / "game_data_public.FRA.PremierDraft.csv.gz"


def cls(lane, S):
    n = len(S)
    return ("2+spl" if n >= 2 else "1spl" if n == 1 else "clean2") if len(lane) == 2 else ("c45" if n >= 1 else "tri")


def sgroup(s):
    return "good" if s >= .6 - 1e-9 else "avg" if .52 - 1e-9 <= s <= .58 + 1e-9 else "weak" if s < .52 - 1e-9 else None


def read():
    drafts = {d["draft"] for d in json.load(gzip.open(OUT / "decks_FRA.json.gz", "rt"))}
    out, seen = [], set()
    with gzip.open(CSV, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[5:]) for i, h in enumerate(header) if h.startswith("deck_")]
        side = {h[10:]: i for i, h in enumerate(header) if h.startswith("sideboard_")}
        for r, row in enumerate(reader):
            if r % 200000 == 0:
                print("rows", r, "builds", len(out), flush=True)
            if row[col["draft_id"]] not in drafts:
                continue
            key = (row[col["draft_id"]], row[col["build_index"]])
            skill, pair = row[col["user_game_win_rate_bucket"]], row[col["main_colors"]]
            if key in seen or len(pair) not in (2, 3) or skill in ("", "None"):
                continue
            seen.add(key)
            deck, pool = {}, {}
            for i, name in deck_cols:
                d = int(float(row[i] or 0))
                s = int(float(row[side[name]] or 0)) if name in side else 0
                if d:
                    deck[name] = d
                if d + s:
                    pool[name] = d + s
            lane = "".join(sorted(pair, key="WUBRG".index))
            out.append(dict(draft=key[0], build=key[1], skill=float(skill), lane=lane, splash=row[col["splash_colors"]], deck=deck, pool=pool))
    with gzip.open(TMP / "pools_FRA.json.gz", "wt") as f:
        json.dump(out, f)
    print("saved", len(out), "builds")


def sample():
    rows = json.load(gzip.open(TMP / "pools_FRA.json.gz", "rt"))
    cells = defaultdict(list)
    for r in rows:
        g = sgroup(r["skill"])
        if g:
            r["cls"], r["group"] = cls(r["lane"], r["splash"]), g
            cells[(r["cls"], g)].append(r)
    out = []
    for k, v in sorted(cells.items()):
        v.sort(key=lambda r: hashlib.md5(f"{r['draft']}|{r['build']}".encode()).hexdigest())
        print(k, "available", len(v), flush=True)
        out += v[:CAPS[k[0]]]
    return out


ST = {}


def init():
    from tools import builder_check as bc
    bc.init("FRA", [])
    ST["bc"] = bc
    ST["yard"] = DeckModel("FRA", "deckmodel")
    ST["a"] = (deckbuilder.OBJECTIVE, deckbuilder.MODEL)


def rate(counts, main, splash):
    return float(ST["yard"].win_rate(counts, main, splash))


def one(r):
    bc = ST["bc"]
    A = bc.ARGS
    by_name = A["by_name"]
    cards = [by_name[n] for n, k in r["pool"].items() if n in by_name for _ in range(k)]
    their = rate(r["deck"], r["lane"], r["splash"])
    out = {k: r[k] for k in ("draft", "build", "skill", "lane", "splash", "cls", "group")}
    out["their"] = their
    for tag, (obj, mod) in (("A", ST["a"]), ("B", ("model", "deckmodel"))):
        deckbuilder.OBJECTIVE = obj
        model = A["model"] if tag == "A" else ST["yard"]
        deckbuilder.MODEL = mod
        try:
            builds = deckbuilder.build(cards, model, A["targets"], top=1, rule=A["rule"], three=A["three"])
        except Exception as e:  # keep going; counted in the report
            out[f"err{tag}"] = repr(e)[:80]
            continue
        if not builds:
            out[f"ours{tag}"] = None
            continue
        b = builds[0]
        counts = Counter(c["name"] for c in b["spells"])
        counts.update(Counter(c["name"] for c in b["nonbasics"]))
        counts.update(b["basics"])
        out[f"ours{tag}"] = rate(dict(counts), b["lane"], b["splash"])
        out[f"lane{tag}"], out[f"splash{tag}"] = b["lane"], b["splash"]
        theirs = {n: k for n, k in r["deck"].items() if n in by_name and "Land" not in (by_name[n].get("types") or [])}
        out[f"overlap{tag}"] = sum((Counter(c["name"].split(" // ")[0] for c in b["spells"]) & Counter(theirs)).values()) / max(1, sum(theirs.values()))
    deckbuilder.OBJECTIVE, deckbuilder.MODEL = ST["a"]
    return out


def draw():
    items = sample()
    with gzip.open(TMP / "pool_sample.json.gz", "wt") as f:
        json.dump(items, f)
    print("sampled", len(items))


def build():
    items = json.load(gzip.open(TMP / "pool_sample.json.gz", "rt"))
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None  # classes to build (the full sample took too long: 1spl ran first, then 2+spl)
    if only:
        items = [r for r in items if r["cls"] in only]
    print("pools to build", len(items), flush=True)
    done = 0
    with Pool(11, initializer=init) as pool, open(TMP / "pool_gaps.jsonl", "a") as f:
        for res in pool.imap_unordered(one, items, chunksize=4):
            f.write(json.dumps(res) + "\n")
            done += 1
            if done % 100 == 0:
                print("built", done, flush=True)
                f.flush()


def main():
    {"read": read, "draw": draw, "build": build}[sys.argv[1]]()


if __name__ == "__main__":
    main()
