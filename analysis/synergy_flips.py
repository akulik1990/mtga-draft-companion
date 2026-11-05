"""Where the strategy-synergy term (engine STRAT_SYN) changes the drafter's top pick, and whether the change moves it onto the card a
top drafter took (helped) or off it (hurt). Exploratory (Akshin, 2026-11-05: find what decides when synergy matters); any rule it
suggests is tested on the train split before it goes near the engine.

Input: replay logs written by a wrapper around tools/replay/run.py (one entry per pick: draft id, pack, pick, every candidate's
score, pool names), one log with the term off and one on, over the same drafts.

    python3 analysis/synergy_flips.py <log_off.json> <log_on.json> <drafts.json.gz> [out.csv]
"""
import gzip, json, sys
from pathlib import Path
from statistics import mean, stdev

R = Path(__file__).resolve().parents[1]
off, on = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
human = {(d["id"], p[0], p[1]): p[2] for d in json.load(gzip.open(sys.argv[3], "rt")) for p in d["picks"]}
WEIGHTS = json.loads((R / "companion/advisor_v2/sets/FRA/fitted.json").read_text())["strategy"]  # lane -> {enabler tag, payoff tag}
PAIR_TAGS = {lane: list(w) for lane, w in WEIGHTS.items()}  # [enabler, payoff], in the order the fit wrote them
front = lambda n: n.split(" // ")[0]
roles = {r["name"]: set(r["tags"]) for r in json.loads((R / "companion/advisor_v2/sets/FRA/roles.json").read_text())}
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)

def info(n): return cards.get(front(n), {})
def is_land(n): return "Land" in (info(n).get("types") or [])
def tags(n):
    t = set(roles.get(front(n), ()))
    if not {"Creature", "Land"} & set(info(n).get("types") or []): t.add("spells")
    return t
def gih(n): return (info(n).get("deck_colors", {}).get("All Decks") or {}).get("gihwr") or 0.0
def colours(n): return set(info(n).get("colors") or [])

def main_pair(pool):
    count = {c: 0 for c in "WUBRG"}
    for n in pool:
        if not is_land(n):
            for c in colours(n): count[c] += 1
    top = sorted(count, key=lambda c: -count[c])[:2]
    lane = "".join(c for c in "WUBRG" if c in top)
    coloured = sum(count.values())
    return lane, (count[top[0]] + count[top[1]]) / coloured if coloured else 0.0

def partners(n, lane, pool):
    """Pool cards that pair with n in the lane's strategy: payoffs for an enabler, enablers for a payoff."""
    if lane not in PAIR_TAGS: return 0
    e, p = PAIR_TAGS[lane]
    t, out = tags(n), 0
    if e in t: out += sum(1 for m in pool if p in tags(m))
    if p in t: out += sum(1 for m in pool if e in tags(m))
    return out

rows = []
for a, b in zip(off, on):
    assert a[:3] == b[:3]
    if not a[3] or not b[3] or a[3][0][0] == b[3][0][0]: continue
    key, pool = tuple(a[:3]), a[4]
    h, A, B = human[key], a[3][0][0], b[3][0][0]  # A: top with synergy off, B: top with it on
    so, sn = dict(a[3]), dict(b[3])
    lane, commit = main_pair(pool)
    in_lane = lambda n: colours(n) <= set(lane)
    strat = lambda n: sum(1 for t in tags(n) if t in PAIR_TAGS.get(lane, ()))
    nstrat = lambda n: sum(1 for l, ts in PAIR_TAGS.items() for t in ts if t in tags(n))
    rows.append(dict(
        draft=key[0], pack=key[1], pick=key[2], outcome="helped" if B == h else "hurt" if A == h else "neither",
        A=A, B=B, human=h, lane=lane, commit=round(commit, 2), pool=sum(1 for n in pool if not is_land(n)),
        gap_off=round(so[A] - so.get(B, 0), 3), syn_shift=round((sn[B] - so.get(B, 0)) - (sn.get(A, 0) - so[A]), 3),
        gih_A=gih(A), gih_B=gih(B), gih_diff=round(gih(B) - gih(A), 2),
        B_in_lane=int(in_lane(B)), A_in_lane=int(in_lane(A)), B_strat=strat(B), A_strat=strat(A),
        B_partners=partners(B, lane, pool), A_partners=partners(A, lane, pool), B_lanes_tagged=nstrat(B), A_lanes_tagged=nstrat(A)))

if len(sys.argv) > 4:
    import csv
    with open(sys.argv[4], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def describe(name, rs):
    print(f"\n{name}: {len(rs)} flips")
    for k in ("pack", "pick", "pool", "commit", "gap_off", "syn_shift", "gih_A", "gih_B", "gih_diff", "B_in_lane", "A_in_lane",
              "B_strat", "A_strat", "B_partners", "A_partners", "B_lanes_tagged", "A_lanes_tagged"):
        xs = [r[k] for r in rs]
        print(f"  {k:15s} {mean(xs):7.2f} (SE {stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0:.2f})")

by = {o: [r for r in rows if r["outcome"] == o] for o in ("helped", "hurt", "neither")}
print(f"flips {len(rows)}: helped {len(by['helped'])}, hurt {len(by['hurt'])}, neither {len(by['neither'])}")
for o in ("helped", "hurt"): describe(o, by[o])

def split(label, f):
    for v in sorted({f(r) for r in rows if r["outcome"] != "neither"}):
        hp = sum(1 for r in by["helped"] if f(r) == v); ht = sum(1 for r in by["hurt"] if f(r) == v)
        print(f"  {label}={v}: helped {hp}, hurt {ht}, share helped {hp / (hp + ht):.2f}" if hp + ht else "")
print("\nShare helped by condition")
split("pack", lambda r: r["pack"])
split("promoted card in pool's main pair", lambda r: r["B_in_lane"])
split("promoted card has the main pair's strategy tag", lambda r: int(r["B_strat"] > 0))
split("promoted card has pool partners", lambda r: min(r["B_partners"], 3))
split("promoted card tagged for N pair strategies", lambda r: min(r["B_lanes_tagged"], 3))
split("pool commitment >= 0.8", lambda r: int(r["commit"] >= 0.8))
