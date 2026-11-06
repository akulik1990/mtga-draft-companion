"""Part 1(b): the builder's limits relaxed one at a time and together, in memory only (module constants set, then restored).
Scores with the mock's yardstick (session.deck_win_rate, "deckmodel"). Second half: a model-only search of his pool
(no builder objective) for every splash-colour set S of {B,R,G} on his WU base, to price the second and third splash colour.
Output: analysis/out/builder_gap_relax.txt"""

import itertools
from collections import Counter

from builder_gap_common import *  # noqa: F401,F403
from companion.advisor_v2.lanes import castable, off_pips

setup()
pool = pool_cards()
his = his_deck(pool)
yard = DeckModel("FRA", "deckmodel")
names_in = lambda b: {c["name"] for c in b["deck"] if "Basic" not in (c.get("types") or [])}
his_names = names_in(his)


def run(label, **over):
    saved = {k: getattr(deckbuilder, k) for k in over}
    try:
        for k, v in over.items():
            setattr(deckbuilder, k, v)
        deckbuilder._structure.cache_clear()
        builds = session.own_builds("FRA", pool) or []
    finally:
        for k, v in saved.items():
            setattr(deckbuilder, k, v)
    if not builds:
        print(f"{label:58s} no build")
        return None
    b = builds[0]
    wr = session.deck_win_rate("FRA", b)
    n = names_in(b)
    print(f"{label:58s} {b['label']:12s} yardstick {wr:6.2f}  spells shared with his {len(n & his_names):2d}  "
          f"in ours not his: {', '.join(sorted(n - his_names)) or '-'}  | in his not ours: {', '.join(sorted(his_names - n)) or '-'}", flush=True)
    return wr


print(f"his 40 yardstick {session.deck_win_rate('FRA', his)}")
run("baseline (as configured)")
run("OBJECTIVE=model (builder maximises the deck model)", OBJECTIVE="model")
run("OBJECTIVE=model, MODEL=deckmodel (the yardstick)", OBJECTIVE="model", MODEL="deckmodel")
run("MAX_SPLASH 2 -> 5", MAX_SPLASH=5)
run("MAX_SPLASH_BASICS 2 -> 4", MAX_SPLASH_BASICS=4)
run("SPLASH_MIN_GIH 56.5 -> 0", SPLASH_MIN_GIH=0.0)
run("DP_BOMB_SPLASH off", DP_BOMB_SPLASH=False)
run("SPLASH_LANDS_ONLY off", SPLASH_LANDS_ONLY=False)
run("SPLASH_RULE_LANDS_ONLY off", SPLASH_RULE_LANDS_ONLY=False)
run("shape penalty off (PENALTY 0)", PENALTY=0.0)
run("SPLASH_FORCE off", SPLASH_FORCE=False)
run("splash rule ignored (SPLASH_AT 0, THREE_AT 9)", SPLASH_AT=0.0, THREE_AT=9.0)
run("three-colour always (THREE_AT 0)", THREE_AT=0.0)
splash_all = dict(MAX_SPLASH=5, MAX_SPLASH_BASICS=4, SPLASH_MIN_GIH=0.0, DP_BOMB_SPLASH=False, SPLASH_LANDS_ONLY=False, SPLASH_RULE_LANDS_ONLY=False)
run("all splash limits relaxed together", **splash_all)
run("all splash limits + shape penalty off", PENALTY=0.0, **splash_all)
run("all splash limits + OBJECTIVE=model/yardstick", OBJECTIVE="model", MODEL="deckmodel", **splash_all)
run("all of the above + PENALTY 0", OBJECTIVE="model", MODEL="deckmodel", PENALTY=0.0, **splash_all)
run("same, three always", OBJECTIVE="model", MODEL="deckmodel", PENALTY=0.0, THREE_AT=0.0, **splash_all)

# ---- model-only search of his pool, WU base, splash colours S -----------------------------------------------------------
print("\n=== yardstick search of his pool, base WU, splash colour set S (no builder objective; builder shape penalty on/off) ===")
lane = "WU"
targets = config.fitted("FRA", "decks")["pairs"]
spells = [c for c in pool if "Land" not in (c.get("types") or [])]
lands_pool = [c for c in pool if "Land" in (c.get("types") or []) and "Basic" not in (c.get("types") or [])]


def allowed(c, S):
    if castable(c, lane):
        return True
    off = off_pips(c, lane)
    return len(off) == 1 and bool(set(off[0]) & set(S))


def mana(chosen, S):
    nb = [c for c in lands_pool if ((c.get("colors") or []) and all(x in lane + S for x in c["colors"]) and any(x in lane for x in c["colors"]))
          or (not c.get("colors") and any(s in (c.get("oracle_text") or "") for s in ("chosen color", "any color")))]
    used = "".join(s for s in S if any(not castable(c, lane) and s in off_pips(c, lane)[0] for c in chosen))
    basics = deckbuilder.lands_for(chosen, lane, nb, "", 0, 17)
    extra = {}
    for s in used:
        have = sum(1 for c in nb if s in (c.get("colors") or []) or not c.get("colors"))
        extra[deckbuilder.BASIC[s]] = min(deckbuilder.MAX_SPLASH_BASICS, max(1, 3 - have))
    tot = sum(basics.values()) + sum(extra.values())
    big = max(basics, key=basics.get)
    basics[big] -= max(0, tot - (17 - len(nb)))
    return nb, {**basics, **extra}, used


def score(idx, S, pen):
    chosen = [spells[i] for i in idx]
    nb, basics, used = mana(chosen, S)
    counts = Counter(c["name"] for c in chosen)
    counts.update(basics)
    counts.update(Counter(c["name"] for c in nb))
    wr = float(yard.win_rate(dict(counts), lane, used))
    p = deckbuilder.shape_penalty(chosen, targets[lane], lane) if pen else 0.0
    return wr - p, wr, p, used, nb, basics


def search(S, pen, max_splash_cards):
    cand = [i for i, c in enumerate(spells) if allowed(c, S)]
    n_splash = lambda idx: sum(1 for i in idx if not castable(spells[i], lane))
    ok = lambda idx: n_splash(idx) <= max_splash_cards
    start = set(sorted(cand, key=lambda i: (not castable(spells[i], lane), -yard.m["cards"].get(spells[i]["name"], 0)))[:23])
    if len(start) < 23:
        return None
    best = score(start, S, pen)[0]
    deck, side = set(start), set(cand) - start
    improved = True
    while improved:
        improved = False
        for i in sorted(deck):
            for j in sorted(side):
                t = (deck - {i}) | {j}
                if not ok(t):
                    continue
                s = score(t, S, pen)[0]
                if s > best + 1e-9:
                    deck, side, best, improved = t, (side - {j}) | {i}, s, True
                    break
            if improved:
                break
    return deck, score(deck, S, pen)


for pen in (True, False):
    for cap in (2, 3, 9):
        print(f"\n-- shape penalty {'on' if pen else 'off'}, at most {cap} splash cards")
        for k in range(4):
            for S in itertools.combinations("BRG", k):
                S = "".join(S)
                r = search(S, pen, cap)
                if not r:
                    continue
                deck, (obj, wr, p, used, nb, basics) = r
                off = sorted(spells[i]["name"] for i in deck if not castable(spells[i], lane))
                print(f"  S={S or '-':4s} used={used or '-':4s} yardstick {wr:6.2f} penalty {p:4.1f}  splash cards {len(off)}: {', '.join(off)}")
