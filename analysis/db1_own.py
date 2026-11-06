import json
O = "/home/morris/Projects/mtga-draft-companion/data/Temp/sim/db1/"
for arm in ("base", "new"):
    r = json.load(open(O + f"own_{arm}.json")); n = len(r)
    sp = lambda x: len(x["our_splash"])
    th = lambda x: len(x["their_splash"] or "")
    print(arm, n)
    print("  ours  base-3colour %.1f%%  splash0 %.1f%% splash1 %.1f%% splash2+ %.1f%%" % tuple(100 * s / n for s in (
        sum(len(x["our_lane"]) == 3 for x in r), sum(sp(x) == 0 for x in r), sum(sp(x) == 1 for x in r), sum(sp(x) >= 2 for x in r))))
    two = [x for x in r if len(x["our_lane"]) == 2]
    print("  ours two-colour decks %d: splash2+ %.1f%%, splash1 %.1f%%" % (len(two), 100 * sum(sp(x) >= 2 for x in two) / len(two), 100 * sum(sp(x) == 1 for x in two) / len(two)))
    t2 = [x for x in r if len(x["lane"]) == 2]
    print("  theirs (n=%d): three-col base %.1f%%; of two-col (n=%d): splash0 %.1f%% 1 %.1f%% 2+ %.1f%%" % (n, 100 * sum(len(x["lane"]) == 3 for x in r) / n, len(t2),
        *[100 * sum(th(x) == k if k < 2 else th(x) >= 2 for x in t2) / len(t2) for k in (0, 1, 2)]))
    print("  same main pair %.1f%%  mean overlap %.1f%%  yardstick %.2f  secs/pool %.1f" % (100 * sum(x["lane"] == x["our_lane"] for x in r) / n, 100 * sum(x["overlap"] for x in r) / n,
        sum(x["yardstick"] for x in r) / n, sum(x["secs"] for x in r) / n))
    sh = [x for x in r if x["their_splash"] and x["our_splash"]]
    print("  they splash: %d; we splash same time %d; our splash overlaps theirs in colour %d" % (sum(bool(th(x)) for x in r), sum(bool(th(x)) and bool(sp(x)) for x in r), sum(bool(x["their_splash"]) and any(c in x["their_splash"] for c in x["our_splash"]) for x in r)))
    print("  of 2-col-lane pools, they 2+ splash %.1f%%" % (100 * sum(th(x) >= 2 for x in t2) / len(t2)))
