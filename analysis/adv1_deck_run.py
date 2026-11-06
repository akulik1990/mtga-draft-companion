"""adv1 final-deck comparison (Akshin, 2026-11-06). PRE-REGISTERED before any outcome was read:
  PRIMARY: mean over the 100 sp3 drafts of [yardstick WR of the default build of v2's own pool] minus [yardstick WR of the default
           build of the player's pool], same builder (session.own_builds, builds[0]) and same yardstick (session.deck_win_rate on a
           build dict with mana_identity), paired by draft, 95% CI bootstrap over drafts (B=4000, seed 1).
  SECONDARY (a) share of drafts where v2's deck >= the player's built deck; (b) v2 built deck minus the player's REGISTERED 40, same yardstick;
           (c) share of built decks that splash / go three-colour (builder label lane length 3, and mana_identity splash non-empty), v2 vs player's
           registered deck (decks_FRA lane/splash) vs player-pool build. Splash and three-colour drafts also reported separately.
  v2's pool = v2's top-scored card at every step from the PLAYER's packs (approximation: packs are the player's, so v2 never changes what it
  is passed; a card v2 leaves is the one the player may have taken). Seen packs = the player's.
    env -C <repo> uv run python analysis/adv1_deck_run.py <out.json> [n_drafts]"""
import gzip, json, sys, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
out = Path(sys.argv[1]).resolve(); N = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
R = Path(__file__).resolve().parents[1]
from tools.replay import run  # noqa: E402  (sets cwd to the data dir)
from companion import session  # noqa: E402
from companion.advisor_v2 import config, deckbuilder  # noqa: E402
front = lambda n: n.split(" // ")[0]
meta = json.loads((R / "data/Temp/adv1/meta.json").read_text())
drafts = json.load(gzip.open(R / "data/Temp/adv1/sp3.drafts.json.gz", "rt"))[:N]

def init():
    global app, cls
    app = run.replay.App(str(run.dataset_path("FRA")))
    run.use_advisor("v2"); cls = run.advisor_class("v2"); config.activate("FRA")

def grouped(cards):
    d = {}
    for c in cards:
        e = d.setdefault(c["name"], {**session._trim(c), "count": 0}); e["count"] += 1
    return list(d.values())

def score(deck):
    """(yardstick WR, main, splash) for a list of card entries with counts."""
    mi = session.mana_identity(deck)
    return session.deck_win_rate("FRA", {"deck": deck, **mi}), mi

def build_info(pool):
    bs = session.own_builds("FRA", pool)
    if not bs: return None
    b = bs[0]; wr, mi = score(b["deck"])
    return dict(label=b["label"], variant=b["variant"], lane=b["label"].split(" ")[0], model_wr=b.get("model_wr"), wr=wr, main=mi["main"],
                splash=mi["splash"], deck={c["name"]: c["count"] for c in b["deck"]})

def job(d):
    picks = d["picks"]
    mypool, plpool, mypicks = [], [], []
    WHEEL = bool(int(__import__("os").environ.get("ADV1_WHEEL", "0")))  # drop copies of a card v2 already took from this physical pack 8 picks ago
    packs = {(p[0], p[1]): __import__("collections").Counter(p[3]) for p in picks}
    for pack_no, pick_no, taken, pack_names, _pool in picks:
        pack = app.cards(pack_names); got = app.cards([taken])
        if got: plpool.append(got[0])
        seen = [(p, k, app.cards(n)) for p, k, _t, n, _ in picks if (p, k) <= (pack_no, pick_no)]
        pick = None
        if WHEEL and pick_no > 8 and mypicks[(pack_no - 1) * 14 + pick_no - 9] is not None:
            first, used = packs[(pack_no, pick_no - 8)], mypicks[(pack_no - 1) * 14 + pick_no - 9]
            cap = min(packs[(pack_no, pick_no)][used], first[used] - 1)  # copies of `used` left in the pack v2 saw
            seen_n = 0; keep = []
            for c in pack:
                if c["name"] == used:
                    seen_n += 1
                    if seen_n > cap: continue
                keep.append(c)
            pack = keep
        if pack:
            recs = cls(app.metrics, mypool, seen_packs=seen).evaluate_pack(pack, pick_no, current_pack=pack_no)
            sc = {}
            for r in recs: sc.setdefault(r.card_name, r.contextual_score)
            if sc:
                pick = max(sc, key=sc.get); mypool.append(next(c for c in pack if c["name"] == pick))
        if pick is None and got: mypool.append(got[0]); pick = got[0]["name"]
        mypicks.append(pick)
    m = meta[d["id"]]
    reg = []
    for n, k in m["counts"].items():
        c = next(iter(app.cards([n])), None)
        if c: reg.append({**session._trim(c), "count": k})
        elif n in deckbuilder.BASIC.values():
            col = next(x for x, y in deckbuilder.BASIC.items() if y == n)
            reg.append({"name": n, "count": k, "cmc": 0, "mana_cost": "", "types": ["Basic", "Land"], "colors": [col]})
    rwr, rmi = score(reg)
    overlap = len({front(c["name"]) for c in mypool} & {front(c["name"]) for c in plpool})
    return dict(id=d["id"], kind=m["kind"], reg_lane=m["lane"], reg_splash=m["splash"], reg_wr=rwr, reg_main=rmi["main"], reg_ms=rmi["splash"],
                reg_cards=sum(c["count"] for c in reg), v2=build_info(mypool), pl=build_info(plpool), v2_picks=mypicks, pool_overlap=overlap, n_picks=len(plpool))

if __name__ == "__main__":
    with mp.Pool(4, initializer=init) as p:
        res = []
        for i, r in enumerate(p.imap_unordered(job, drafts), 1):
            res.append(r); print(i, r["id"][:8], r["kind"], r["v2"] and r["v2"]["wr"], r["pl"] and r["pl"]["wr"], r["reg_wr"], flush=True)
    out.write_text(json.dumps(res)); print("done", len(res))
