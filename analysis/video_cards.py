"""Descriptive check: do video drafters side with v2 or with top 17Lands players on flagged cards?
Reads analysis/videos/*/{picks,deck,ideas}.json and engine/latest.json (read-only). Writes analysis/out/video_cards.txt.
Engine records keep only v2's top 3 per pick: v2 rank is exact for #1, for ranks 2-3 and for the card he took (his_rank);
other cards are '>3'."""
import json, re, sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VID = ROOT / "videos"
OUT = ROOT / "out" / "video_cards.txt"

OVER = ["Murmuring Volume", "Stingerquill Annex", "Archive Arbiter", "Rescue Girl First Responder", "Konstrari Annex",
        "Twinned Vision", "Geist of Saint Thalia", "Ruric Thar Biomagus", "Keeper of the Quiet Hour",
        "Fatehold Chronologist", "Bestial Incursion", "Tetsuko Umezawa Fugitive"]
UNDER = ["Precise Redaction", "Extrapolate the Impossible", "Kiora of Salt and Sand", "Edgar Ancient Bloodlord",
         "Mabel Valley Hero", "Koth the Geomancer", "Transformative Commons", "Warrior's Blades", "Stingerquill Annex",
         "Restore with Empathy", "Winter Team Player", "Cruel Calculations"]


def n(s):
    s = s or ""
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace(",", "").replace("'", "")).strip()


def cp(k, m, a=0.05):
    if m == 0:
        return (float("nan"),) * 2
    def cdf(x, p):  # P(X<=x)
        return sum(comb(m, i) * p**i * (1 - p) ** (m - i) for i in range(x + 1))
    def bis(f, lo, hi):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else bis(lambda p: 1 - cdf(k - 1, p) < a / 2, 0, 1)
    hi = 1.0 if k == m else bis(lambda p: cdf(k, p) > a / 2, 0, 1)
    return lo, hi


def share(k, m):
    if m == 0:
        return "0/0"
    lo, hi = cp(k, m)
    return f"{k}/{m} = {100*k/m:.0f}% (CI {100*lo:.0f}-{100*hi:.0f}%)"


def mmss(t):
    t = int(t)
    return f"{t//60}:{t%60:02d}"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


vids = json.loads((VID / "index.json").read_text())
lists = {"OVERRATED by v2": OVER, "UNDERRATED by v2": UNDER}
rows = []  # one per (video, pick, flagged card offered)
decks = {}
ideas = []
nick = {v["id"]: f'{v["player"]}' for v in vids}
npicks = nscored = 0
for v in vids:
    i = v["id"]
    pk, dk, idd, en = (load(VID / i / f) for f in ("picks.json", "deck.json", "ideas.json", "engine/latest.json"))
    if not pk or not en:
        continue
    epk = {(e["pack"], e["pick"]): e for e in en["picks"]}
    for p in pk["picks"]:
        npicks += 1
        e = epk.get((p["pack"], p["pick"]))
        if e and e.get("scored"):
            nscored += 1
        for card in p["pack_cards"]:
            for lname, lst in lists.items():
                if n(card) in map(n, lst):
                    took = n(card) == n(p["taken"])
                    v2_1 = v2r = None
                    if e and e.get("scored"):
                        t3 = [n(x["name"]) for x in e["top3"]]
                        v2_1 = t3[0] == n(card)
                        if n(card) in t3:
                            v2r = t3.index(n(card)) + 1
                        elif took:
                            v2r = e["his_rank"]
                        else:
                            v2r = ">3"
                        v2top = e["top3"][0]["name"]
                    else:
                        v2top = None
                    rows.append(dict(vid=i, pack=p["pack"], pick=p["pick"], card=card, lst=lname, took=took, v2_1=v2_1,
                                     v2r=v2r, v2top=v2top, taken=p["taken"], reason=p.get("reasoning", ""),
                                     considered=(p.get("considered") or {}).get(card), time=p.get("time")))
    if dk:
        decks[i] = {n(c): k for c, k in {**dk["main"], **dk.get("lands", {})}.items()}
        decks[i + "_side"] = {n(c): k for c, k in dk.get("sideboard", {}).items()}
    if idd:
        for idea in idd["ideas"]:
            ideas.append((i, idea))

out = []
w = out.append
w(f"Videos: {len(vids)} recorded; picks read: {npicks}; scored by v2 (engine/latest.json): {nscored}.")
w("Players: " + ", ".join(sorted(set(nick.values()))) + ". v2 engine record keeps its top 3 per pick only.")
w("Decks: " + str(len([k for k in decks if not k.endswith('_side')])) + " deck.json files (pdITCHUnEa8 deck cut off at 22:40).")
w("")
for lname, lst in lists.items():
    w("=" * 100)
    w(lname)
    w("=" * 100)
    R = [r for r in rows if r["lst"] == lname]
    w(f"{'card':32s}{'offered':>8s}{'took':>6s}{'v2 #1':>7s}{'both':>6s}{'v2#1 not taken':>16s}{'taken, v2#1 other':>19s}{'in final 40':>13s}{'decks':>7s}")
    tot = dict(off=0, took=0, v21=0, both=0, v21nt=0, tko=0, in40=0, dk=0, offsc=0)
    for c in lst:
        rc = [r for r in R if n(r["card"]) == n(c)]
        sc = [r for r in rc if r["v2_1"] is not None]
        off, took = len(rc), sum(r["took"] for r in rc)
        v21 = sum(r["v2_1"] for r in sc)
        both = sum(r["v2_1"] and r["took"] for r in sc)
        tko = sum((not r["v2_1"]) and r["took"] for r in sc)
        in40 = sum(1 for k, d in decks.items() if not k.endswith("_side") and n(c) in d)
        # drafts where he took it at least once and it is in final 40
        w(f"{c:32s}{off:8d}{took:6d}{v21:7d}{both:6d}{v21-both:16d}{tko:19d}{in40:13d}{len(decks)//2:7d}")
        tot["off"] += off; tot["took"] += took; tot["v21"] += v21; tot["both"] += both
        tot["v21nt"] += v21 - both; tot["tko"] += tko; tot["in40"] += in40; tot["offsc"] += len(sc)
    w(f"{'TOTAL offers':32s}{tot['off']:8d}{tot['took']:6d}{tot['v21']:7d}{tot['both']:6d}{tot['v21nt']:16d}{tot['tko']:19d}{tot['in40']:13d}")
    w(f"(offers scored by v2: {tot['offsc']}; 'in final 40' counts draft-deck pairs where the card is in the main, any copy)")
    w("")
    sc = [r for r in R if r["v2_1"] is not None]
    if lname.startswith("OVER"):
        k = tot["both"]; m = tot["v21"]
        w(f"When v2 had the card #1 and it was in his pack: he took it {share(k, m)}   [17Lands players: 30-45%]")
        w(f"Offers taken (any v2 rank): {share(tot['took'], tot['off'])}")
        w(f"Offers where v2 had it #1 but he passed: {tot['v21nt']}; of those, what he took instead: ")
        for r in sc:
            if r["v2_1"] and not r["took"]:
                w(f"   {r['vid']} P{r['pack']}p{r['pick']}: {r['card']} (v2 #1) -> he took {r['taken']}")
    else:
        w(f"Offers taken: {share(tot['took'], tot['off'])}")
        w(f"Taken while v2's #1 was a different card: {share(tot['tko'], tot['took'])} of takes (scored takes only in numerator)")
        w(f"Taken and v2 agreed (#1): {share(tot['both'], tot['took'])}")
        for r in sc:
            if r["took"]:
                w(f"   {r['vid']} P{r['pack']}p{r['pick']}: took {r['card']}, v2 rank {r['v2r']}, v2 #1 = {r['v2top']}")
    w("")
    w("Per-offer rows (video pack/pick, card, taken?, v2 rank, v2 #1):")
    for r in R:
        w(f"   {r['vid']} P{r['pack']}p{r['pick']} {r['card']:30s} took={'Y' if r['took'] else 'n'} v2rank={r['v2r']} v2#1={r['v2top']}  (he took: {r['taken']})")
    w("")

# final deck sections
w("=" * 100)
w("Flagged cards in final 40 / sideboard, per draft")
w("=" * 100)
for i in [k for k in decks if not k.endswith("_side")]:
    for lname, lst in lists.items():
        m = [f"{c}{'' if decks[i][n(c)]==1 else ' x'+str(decks[i][n(c)])}" for c in lst if n(c) in decks[i]]
        s = [c for c in lst if n(c) in decks[i + "_side"]]
        w(f"{i} {lname[:5]}: main {m or '-'}; side {s or '-'}")
w("")

# reasoning
w("=" * 100)
w("His stated reasons mentioning flagged cards (pick reasoning and ideas.json quotes)")
w("=" * 100)
allc = {n(c): c for c in OVER + UNDER}
def tag(c):
    t = [x for x, l in (("OVER", OVER), ("UNDER", UNDER)) if n(c) in map(n, l)]
    return "/".join(t)
for r in rows:
    if r["considered"] or (r["took"] and r["reason"]):
        why = r["considered"] if r["considered"] else r["reason"]
        w(f"[{r['vid']} {r['time']}] P{r['pack']}p{r['pick']} {r['card']} ({tag(r['card'])}, {'took' if r['took'] else 'passed, took '+r['taken']}): {why[:300]}")
seen = set()
for r in rows:  # mentions in the reasoning of other picks (pass reasons)
    pass
w("")
w("-- ideas.json (time = seconds into video) --")
for i, idea in ideas:
    hit = [c for c in idea.get("cards", []) if n(c) in allc]
    if hit:
        w(f"[{i} {mmss(idea['time'])}] {', '.join(f'{c} ({tag(c)})' for c in hit)}: \"{idea.get('quote','')[:260]}\"")
w("")
w("Sample sizes: 11 drafts x 42 picks = %d picks, from 2 players (NicolaiBolas x6, Alex Nikolic x5, one with a guest); offers include wheel re-offers of the same pack card, so they are not independent; " % npicks +
  "each card is offered in the few packs listed above. Counts and exact CIs only; no test.")

OUT.write_text("\n".join(out) + "\n")
print("\n".join(out))
