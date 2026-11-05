"""Read real decks for synergy: Premier decks of one pair that all run one payoff card, trophies and 0-1 runs (one deck for the whole run), enabler counts spread.
For every payoff card in a deck, the enablers the deck holds for each of its payoff tags (tags: companion/advisor_v2/sets/FRA/roles.json).

    python3 analysis/deck_synergy_view.py BR "Grim Repriser" [n] [out_dir]   # out_dir: one ManaBox import file per deck
"""
import gzip, json, sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
ENABLER = {"ping_payoff": "ping_enabler", "spells_payoff": "spells", "gy_payoff": "mill", "jace_payoff": "jace_enabler",
           "death_payoff": "death_enabler", "lifegain_payoff": "lifegain_enabler", "etb_payoff": "etb_enabler",
           "counters_payoff": "counters_enabler", "ramp_payoff": "ramp", "draw_payoff": "draw"}
SHAPE = ("removal", "tempo", "delay", "draw", "defence")
lane, payoff, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5

front = lambda s: s.split(" // ")[0]
roles = {r["name"]: set(r["tags"]) for r in json.loads((R / "companion/advisor_v2/sets/FRA/roles.json").read_text())}
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)

def tags(name):
    c = cards.get(front(name), {})
    t = set(roles.get(name) or roles.get(front(name)) or ())
    if not {"Creature", "Land"} & set(c.get("types") or []): t.add("spells")
    return t

def spells(d):
    return {k: v for k, v in d["counts"].items() if "Land" not in (cards.get(front(k), {}).get("types") or [])}

raw = [d for d in json.loads(gzip.open(R / "analysis/out/decks_FRA.json.gz").read()) if d["format"] == "PremierDraft"]
rows_per_draft = Counter(d["draft"] for d in raw)  # more than one row: the deck changed during the run
decks = [d for d in raw if rows_per_draft[d["draft"]] == 1 and d["lane"] == lane and sum(d["counts"].values()) == 40 and payoff in d["counts"]]
main = next((t for t in sorted(tags(payoff)) if t in ENABLER), None)  # the payoff tag the decks are spread on
key = lambda d: sum(k for c, k in spells(d).items() if ENABLER.get(main) in tags(c))
# A Premier run ends at 7 wins or 3 losses: winners are trophies, losers finished runs at 0-1 wins
win = sorted((d for d in decks if d["trophy"]), key=key)
lose = sorted((d for d in decks if d["match_wins"] <= 1 and d["deck_games"] - d["deck_wins"] == 3), key=key)
nw = (n + 1) // 2
pick = lambda xs, k: [xs[round(i * (len(xs) - 1) / max(k - 1, 1))] for i in range(k)] if xs else []  # spread over enabler counts
chosen = [("WIN", d) for d in pick(win, nw)] + [("LOSE", d) for d in pick(lose, n - nw)]
print(f"{lane} decks with {payoff}: {len(decks)} (one deck all run; {len(win)} trophies, {len(lose)} finished at 0-1 wins); main payoff tag {main}\n")

out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
for i, (label, d) in enumerate(chosen, 1):
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = f"{int(d['deck_wins'])}-{int(d['deck_games'] - d['deck_wins'])}"
        (out_dir / f"{i}_{label}_{rec}.txt").write_text("".join(f"{k} {c}\n" for c, k in sorted(d["counts"].items())))
    sp = spells(d)
    gih = [((cards.get(front(c), {}).get("deck_colors", {}).get(lane) or {}).get("gihwr") or 0, k) for c, k in sp.items()]
    q = sum(g * k for g, k in gih) / sum(k for _, k in gih)
    print(f"=== {label} {int(d['deck_wins'])}-{int(d['deck_games'] - d['deck_wins'])}  skill {d['skill']:.2f}  mean GIH {q:.1f}  "
          f"splash {d['splash'] or '-'}  lands {40 - sum(sp.values())}")
    for c, k in sorted(sp.items(), key=lambda x: (cards.get(front(x[0]), {}).get("cmc") or 0, x[0])):
        print(f"  {k}x {c} ({cards.get(front(c), {}).get('cmc', '?')}): {', '.join(sorted(tags(c) - {'spells'})) or '-'}")
    print("  synergies:")
    for c in sp:
        for t in sorted(tags(c)):
            if t in ENABLER:
                en = [f"{k}x {e}" for e, k in sp.items() if e != c and ENABLER[t] in tags(e)]
                print(f"    {c} [{t}] <- {sum(int(x.split('x ')[0]) for x in en)} {ENABLER[t]}: {', '.join(en) or 'none'}")
    print("  shape: " + ", ".join(f"{t} {sum(k for c, k in sp.items() if t in tags(c))}" for t in SHAPE) + "\n")
