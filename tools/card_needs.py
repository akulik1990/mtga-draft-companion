"""Do the set's top cards need what we think they need? Tests manual.json card_wants against 17Lands game data.

    uv run python tools/card_needs.py FRA       # prints each card's verdict, writes card_needs to the set's fitted.json

For each card in manual.json "card_wants": its enablers are matched across the set by rules text (reminder text
stripped), tags or types. Then, over every game whose deck held the card: a logistic model of the game result on the
deck's enabler count, whether the card was drawn (opening hand or drawn), their product, the deck's mean GIH WR, the
player's skill bucket and play/draw. The product is the test: how much more each enabler is worth when the card is in
hand than when it is not. Enablers that are simply good cards help either way and cancel out.
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402

P_TYPICAL = 0.55  # log-odds to WR points at a typical win rate: x p(1-p) x 100


@lru_cache(maxsize=1)
def scryfall(set_code="FRA"):
    """{name: {power, toughness}} cached from Scryfall (the dataset has no power)."""
    path = DATA_DIR / "Temp" / f"scryfall_{set_code}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def power(card):
    p = (scryfall().get(card.get("name")) or {}).get("power")
    return int(p) if p and p.isdigit() else None


def is_enabler(card, spec):
    text = re.sub(r"\([^)]*\)", "", (card.get("oracle_text") or "").lower())
    types, tags = card.get("types") or [], card.get("tags") or []
    by_text = bool(spec.get("enabler") and re.search(spec["enabler"], text))
    by_type = (bool(spec.get("enabler_types")) and any(t in types for t in spec["enabler_types"])
               and spec.get("enabler_min_cmc", 0) <= (card.get("cmc") or 0) <= spec.get("enabler_max_cmc", 99))
    if spec.get("min_power") and "Creature" in types and (power(card) or 0) >= spec["min_power"]:
        return True
    if spec.get("names"):
        return card.get("name") in spec["names"]
    if spec.get("cheap"):
        return "Land" not in types and (card.get("cmc") or 0) <= spec["cheap"]
    by_tag = any(t in tags for t in spec.get("enabler_tags", ()))
    if spec.get("multi_type"):
        return sum(t in types for t in ("Artifact", "Creature", "Enchantment", "Planeswalker", "Instant", "Sorcery", "Battle")) >= 2
    if spec.get("all_types_tags"):
        return any(t in types for t in spec["enabler_types"]) and by_tag
    if spec.get("all"):
        return by_text and by_type
    return by_text or by_type or by_tag


def prefer(card, spec):
    """The enablers a card wants most (manual "prefer" pattern): listed first among the examples."""
    text = re.sub(r"\([^)]*\)", "", (card.get("oracle_text") or "").lower())
    return bool(spec.get("prefer") and re.search(spec["prefer"], text))


def logit(X, y):
    """(coefficients, SEs) by Newton's method."""
    w = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    a = p.parse_args(argv)

    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    cards = {}
    for c in data.values():
        cards.setdefault(c["name"], c)
    wants = {}
    for k, v in (config.layer(a.set, "manual").get("card_wants") or {}).items():
        if k.startswith("_"):
            continue
        wants[k] = v
        for x in v.get("extra") or []:  # further enabler lists, each tested on its own as "card: name"
            wants[f"{k}: {x['name']}"] = {**x, "card": k}
    gih = {n: ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0 for n, c in cards.items()}
    enablers = {x: sorted(n for n, c in cards.items() if n != spec.get("card", x) and "Basic" not in (c.get("types") or []) and is_enabler(c, spec))
                for x, spec in wants.items()}
    rows = defaultdict(list)  # card -> [enablers, in hand, deck quality, skill, on play, won]
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{a.set}.PremierDraft.csv.gz"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[len("deck_"):]) for i, h in enumerate(header) if h.startswith("deck_")]
        base = lambda x: wants[x].get("card", x)
        watch = [(x, col.get(f"deck_{base(x)}"), col.get(f"opening_hand_{base(x)}"), col.get(f"drawn_{base(x)}"), set(enablers[x])) for x in wants]
        watch = [w for w in watch if w[1] is not None]
        for row in reader:
            held = [w for w in watch if row[w[1]] not in ("", "0")]
            if not held:
                continue
            deck = {name: int(float(row[i])) for i, name in deck_cols if row[i] and row[i] != "0"}
            spells = [(n, k) for n, k in deck.items() if gih.get(n, 0) > 0]
            quality = sum(gih[n] * k for n, k in spells) / max(1, sum(k for _, k in spells))
            sk = row[col["user_game_win_rate_bucket"]]
            skill = float(sk) if sk not in ("", "None") else 0.55
            for x, dc, oc, rc, en in held:
                in_hand = float((row[oc] not in ("", "0")) or (row[rc] not in ("", "0")))
                rows[x].append([sum(k for n, k in deck.items() if n in en), in_hand, quality, skill,
                                float(row[col["on_play"]] == "True"), float(row[col["won"]] == "True")])
    out = {}
    scale = P_TYPICAL * (1 - P_TYPICAL) * 100
    for x in wants:
        R = np.array(rows.get(x) or [], float)
        if len(R) < 200:
            print(f"{x}: {len(R)} games, too few")
            continue
        n_en, hand, q, sk, play, won = R.T
        X = np.column_stack([np.ones(len(R)), n_en, hand, n_en * hand, q - q.mean(), sk - sk.mean(), play])
        w, se = logit(X, won)
        held = hand == 1
        lo, hi = np.percentile(n_en[held], [33, 67]) if held.any() else (0, 0)
        wr = lambda m: round(float(won[m].mean() * 100), 1) if m.any() else None
        out[x] = {"games": int(len(R)), "in_hand": int(held.sum()), "enablers": len(enablers[x]), "enabler_names": enablers[x],
                  "mean_enablers": round(float(n_en.mean()), 1),
                  "per_enabler_in_hand": round(float(w[3] * scale), 2), "se": round(float(se[3] * scale), 2), "z": round(float(w[3] / se[3]), 1),
                  "in_hand_wr_few": wr(held & (n_en <= lo)), "in_hand_wr_many": wr(held & (n_en >= hi)),
                  "few_max": float(lo), "many_min": float(hi)}
        r = out[x]
        print(f"{x:30s} {r['enablers']:3d} enablers (deck mean {r['mean_enablers']:4.1f}), {r['in_hand']:6d} games in hand: "
              f"{r['per_enabler_in_hand']:+5.2f} WR pts per enabler when drawn (SE {r['se']:.2f}, z {r['z']:+.1f}); "
              f"in-hand WR {r['in_hand_wr_few']} with <= {lo:g}, {r['in_hand_wr_many']} with >= {hi:g}")
    config.write_fitted(a.set, "card_needs", out, "tools/card_needs.py: manual.json card_wants tested on 17Lands game data (enablers x card in hand, controlling for deck quality, skill, play/draw)")
    print(f"wrote {report(a.set)}")



def report(set_code, top=30):
    """docs/top-cards-<SET>.html (print to PDF) for review: the set's top cards by GIH WR, their rules text, what we read
    them as needing (manual.json card_wants) and every enabler the rule matches."""
    import html
    import math

    from companion.advisor_v2.needs import deckmates

    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    cards = {}
    for c in data.values():
        if ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr"):
            cards.setdefault(c["name"], c)
    ranked = sorted(cards.values(), key=lambda c: -c["deck_colors"]["All Decks"]["gihwr"])[:top]
    manual = config.layer(set_code, "manual").get("card_wants") or {}
    cards_all = {}
    for c in data.values():
        cards_all.setdefault(c["name"], c)
    standalone = set(manual.get("_standalone") or [])
    e = html.escape
    out = []
    for i, c in enumerate(ranked, 1):
        a = c["deck_colors"]["All Decks"]
        name = c["name"]
        if name in standalone:
            read = "<b>Standalone.</b> Strong on its own; nothing in the deck needed for it to shine."
            verdict = ""
        elif name in manual:
            spec = manual[name]
            wr = lambda o: ((o.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0
            names = sorted((n for n, o in cards_all.items() if n != name and "Basic" not in (o.get("types") or []) and is_enabler(o, spec)),
                           key=lambda n: (not prefer(cards_all[n], spec), -wr(cards_all[n])))
            read = f"<b>Reading:</b> wants {e(spec['wants'])}."
            verdict = (f"<div class='en'><b>Enabler rule:</b> {e(spec.get('rule', ''))}. "
                       f"<b>Examples</b> (strongest of {len(names)}): {', '.join(e(n) for n in names[:6])}.</div>")
            for x in spec.get("extra") or []:
                more = sorted((n for n, o in cards_all.items() if n != name and "Basic" not in (o.get("types") or []) and is_enabler(o, x)),
                              key=lambda n: -wr(cards_all[n]))
                verdict += (f"<div class='en'><b>Also, {e(x['name'])}:</b> {e(x['rule'])}. "
                            f"<b>Examples</b> (strongest of {len(more)}): {', '.join(e(n) for n in more[:6])}.</div>")
        else:
            read, verdict = "<i>Not reviewed.</i>", ""
        mates = sorted(((v, o) for lane, row in deckmates(set_code).items() for o, v in (row.get(name) or {}).items()), reverse=True)
        best = {}
        for v, o in mates:
            best.setdefault(o, v)
        top = sorted(best.items(), key=lambda kv: -kv[1])[:8]
        if top:
            verdict += ("<div class='en'><b>Seen in the same top-player decks</b> (times more often than chance): "
                        + ", ".join(f"{e(o)} {math.exp(v):.1f}x" for o, v in top) + ".</div>")
        text = e(c.get("oracle_text") or "").replace("\n", "<br>")
        img = f"<img src='{e(c['image'][0] if isinstance(c.get('image'), list) else c.get('image') or '')}'>" if c.get("image") else ""
        out.append(f"<section>{img}<div><h2>{i}. {e(name)} <span>{e(c.get('mana_cost') or '')}</span></h2>"
                   f"<div class='meta'>{e(' '.join(c.get('types') or []))} · {e(c.get('rarity') or '')} · GIH WR {a['gihwr']:.1f}% ({a['samples']:,} games in hand)</div>"
                   f"<p class='text'>{text}</p><p>{read}</p>{verdict}<p class='fix'>Your correction:</p></div></section>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Top Cards Review</title><style>
body{{font:11px/1.4 system-ui,sans-serif;color:#111;margin:18px}} h1{{font-size:18px;margin:0 0 4px}} .lede{{color:#444;max-width:760px}}
section{{display:flex;gap:12px;border-top:1px solid #ccc;padding:10px 0;break-inside:avoid}} img{{width:130px;height:auto;border-radius:6px;flex:none}}
h2{{font-size:13px;margin:0}} h2 span{{font-weight:400;color:#555}} .meta{{color:#555;margin:2px 0 4px}} .text{{background:#f4f4f2;padding:6px 8px;border-radius:4px}}
code{{font-size:10px;background:#eee;padding:0 3px}} .v{{padding:4px 8px;border-left:3px solid #999;margin:4px 0}} .v.confirmed{{border-color:#1baf7a}}
.v.leaning-yes{{border-color:#eda100}} .v.opposite{{border-color:#e34948}} .en{{margin:4px 0}} .fix{{color:#999;border-bottom:1px dotted #bbb;padding-bottom:14px}}
</style></head><body><h1>Top {top} {set_code} cards: how Claude reads them</h1><p class="lede">By All Decks GIH WR. For each card: its rules text, how Claude read it (standalone, or what it needs to shine), the rule for finding its
enablers, and a few of the cards the rule picks up. Mark where the reading or the rule is wrong.</p>
{''.join(out)}</body></html>"""
    path = ROOT / "docs" / f"top-cards-{set_code}.html"
    path.write_text(page)
    return path


if __name__ == "__main__":
    main()
