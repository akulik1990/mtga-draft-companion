"""Do strategy claims hold up in real games? Deck features tested against wins, beyond the deck model.

    uv run python tools/deck_features.py FRA

The deck model (tools/deck_model.py) gives every card its own coefficient, so a fixed property of a card
("costs 1UU") is already inside that card's number and cannot be separated. What can be tested is a
card's effect depending on the deck around it: a cheap double-pip spell in a deck short of that colour,
fixing in a deck that splashes. The deck model is fitted on 80% of drafts; its prediction is the offset
for the other 20%, where each feature's extra effect is estimated with a standard error.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from tools.deck_model import SHAPE, card_facts, fit_logistic, read, standardise  # noqa: E402

BASICS = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}
FEATURES = {
    "dp_short": "cheap (≤4) double-pip spells whose colour has < 8 sources",
    "dp_splash": "double-pip spells whose colour is a splash colour",
    "dp_multi": "double-pip spells in decks that splash or run 3+ main colours",
    "dual_splash": "lands making 2+ colours, in splash decks",
    "cycler_splash": "basic landcyclers, in splash decks",
    "rock_splash": "any-colour mana spells, in splash decks",
    "ur_other": "Izzet (UR main) decks: spells that are not interaction (removal, counter, bounce), card draw/selection or a top-30 card (video: Izzet wants only those)",
}


def extra_facts(set_code):
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    wr = lambda c: ((c.get("deck_colors") or {}).get("All Decks") or {}).get("gihwr") or 0.0
    top30 = {c["name"] for c in sorted(data.values(), key=lambda c: -wr(c))[:30]}
    interact = re.compile(r"counter target|return target [^.]*to its owner's hand|put target [^.]*on (the )?(top|bottom)", re.I)
    out = {}
    for c in data.values():
        front = (c.get("mana_cost") or "").split("//")[0]
        singles = [p for p in re.findall(r"\{([WUBRG])\}", front)]
        double = next((col for col in "WUBRG" if singles.count(col) >= 2), None)
        land = "Land" in (c.get("types") or [])
        out.setdefault(c["name"], {
            "cmc": int(c.get("cmc") or 0), "double": double, "land": land,
            "makes": [x for x in c.get("colors") or [] if x in "WUBRG"] if land else [],
            "cycler": "Basic landcycling" in (c.get("keywords") or []),
            "rock": not land and "mana of any color" in (c.get("oracle_text") or ""),
            "plan": land or c["name"] in top30 or bool({"removal", "card_advantage"} & set(config.tags(c, set_code)))
                    or bool(interact.search(c.get("oracle_text") or "")),
        })
    return out


def deck_features(counts, facts, main, splash):
    sources = dict.fromkeys("WUBRG", 0)
    for name, n in counts.items():
        if name in BASICS:
            sources[BASICS[name]] += n
        elif facts.get(name, {}).get("land"):
            for col in facts[name]["makes"]:
                sources[col] += n
    multi = bool(splash) or len(main) >= 3
    f = dict.fromkeys(FEATURES, 0.0)
    for name, n in counts.items():
        x = facts.get(name)
        if not x:
            continue
        if x["double"]:
            f["dp_short"] += n * (x["cmc"] <= 4 and sources[x["double"]] < 8)
            f["dp_splash"] += n * (x["double"] in splash)
            f["dp_multi"] += n * multi
        if sorted(main) == ["R", "U"]:
            f["ur_other"] += n * (not x["plan"])
        if splash:
            f["dual_splash"] += n * (x["land"] and len(x["makes"]) >= 2)
            f["cycler_splash"] += n * x["cycler"]
            f["rock_splash"] += n * x["rock"]
    return [f[k] for k in FEATURES]


def read_features(set_code, facts):
    path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz"
    rows = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        deck_cols = [(i, h[len("deck_"):]) for i, h in enumerate(header) if h.startswith("deck_")]
        for row in reader:
            counts = {name: int(float(row[i])) for i, name in deck_cols if row[i] and row[i] != "0"}
            rows.append(deck_features(counts, facts, row[col["main_colors"]], row[col["splash_colors"]]))
    return np.array(rows, np.float32)


def offset_logit(X, y, offset, iters=25):
    """Logistic regression of y on X with a fixed offset; (coefficients, standard errors), intercept first."""
    X1 = np.column_stack([np.ones(len(y)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + X1 @ w)))
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, X1.T @ (y - p))
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--l2", type=float, default=None, help="deck model penalty (default: the fitted model's)")
    a = p.parse_args(argv)

    facts = card_facts(a.set)
    names, C, ctx, shp, y, test = read(a.set, facts)
    F = read_features(a.set, extra_facts(a.set))
    l2 = a.l2 or config.fitted(a.set, "deckmodel").get("l2") or 30.0
    mu, sd = standardise(shp[~test])
    S = (shp - mu) / sd
    S2 = np.column_stack([S, S[:, SHAPE.index("creatures")] ** 2, S[:, SHAPE.index("lands")] ** 2, S[:, SHAPE.index("cmc2")] ** 2])
    full = sparse.hstack([sparse.csr_matrix(ctx), sparse.csr_matrix(S2), C]).tocsr()
    w = fit_logistic(full[~test], y[~test], l2)
    offset = full[test] @ w[1:] + w[0]
    coef, se = offset_logit(F[test], y[test], offset)
    base = float((1 / (1 + np.exp(-offset))).mean())
    print(f"{len(y)} games; deck model fitted on {int((~test).sum())}, features estimated on {int(test.sum())} held-out games (l2 {l2})")
    print(f"{'feature':14} {'decks with it':>13} {'mean when >0':>12} {'per copy, WR pts':>17} {'(SE)':>7}  z     what")
    Ft = F[test]
    for i, (k, desc) in enumerate(FEATURES.items()):
        b, s = coef[i + 1], se[i + 1]
        has = Ft[:, i] > 0
        pts = 100 * b * base * (1 - base)  # log-odds to win-rate points near the average
        print(f"{k:14} {100 * has.mean():12.1f}% {Ft[has, i].mean() if has.any() else 0:12.2f} {pts:+17.2f} {100 * s * base * (1 - base):7.2f}  {b / s:+5.1f}  {desc}")


if __name__ == "__main__":
    main()
