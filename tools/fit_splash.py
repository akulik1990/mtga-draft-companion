"""When top players splash, and when they go three colours: logistic rules on the pool, fitted from their decks.

    uv run python tools/fit_splash.py FRA        # writes the splash and three sections of the set's fitted.json

From 17Lands' public game data (deck + sideboard = pool) for 60%+ players' two-colour builds: did the deck splash
(17Lands' splash_colors)? Inputs per pool and pair (companion/advisor_v2/deckbuilder.splash_features): how much the
best single-pip off-colour spell beats the 23rd on-colour one, fixing sources for its colour, on-colour playables.
Fitted on draft-id hash buckets outside tools/builder_check.py's sample; 20% of those held out for the accuracy line.
The builder splashes only where the rule gives at least 50%.

Three colours: over their two- and three-colour builds, was the deck three main colours? Inputs (deckbuilder.three_features,
on the core pair: the pair casting most of their deck): spells a third colour adds above the pair's 23rd, fixing for it,
the pair's playables. The builder opens three-colour builds where this gives at least 50%.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.deckbuilder import SPLASH_INPUTS, THREE_INPUTS, splash_features, three_features  # noqa: E402
from companion.advisor_v2.lanes import LANES, castable  # noqa: E402
from tools.builder_check import CHECK, sample  # noqa: E402

NAMES = ["edge", "fixing", "playables"]


def fit(X, y):
    """Logistic fit on standardised inputs: (weights, mu, sd, held-out accuracy on every fifth row)."""
    X, y = np.array(X, dtype=float), np.array(y)
    mu, sd = X.mean(0), X.std(0)
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    train = np.arange(len(y)) % 5 != 0

    def nll(w, Z, y):
        q = np.clip(1 / (1 + np.exp(-Z @ w)), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)) + 1e-3 * w[1:] @ w[1:]

    w = minimize(nll, np.zeros(Z.shape[1]), args=(Z[train], y[train])).x
    held = 1 / (1 + np.exp(-Z[~train] @ w))
    acc = float(((held > 0.5) == (y[~train] > 0.5)).mean())
    w = minimize(nll, w, args=(Z, y)).x  # final fit on everything
    return w, mu, sd, acc


def record(set_code, section, names, w, mu, sd, acc, y, how, extra=None):
    y = np.array(y)
    print(f"{section}: {len(y)} pools, rate {y.mean():.3f}; held-out accuracy {acc:.3f} (majority {max(y.mean(), 1 - y.mean()):.3f})")
    print("  per SD:", ", ".join(f"{n} {v:+.2f}" for n, v in zip(["intercept"] + names, w)))
    rule = {"inputs": names, "w": [round(float(v), 4) for v in w], "mu": [round(float(v), 4) for v in mu],
            "sd": [round(float(v), 4) for v in sd], "n": int(len(y)), "rate": round(float(y.mean()), 3), "heldout_accuracy": round(acc, 3),
            **(extra or {})}
    path = config.write_fitted(set_code, section, rule, how)
    print(f"  wrote {section} to {path.relative_to(ROOT)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--n", type=int, default=8000)
    a = p.parse_args(argv)

    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by = {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by.setdefault(n, c)
    X, y, X3, y3, lanes = [], [], [], [], []
    for lane, splash, deck, pool in sample(a.set, a.n, buckets=(CHECK[1], 1000), colours=(2, 3)):
        cards = [by[n] for n, k in pool.items() if n in by for _ in range(k)]
        spells = [by[n] for n in deck if n in by and "Land" not in (by[n].get("types") or [])]
        core = lane if len(lane) == 2 else max(LANES, key=lambda pair: sum(castable(c, pair) for c in spells))
        X3.append(three_features(cards, core))
        y3.append(1.0 if len(lane) == 3 else 0.0)
        if len(lane) == 2:
            X.append(splash_features(cards, lane))
            y.append(1.0 if splash else 0.0)
            lanes.append(lane)
    wide = ["edge", "fixing", "playables", "above", "duals", "depth"]
    card = wide + ["removal", "cmc"]  # the best splash card itself
    # How often top players splash in each pair (shrunk toward the overall rate with 20 decks).
    from collections import Counter
    n_l, s_l = Counter(lanes), Counter(l for l, v in zip(lanes, y) if v)
    overall = sum(y) / len(y)
    pair_rate = {l: round((s_l[l] + 20 * overall) / (n_l[l] + 20), 3) for l in n_l}
    pick = lambda names: [[{**dict(zip(SPLASH_INPUTS, x)), "pair": pair_rate[l]}[k] for k in names] for x, l in zip(X, lanes)]
    paired = card + ["pair"]
    base, ext, ext2, ext3 = fit(pick(NAMES), y), fit(pick(wide), y), fit(pick(card), y), fit(pick(paired), y)
    print(f"splash rule held-out accuracy: {NAMES} {base[3]:.3f}, {wide} {ext[3]:.3f}, + removal, cmc {ext2[3]:.3f}, + pair rate {ext3[3]:.3f}")
    print("  pair splash rates:", ", ".join(f"{l} {r:.2f}" for l, r in sorted(pair_rate.items(), key=lambda kv: -kv[1])))
    names, best = (wide, ext) if ext[3] >= base[3] + 0.005 else (NAMES, base)
    if ext2[3] >= best[3]:  # level or better: the card inputs stay in
        names, best = card, ext2
    if ext3[3] >= best[3]:
        names, best = paired, ext3
    record(a.set, "splash", names, *best, y,
           f"tools/fit_splash.py: when 60%+ players splash (17Lands game data), logistic on {', '.join(names)}",
           extra={"pair_rate": pair_rate} if "pair" in names else None)
    pick3 = lambda names: [[dict(zip(THREE_INPUTS, x))[k] for k in names] for x in X3]
    n3, w3 = ["adds", "fixing", "playables"], ["adds", "fixing", "playables", "duals", "depth"]
    base3, ext3 = fit(pick3(n3), y3), fit(pick3(w3), y3)
    print(f"three rule held-out accuracy: {n3} {base3[3]:.3f}, {w3} {ext3[3]:.3f}")
    names3, best3 = (w3, ext3) if ext3[3] >= base3[3] + 0.005 else (n3, base3)
    record(a.set, "three", names3, *best3, y3,
           f"tools/fit_splash.py: when 60%+ players play three main colours (17Lands game data), logistic on {', '.join(names3)}")


if __name__ == "__main__":
    main()
