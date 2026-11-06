"""Synergy measured inside a deck: same player, same 40 cards, different draws.

    uv run python tools/within_deck.py FRA [--placebos 20] [--boot 200]

For a payoff (a reviewed top card from manual.json card_wants, or a theme's payoffs from fitted synergy) and its
enablers: among games where the payoff was in hand (opening hand or drawn), does having more of its enablers in hand
go with winning more, compared only with other games of the same deck (draft id), holding the number of cards and of
lands seen fixed? Deck quality and player skill cancel out; drawing a spell instead of a land does not count as synergy.

Each test also runs on --placebos random sets of ordinary spells the same size, drawn from cards those decks hold as
often as the enablers; the effect is judged against that spread (empirical z), with a bootstrap over decks for its own
standard error. Writes docs/within-deck-SET.md and the within_deck section of the set's fitted.json.
"""

import argparse
import csv
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config, needs  # noqa: E402


def load(set_code):
    """[(draft, {card: copies in hand}, {card in deck}, won)] for every game."""
    games = []
    with gzip.open(DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.PremierDraft.csv.gz", "rt") as f:
        r = csv.reader(f)
        h = next(r)
        ix = {k: i for i, k in enumerate(h)}
        names = [k[len("deck_"):] for k in h if k.startswith("deck_")]
        cols = [(n, ix[f"opening_hand_{n}"], ix[f"drawn_{n}"], ix[f"deck_{n}"]) for n in names]
        for row in r:
            hand, deck = {}, set()
            for n, o, d, k in cols:
                if row[k] not in ("", "0"):
                    deck.add(n)
                    c = (int(float(row[o])) if row[o] not in ("", "0") else 0) + (int(float(row[d])) if row[d] not in ("", "0") else 0)
                    if c:
                        hand[n] = c
            games.append((row[ix["draft_id"]], hand, deck, row[ix["won"]] == "True"))
    return games


def effect(rows, boot=0, rng=None):
    """Within-deck slope of winning on enablers in hand (cards and lands in hand held fixed), win-rate points per
    enabler; with boot > 0 also a bootstrap-over-decks standard error."""
    by = defaultdict(list)
    for d, e, t, nl, w in rows:
        by[d].append((e, t, nl, w))
    blocks = []
    for gs in by.values():
        if len(gs) >= 2:
            a = np.array(gs, float)
            blocks.append(a - a.mean(0))
    if len(blocks) < 50:
        return None, None, 0

    def fit(bs):
        a = np.vstack(bs)
        X, y = a[:, :3], a[:, 3]
        if X[:, 0].std() == 0:
            return None
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 100 * b[0]

    est = fit(blocks)
    if est is None or not boot:
        return est, None, len(blocks)
    draws = [fit([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(boot)]
    draws = [x for x in draws if x is not None]
    return est, float(np.std(draws)), len(blocks)


def rows_for(games, payoffs, enablers, lands):
    out = []
    for d, hand, _, w in games:
        if any(p in hand for p in payoffs):
            out.append((d, sum(k for n, k in hand.items() if n in enablers), sum(hand.values()),
                        sum(k for n, k in hand.items() if n in lands), w))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--placebos", type=int, default=20)
    p.add_argument("--boot", type=int, default=200)
    a = p.parse_args(argv)
    data = json.loads((DATA_DIR / "Sets" / f"{a.set}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    lands = {c["name"] for c in data.values() if "Land" in (c.get("types") or [])}
    tests = [(card, {card}, set(en)) for card, en in needs.synergy_lists(a.set, every=True)]
    for name, t in ((config.fitted(a.set, "synergy") or {}).get("themes") or {}).items():
        if t.get("enablers") and t.get("payoffs"):
            tests.append((f"theme: {name}", set(t["payoffs"]), set(t["enablers"]) - set(t["payoffs"])))
    print(f"loading games ...", flush=True)
    games = load(a.set)
    print(f"{len(games)} games; {len(tests)} tests", flush=True)
    rng = np.random.default_rng(0)
    pick = random.Random(0)
    results, seen = [], Counter()
    for label, payoffs, enablers in tests:
        enablers = {e for e in enablers if e not in lands}
        if not enablers:
            continue
        sub = [g for g in games if any(x in g[1] for x in payoffs)]  # games with the payoff in hand, once per test
        rows = rows_for(sub, payoffs, enablers, lands)
        est, se, decks = effect(rows, a.boot, rng)
        if est is None:
            continue
        # placebo pool: ordinary spells those decks hold, as often as the enablers are held
        held = Counter(n for d, hand, deck, w in sub for n in deck)
        en_rate = np.mean([held[e] for e in enablers]) if enablers else 0
        pool = [n for n, k in held.items() if n not in enablers and n not in payoffs and n not in lands and 0.5 * en_rate <= k <= 2 * en_rate]
        nulls = []
        for _ in range(a.placebos):
            if len(pool) < len(enablers):
                break
            fake = set(pick.sample(pool, len(enablers)))
            x, _, _ = effect(rows_for(sub, payoffs, fake, lands))
            if x is not None:
                nulls.append(x)
        mu, sd = (float(np.mean(nulls)), float(np.std(nulls))) if len(nulls) >= 5 else (0.0, 0.0)
        # The same enablers in games where the payoff is in the deck but not in hand: their value on their own.
        alone = [(d, sum(k for n, k in hand.items() if n in enablers), sum(hand.values()), sum(k for n, k in hand.items() if n in lands), w)
                 for d, hand, deck, w in games if any(x in deck for x in payoffs) and not any(x in hand for x in payoffs)]
        est0, se0, _ = effect(alone, a.boot, rng)
        z = (est - mu) / np.sqrt((se or 0) ** 2 + sd ** 2) if (se or sd) else 0.0
        seen[label.split(" (")[0]] += 1
        results.append({"test": label if seen[label.split(" (")[0]] == 1 else f"{label} (list {seen[label]})", "effect": round(est, 2),
                        "se": round(se or 0, 2), "placebo_mean": round(mu, 2), "placebo_sd": round(sd, 2), "z_vs_placebo": round(float(z), 1),
                        "decks": decks, "enablers": len(enablers),
                        "without_payoff": None if est0 is None else round(est0, 2), "without_se": round(se0 or 0, 2),
                        "synergy": None if est0 is None else round(est - est0, 2),
                        "synergy_z": None if est0 is None else round(float((est - est0) / np.sqrt((se or 0) ** 2 + (se0 or 0) ** 2)), 1)})
        r = results[-1]
        print(f"{r['test'][:44]:44} {r['effect']:+6.2f} (SE {r['se']:.2f})  placebo {r['placebo_mean']:+.2f} ± {r['placebo_sd']:.2f}  z {r['z_vs_placebo']:+.1f}"
              f"  without payoff {r['without_payoff']}  synergy {r['synergy']} (z {r['synergy_z']})  {decks} decks", flush=True)
    results.sort(key=lambda r: -abs(r["synergy_z"] or 0))
    config.write_fitted(a.set, "within_deck", {"tests": results},
                        f"tools/within_deck.py: within-deck synergy, {a.placebos} placebo sets, {a.boot} deck bootstraps")
    lines = [f"# Within-deck synergy, {a.set}", "",
             "Same deck, different draws: win-rate points per enabler in hand when the payoff is in hand, cards and lands in hand",
             f"held fixed; judged against {a.placebos} random sets of ordinary spells the same decks hold as often (z vs placebo).", "",
             "Synergy = the effect with the payoff in hand minus the same enablers' effect when the payoff is in the deck but not",
             "drawn (their value on their own).", "",
             "| test | with payoff | placebo | z vs placebo | without payoff | synergy | z | decks |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['test']} | {r['effect']:+.2f} ± {r['se']:.2f} | {r['placebo_mean']:+.2f} ± {r['placebo_sd']:.2f} | {r['z_vs_placebo']:+.1f} | "
              f"{r['without_payoff']} | {r['synergy']} | {r['synergy_z']} | {r['decks']} |" for r in results]
    (ROOT / "docs" / f"within-deck-{a.set}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote docs/within-deck-{a.set}.md")


if __name__ == "__main__":
    main()
